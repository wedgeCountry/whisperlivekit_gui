"""EngineManager — owns the asyncio event loop, WhisperLiveKit model lifecycle,
and sounddevice audio stream.

All communication back to the UI is via injected callbacks, keeping this module
free of any tkinter dependency.

Thread model
------------
* UI thread   — calls start(), reload(), start_session(), stop_session(), shutdown()
* Async thread — runs the asyncio event loop; _load_engine and _run_session live here
* SD callback  — fires from a sounddevice thread; forwards PCM via run_coroutine_threadsafe
"""

import asyncio
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np
from whisperlivekit import diarization

from transcribe_app.config import CHUNK_SECONDS, CHANNELS, DTYPE, GPU, IS_WINDOWS, LANGUAGE_OPTS, SAMPLE_RATE
from transcribe_app.i18n import t

# WhisperLiveKit types are imported lazily inside the background thread so that
# PortAudio / CUDA initialisation does not block the Tk window from opening.
_AudioProcessor:       object = None  # type: ignore[assignment]
_TranscriptionEngine:  object = None  # type: ignore[assignment]
_WhisperLiveKitConfig: object = None  # type: ignore[assignment]


# ── Model cache check ──────────────────────────────────────────────────────────

def _is_model_cached(model_size: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = [
        hub / f"models--Systran--faster-whisper-{model_size}",
        hub / f"models--mobiuslabsgmbh--faster-whisper-{model_size}",
    ]
    try:
        return any(
            (d / "snapshots").is_dir() and any((d / "snapshots").iterdir())
            for d in candidates
        )
    except OSError:
        return False


def loading_status(model_size: str, lang: str, use_gpu: bool = GPU) -> str:
    device = "GPU" if use_gpu else "CPU"
    key    = "status.loading" if _is_model_cached(model_size) else "status.downloading"
    return t(key, model=model_size, lang=lang, device=device)


# ── tqdm progress capture ──────────────────────────────────────────────────────

class _TqdmCapture:
    """File-like object that intercepts tqdm progress lines and forwards them
    to a status callback.  Assigned to sys.stderr during model downloads so
    that the tqdm bar appears in the UI status bar instead of the terminal.
    """

    def __init__(self, on_status: Callable[[str], None], original: object) -> None:
        self._on_status = on_status
        self._original  = original

    def write(self, s: str) -> int:
        # tqdm writes lines like "\r82%|████| 376M/461M [00:13<00:12, 7.32MiB/s]"
        stripped = s.strip("\r\n ")
        if stripped and "%|" in stripped:
            self._on_status(stripped)
        elif stripped:
            self._original.write(s)
        return len(s)

    def flush(self) -> None:
        self._original.flush()

    def isatty(self) -> bool:
        return True  # keep tqdm in \r-refresh mode so each write is a full bar update


# ── EngineManager ──────────────────────────────────────────────────────────────

class EngineManager:
    def __init__(
        self,
        on_status:   Callable[[str], None],
        on_ready:    Callable[[], None],
        on_update:   Callable[[str, str], None],
        on_finalise: Callable[[str], None],
        on_open_mic: Callable[[], None],
    ) -> None:
        """
        Parameters
        ----------
        on_status   Called with a status-bar string whenever engine state changes.
        on_ready    Called once the engine has finished loading (enables UI controls).
        on_update   Called with (committed_text, buffer_text) during a live session.
        on_finalise Called with the final committed text when a session ends.
        on_open_mic Called (from the asyncio thread) to request the UI thread open
                    the mic stream via root.after(0, self.open_mic_stream).
        """
        self._on_status   = on_status
        self._on_ready    = on_ready
        self._on_update   = on_update
        self._on_finalise = on_finalise
        self._on_open_mic = on_open_mic

        self._loop:      asyncio.AbstractEventLoop | None = None
        self._engine:    object | None = None
        self._processor: object | None = None
        self._stream:    object | None = None
        self._recording: bool          = False
        self._lang:      str           = ""
        self._use_gpu:   bool          = GPU   # updated by start()/reload()
        self.mic_gain:   float         = 1.0   # linear amplitude multiplier; set by UI

        # Load-cancellation state.  _load_gen is incremented on every reload()
        # call; a running _load_engine checks my_gen against it to detect that
        # it has been superseded.  _load_lock serialises the actual model
        # construction so the WhisperLiveKit singleton is never touched
        # concurrently.  Both are initialised lazily on first use inside the
        # event loop.
        self._load_gen:  int                    = 0
        self._load_lock: asyncio.Lock | None    = None

    # ── Startup ───────────────────────────────────────────────────────────────

    def start(self, lang: str, prompt: str, speed: str = "normal", compute_device: str = "cuda" if GPU else "cpu") -> None:
        """Spin up the background asyncio thread and load the initial model."""
        self._use_gpu = (compute_device == "cuda") and GPU

        def _run() -> None:
            import socket  # noqa: PLC0415
            # Windows process priority ─────────────────────────────────────────
            # Live transcription is extremely latency-sensitive.  The Windows
            # scheduler frequently preempts CPU-heavy background processes to
            # service system tasks (antivirus, telemetry, UI redraws).  Even a
            # 10–20 ms stall during the model's INT8 matrix multiplications can
            # cause the audio buffer to drift, losing a short fragment of speech.
            # Whisper is autoregressive: it uses its own previous output as
            # context for the next token.  A single wrong token caused by missing
            # audio can cascade into a "hallucination loop" where the most likely
            # next token keeps repeating the last word.
            #
            # Raising to HIGH_PRIORITY_CLASS tells the Windows scheduler to
            # prefer this process over normal-priority tasks without starving the
            # UI thread (which remains responsive because Tk runs in a separate
            # thread at normal priority).  This is the programmatic equivalent of
            # the manual Task-Manager workaround.
            if IS_WINDOWS:
                try:
                    import ctypes  # noqa: PLC0415
                    HIGH_PRIORITY_CLASS = 0x0080
                    ctypes.windll.kernel32.SetPriorityClass(  # type: ignore[attr-defined]
                        ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                        HIGH_PRIORITY_CLASS,
                    )
                except Exception:
                    pass
            # Set a generous default socket timeout so that model downloads from
            # openaipublic.azureedge.net or Hugging Face never hang forever in a
            # frozen exe.  The timeout is reset to None once the loop is running.
            socket.setdefaulttimeout(600)
            try:
                global _AudioProcessor, _TranscriptionEngine, _WhisperLiveKitConfig
                from whisperlivekit import AudioProcessor as AP, TranscriptionEngine as TE  # noqa: PLC0415
                from whisperlivekit.config import WhisperLiveKitConfig as WLC               # noqa: PLC0415
                _AudioProcessor       = AP
                _TranscriptionEngine  = TE
                _WhisperLiveKitConfig = WLC

                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.run_until_complete(self._load_engine(lang, prompt, speed))
                socket.setdefaulttimeout(None)  # restore: don't affect audio streaming
                self._loop.run_forever()
            except Exception as exc:  # noqa: BLE001
                # Surface any unhandled error instead of silently killing the thread
                # (which would leave the UI stuck on the loading screen forever).
                self._on_status(t("status.critical_error", exc=repr(exc)))
                self._on_ready()

        threading.Thread(target=_run, daemon=True).start()

    # ── Engine lifecycle ──────────────────────────────────────────────────────

    def reload(self, lang: str, prompt: str, speed: str = "normal", compute_device: str | None = None) -> None:
        """Hot-reload the model (e.g. after a language/speed change). UI-thread safe."""
        if compute_device is not None:
            self._use_gpu = (compute_device == "cuda") and GPU
        asyncio.run_coroutine_threadsafe(self._load_engine(lang, prompt, speed), self._loop)

    async def _load_engine(self, lang: str, prompt: str, speed: str = "normal") -> None:
        from transcribe_app.config import get_model_size  # noqa: PLC0415

        # Snapshot settings that belong to this load request.
        use_gpu    = self._use_gpu
        opts       = LANGUAGE_OPTS[lang]
        model_size = get_model_size(lang, speed, use_gpu)
        fallback   = opts["fallback_model_size"]
        lan        = opts["lan"]

        # Stamp this request.  Any later reload() will increment _load_gen so
        # we know we've been superseded.
        self._load_gen += 1
        my_gen = self._load_gen

        # Update the status bar immediately so the UI reflects the new
        # selection even while waiting for the lock.
        self._on_status(loading_status(model_size, lang, use_gpu))

        # Lazy-create the lock inside the event loop.
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()

        async with self._load_lock:
            # Re-check after acquiring: a later reload() may have come in
            # while we were waiting.
            if my_gen != self._load_gen:
                return

            self._lang = lang

            # Reset the WhisperLiveKit singleton before each fresh load.
            _TranscriptionEngine._instance    = None
            _TranscriptionEngine._initialized = False

            def _make_cfg(size: str):
                win_cpu = IS_WINDOWS and not use_gpu
                # decoder_type / beams ────────────────────────────────────────
                # Greedy decoding always picks the single highest-probability
                # token.  This is fast, but if a tiny floating-point difference
                # (caused by Windows using a different math library path than
                # Linux) pushes the wrong token to the top, that token is fed
                # back as context for the *next* prediction.  On Windows CPU
                # these small rounding errors can cascade into a "repetition
                # loop" where the same word keeps winning.
                #
                # Beam search explores `beams` candidate sequences in parallel
                # and picks the overall best path, so a single bad token can be
                # recovered in the next step.  The cost is higher CPU usage;
                # `beams=5` is a standard trade-off that stays fast enough for
                # real-time use on a modern CPU.
                #
                # min_chunk_size ──────────────────────────────────────────────
                # WhisperLiveKit will not start an inference pass until it has
                # accumulated at least `min_chunk_size` seconds of audio.  On
                # Windows, Windows Audio Processing Objects (APO) may apply
                # driver-level noise suppression or AGC that distorts very short
                # clips in ways that confuse the VAD.  A 0.50 s minimum forces
                # the model to wait for a more complete utterance before
                # committing, reducing false triggers during pauses.
                return _WhisperLiveKitConfig(
                    pcm_input=True, vac=True,
                    model_size=size, lan=lan,
                    decoder_type="beam" if (use_gpu or win_cpu) else "greedy",
                    beams=5 if win_cpu else 1,
                    min_chunk_size=0.50 if win_cpu else 0.30,
                    audio_max_len=12.0, audio_min_len=0.20,
                    direct_english_translation=False,  # diarization=True,
                    static_init_prompt=prompt if prompt.strip() else None,
                )

            # Redirect stderr so tqdm download bars appear in the status bar.
            # We do this for every uncached model; a second redirect for the
            # fallback is set up inside the except block if needed.
            _orig_stderr = None

            def _maybe_capture(size: str) -> None:
                nonlocal _orig_stderr
                if _orig_stderr is None and not _is_model_cached(size):
                    _orig_stderr = sys.stderr
                    sys.stderr   = _TqdmCapture(self._on_status, _orig_stderr)

            new_engine = None
            try:
                _maybe_capture(model_size)
                # Run the blocking constructor in a thread so the event loop
                # stays responsive and can accept new reload() requests.
                new_engine = await self._loop.run_in_executor(
                    None, lambda: _TranscriptionEngine(config=_make_cfg(model_size))
                )
            except Exception as exc:
                if my_gen != self._load_gen:
                    return  # superseded during download/load — silently discard
                if use_gpu and model_size != fallback:
                    self._on_status(t(
                        "status.error_fallback",
                        exc_type=exc.__class__.__name__,
                        status=loading_status(fallback, lang, use_gpu),
                    ))
                    _TranscriptionEngine._instance    = None
                    _TranscriptionEngine._initialized = False
                    try:
                        _maybe_capture(fallback)
                        new_engine = await self._loop.run_in_executor(
                            None, lambda: _TranscriptionEngine(config=_make_cfg(fallback))
                        )
                    except Exception as fallback_exc:
                        if my_gen != self._load_gen:
                            return
                        self._on_status(t("status.error", exc=fallback_exc))
                        self._on_ready()
                        return
                else:
                    self._on_status(t("status.error", exc=exc))
                    self._on_ready()
                    return
            finally:
                if _orig_stderr is not None:
                    sys.stderr = _orig_stderr

            # Final check: another reload() may have arrived while the executor
            # thread was running.
            if my_gen != self._load_gen:
                return

            self._engine = new_engine
            self._on_status(t("status.ready", lang=lang))
            self._on_ready()

    # ── Recording session ─────────────────────────────────────────────────────

    def start_session(self) -> None:
        """Begin a transcription session. Call from the UI thread."""
        self._recording = True
        asyncio.run_coroutine_threadsafe(self._run_session(), self._loop)

    async def _run_session(self) -> None:
        self._processor = _AudioProcessor(transcription_engine=self._engine)
        results_gen = await self._processor.create_tasks()
        # Signal the UI thread to open the mic stream via root.after(0, …)
        self._on_open_mic()

        committed_text = ""
        async for front_data in results_gen:
            lines_text = " ".join(seg.text for seg in front_data.lines if seg.text)
            if lines_text and lines_text != committed_text:
                committed_text = lines_text
            self._on_update(committed_text, front_data.buffer_transcription or "")

        self._on_finalise(committed_text)
        self._on_status(t("status.ready", lang=self._lang))

    def open_mic_stream(self, device: int | None = None) -> None:
        """Open the sounddevice InputStream. Must be called from the UI thread
        (scheduled via root.after) so that self._stream is only ever touched on
        the UI thread, matching stop_session()."""
        import sounddevice as sd  # noqa: PLC0415

        blocksize = int(SAMPLE_RATE * CHUNK_SECONDS)
        loop      = self._loop
        processor = self._processor

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if not self._recording:
                return
            samples = indata.astype(np.float32)
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples /= peak
            gain = self.mic_gain
            if gain != 1.0:
                samples *= gain
            audio = (samples * 32767).clip(-32768, 32767).astype("int16")
            asyncio.run_coroutine_threadsafe(
                processor.process_audio(audio.tobytes()), loop
            )

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
            blocksize=blocksize, callback=_callback,
            device=device,
        )
        self._stream.start()

    def stop_session(self) -> None:
        """Stop recording and flush the processor. Call from the UI thread."""
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._processor is not None:
            asyncio.run_coroutine_threadsafe(
                self._processor.process_audio(None), self._loop
            )

    def dispatch(self, coro) -> "asyncio.Future":
        """Schedule an arbitrary coroutine on the background loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release GPU memory and stop the background thread. Call from WM_DELETE_WINDOW."""
        self.stop_session()
        self._processor = None
        if self._engine is not None and _TranscriptionEngine is not None:
            _TranscriptionEngine._instance = None
            _TranscriptionEngine._initialized = False
            self._engine = None
        if self._use_gpu:
            try:
                import torch  # noqa: PLC0415
                torch.cuda.empty_cache()
            except Exception:
                pass
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
