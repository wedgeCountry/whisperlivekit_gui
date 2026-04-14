"""EngineManager — owns the asyncio event loop, WhisperLiveKit model lifecycle,
and sounddevice audio stream.

All communication back to the UI is via injected callbacks, keeping this module
free of any tkinter dependency.

Thread model
------------
* UI thread    — calls start(), reload(), start_session(), stop_session(), shutdown()
* Async thread — runs the asyncio event loop; _load_engine and _run_session live here
* SD callback  — fires from a sounddevice thread; forwards PCM via run_coroutine_threadsafe
"""

import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np

_log = logging.getLogger(__name__)

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


# ── Environment helpers ────────────────────────────────────────────────────────

def _pin_threads() -> None:
    """Cap OMP/OpenBLAS/MKL thread count to half the physical cores (min 2, max 8).

    CTranslate2 (the INT8 inference backend) picks up these counts from environment
    variables at import time.  The default (0 = all cores) causes contention in a
    live-streaming app where the sounddevice callback, asyncio event loop, and
    Tkinter UI all compete for CPU.
    """
    import os
    if "OMP_NUM_THREADS" in os.environ:
        return
    import multiprocessing
    threads = max(2, min(8, multiprocessing.cpu_count() // 2))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)


def _set_windows_priority() -> None:
    """Raise process priority to HIGH_PRIORITY_CLASS on Windows.

    Live transcription is latency-sensitive.  Even a 10–20 ms scheduler stall
    during INT8 matrix multiplications can cause the audio buffer to drift,
    losing a speech fragment.  Whisper is autoregressive — a single wrong token
    can cascade into a hallucination loop.  HIGH_PRIORITY_CLASS reduces that
    risk without starving the UI thread (which runs at normal priority in its
    own Tk thread).
    """
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        HIGH_PRIORITY_CLASS = 0x0080
        ctypes.windll.kernel32.SetPriorityClass(  # type: ignore[attr-defined]
            ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
            HIGH_PRIORITY_CLASS,
        )
    except Exception:
        _log.warning("Could not set Windows HIGH_PRIORITY_CLASS", exc_info=True)


def _make_warmup_audio() -> np.ndarray:
    """Return 1 s of Gaussian noise at speech-like RMS (≈ 0.07 normalised).

    Unlike silence (trivial encoder pass, 1–2 decoder steps), noise produces a
    dense mel spectrogram that exercises every attention head and drives the
    decoder through many token steps — the same code paths that run on real
    speech.  A fixed seed makes the warmup deterministic and reproducible.
    """
    rng = np.random.default_rng(42)
    return (rng.standard_normal(16000) * 0.07).astype(np.float32)


def _has_avx512() -> bool:
    """Return True if CTranslate2 can use AVX-512 float16 on CPU.

    Without AVX-512, INT8 matrix multiplications fall back to AVX2/SSE4, which
    is noticeably slower on large models.  GPU users are unaffected.
    """
    try:
        import ctranslate2  # noqa: PLC0415
        return "float16" in ctranslate2.get_supported_compute_types("cpu")
    except Exception:
        _log.warning("Could not determine AVX-512 support", exc_info=True)
        return True  # assume OK; don't show a spurious warning


# ── EngineManager ──────────────────────────────────────────────────────────────

class EngineManager:
    def __init__(
        self,
        on_status:   Callable[[str], None],
        on_ready:    Callable[[bool], None],
        on_update:   Callable[[str, str], None],
        on_finalise: Callable[[str], None],
        on_open_mic: Callable[[], None],
    ) -> None:
        """
        Parameters
        ----------
        on_status   Called with a status-bar string whenever engine state changes.
        on_ready    Called once a load attempt finishes. Receives True on success,
                    False if both primary and fallback loads failed (keep UI disabled).
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
        self._started:   bool          = False  # guards against start() re-entry

        # mic_gain is written by the UI thread and read by the sounddevice callback
        # thread.  CPython's GIL makes float reads/writes atomic in practice, but
        # treat it as a best-effort control knob, not a hard synchronisation point.
        self.mic_gain:   float         = 1.0   # linear amplitude multiplier; set by UI

        # Optional audio capture hook.  Set to a callable before open_mic_stream();
        # it receives the same processed int16 ndarray chunk (fixed-normalised,
        # gain-applied) that was sent to the transcription engine, so re-transcription
        # sees identical input to the live session.  Set to None to disable.
        self.audio_sink: "Callable[[np.ndarray], None] | None" = None

        # Load-cancellation state.
        #
        # _load_gen   — incremented on every reload(); _load_engine snapshots
        #               it at entry (my_gen) and aborts if overtaken.
        # _load_lock  — asyncio.Lock; at most one _load_engine coroutine
        #               executes the critical section at a time.
        # _load_task  — the current reload asyncio.Task; cancelled on the next
        #               reload() so waiting-for-lock tasks are dropped
        #               immediately rather than queuing up.
        # _exec_lock  — threading.Lock held for the entire singleton reset +
        #               construction inside run_in_executor.  A zombie thread
        #               from a cancelled task holds this lock until it finishes,
        #               so the next generation cannot race it on the singleton.
        self._load_gen:  int                    = 0
        self._load_lock: asyncio.Lock | None    = None
        self._load_task: asyncio.Task | None    = None
        self._exec_lock: threading.Lock         = threading.Lock()

        # Session cancellation state.
        #
        # _session_gen  — incremented on every start_session(); _run_session
        #                 snapshots it at entry and skips callbacks if superseded.
        # _session_task — the running asyncio.Task; cancelled in start_session()
        #                 so a still-draining old session never races the new one.
        self._session_gen:  int                  = 0
        self._session_task: asyncio.Task | None  = None

    # ── Startup ───────────────────────────────────────────────────────────────

    def start(self, lang: str, prompt: str, speed: str = "normal", compute_device: str = "cuda" if GPU else "cpu") -> None:
        """Spin up the background asyncio thread and load the initial model.
        Must be called at most once per instance."""
        if self._started:
            _log.warning("start() called more than once — ignored")
            return
        self._started = True
        self._use_gpu = (compute_device == "cuda") and GPU
        threading.Thread(target=self._run_loop, args=(lang, prompt, speed), daemon=True).start()

    def _run_loop(self, lang: str, prompt: str, speed: str) -> None:
        """Body of the background thread: configure env, import libs, run event loop."""
        import socket  # noqa: PLC0415
        _pin_threads()
        _set_windows_priority()
        # Set a generous default socket timeout so that model downloads from
        # openaipublic.azureedge.net or Hugging Face never hang forever in a
        # frozen exe.  Always restored in the finally block.
        socket.setdefaulttimeout(60)
        try:
            global _AudioProcessor, _TranscriptionEngine, _WhisperLiveKitConfig
            from whisperlivekit import AudioProcessor as AP, TranscriptionEngine as TE  # noqa: PLC0415
            from whisperlivekit.config import WhisperLiveKitConfig as WLC               # noqa: PLC0415
            _AudioProcessor       = AP
            _TranscriptionEngine  = TE
            _WhisperLiveKitConfig = WLC

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._load_engine(lang, prompt, speed))
            finally:
                socket.setdefaulttimeout(None)
            self._loop.run_forever()
        except Exception as exc:  # noqa: BLE001
            # Surface any unhandled error instead of silently killing the thread
            # (which would leave the UI stuck on the loading screen forever).
            self._on_status(t("status.critical_error", exc=repr(exc)))
            self._on_ready(False)

    # ── Engine lifecycle ──────────────────────────────────────────────────────

    def reload(self, lang: str, prompt: str, speed: str = "normal", compute_device: str | None = None) -> None:
        """Hot-reload the model (e.g. after a language/speed change). UI-thread safe."""
        if compute_device is not None:
            self._use_gpu = (compute_device == "cuda") and GPU

        def _schedule() -> None:
            # Cancel any task still waiting to acquire _load_lock.  A task
            # already inside run_in_executor cannot be interrupted — its thread
            # runs to completion (result discarded via _load_gen) and _exec_lock
            # prevents a concurrent singleton reset from racing it.
            if self._load_task is not None and not self._load_task.done():
                self._load_task.cancel()
            self._load_task = self._loop.create_task(self._load_engine(lang, prompt, speed))

        self._loop.call_soon_threadsafe(_schedule)

    @staticmethod
    def _make_wlk_config(size: str, lan: str, use_gpu: bool, prompt: str) -> object:
        """Build a WhisperLiveKitConfig for the given model size and language.

        Decoder and chunk-size choices are platform-specific:
        - Beam search on Windows CPU recovers from the floating-point rounding
          errors that can push the wrong token to the top and cascade into a
          repetition loop under greedy decoding.
        - A larger min_chunk_size on Windows lets APO noise suppression settle
          before committing, reducing false VAD triggers during pauses.
        """
        win_cpu = IS_WINDOWS and not use_gpu
        return _WhisperLiveKitConfig(
            pcm_input=True, vac=True,
            model_size=size, lan=lan,
            decoder_type="beam" if (use_gpu or win_cpu) else "greedy",
            beams=2 if win_cpu else 1,
            min_chunk_size=0.40 if win_cpu else 0.30,
            audio_max_len=12.0, audio_min_len=0.20,
            direct_english_translation=False,
            static_init_prompt=prompt if prompt.strip() else None,
        )

    def _build_new_engine(self, size: str, lan: str, use_gpu: bool, prompt: str, my_gen: int) -> "object | None":
        """Construct a new TranscriptionEngine inside _exec_lock.

        Holds the lock for the full singleton reset + construction so that a
        zombie thread from a cancelled task cannot race a concurrent reload.
        Returns None if this generation has been superseded.
        """
        with self._exec_lock:
            if my_gen != self._load_gen:
                return None
            _TranscriptionEngine._instance    = None
            _TranscriptionEngine._initialized = False
            return _TranscriptionEngine(config=self._make_wlk_config(size, lan, use_gpu, prompt))

    async def _try_load(
        self,
        model_size: str,
        fallback:   str,
        lan:        str,
        use_gpu:    bool,
        prompt:     str,
        my_gen:     int,
        lang:       str,
    ) -> "object | None":
        """Load model_size, falling back to fallback on GPU failure.

        Redirects stderr to capture tqdm download bars for the status bar.
        Returns the new engine, or None if superseded or permanently failed.
        """
        orig_stderr = None

        def maybe_capture_stderr(size: str) -> None:
            nonlocal orig_stderr
            if orig_stderr is None and not _is_model_cached(size):
                orig_stderr = sys.stderr
                sys.stderr  = _TqdmCapture(self._on_status, orig_stderr)

        try:
            maybe_capture_stderr(model_size)
            engine = await self._loop.run_in_executor(  # type: ignore[union-attr]
                None, lambda: self._build_new_engine(model_size, lan, use_gpu, prompt, my_gen)
            )
            if engine is None:
                return None  # superseded inside the executor

            return engine

        except Exception as exc:
            if my_gen != self._load_gen:
                return None  # superseded during download — silently discard

            can_fallback = use_gpu and model_size != fallback
            if not can_fallback:
                self._on_status(t("status.error", exc=exc))
                self._on_ready(False)
                return None

            # GPU primary failed — try CPU fallback model.
            self._on_status(t(
                "status.error_fallback",
                exc_type=exc.__class__.__name__,
                status=loading_status(fallback, lang, use_gpu),
            ))
            try:
                maybe_capture_stderr(fallback)
                engine = await self._loop.run_in_executor(  # type: ignore[union-attr]
                    None, lambda: self._build_new_engine(fallback, lan, use_gpu, prompt, my_gen)
                )
                return engine  # None if superseded, engine otherwise

            except Exception as fallback_exc:
                if my_gen != self._load_gen:
                    return None
                self._on_status(t("status.error", exc=fallback_exc))
                self._on_ready(False)
                return None

        finally:
            if orig_stderr is not None:
                sys.stderr = orig_stderr

    async def _warmup_model(self, engine: object, model_size: str, lang: str, use_gpu: bool) -> None:
        """Run a dummy inference pass to pre-populate CTranslate2 caches.

        The first real inference after a cold load is 3–5× slower because the
        INT8 GEMM kernels are not yet JIT-compiled and the OS page-cache for
        model weights is cold.  Running noise through the model once brings both
        caches to steady-state so the user's first spoken word is processed at
        full speed.  Any exception is swallowed — warmup is best-effort.
        """
        if getattr(engine, "asr", None) is None:
            return
        device = "GPU" if use_gpu else "CPU"
        self._on_status(t("status.warmup", model=model_size, lang=lang, device=device))
        try:
            await self._loop.run_in_executor(  # type: ignore[union-attr]
                None, lambda: engine.asr.transcribe(_make_warmup_audio())
            )
        except Exception:  # noqa: BLE001
            _log.warning("Warmup inference failed", exc_info=True)

    def _emit_ready_status(self, lang: str, use_gpu: bool) -> None:
        """Update the status bar with ready or AVX-512 warning, then fire on_ready."""
        if use_gpu or _has_avx512():
            self._on_status(t("status.ready", lang=lang))
        else:
            self._on_status(t("status.warn_no_avx512", lang=lang))
        self._on_ready(True)

    async def _load_engine(self, lang: str, prompt: str, speed: str = "normal") -> None:
        """Orchestrate model load: stamp the generation, acquire the lock, load,
        warm up, and signal the UI.  Aborts silently if superseded at any step."""
        from transcribe_app.config import get_model_size  # noqa: PLC0415

        use_gpu    = self._use_gpu
        opts       = LANGUAGE_OPTS[lang]
        model_size = get_model_size(lang, speed, use_gpu)
        fallback   = opts["fallback_model_size"]
        lan        = opts["lan"]

        # Stamp this request so any later reload() can detect it has been superseded.
        self._load_gen += 1
        my_gen = self._load_gen

        # Update the status bar immediately so the UI reflects the new selection
        # even while waiting for the lock.
        self._on_status(loading_status(model_size, lang, use_gpu))

        if self._load_lock is None:
            self._load_lock = asyncio.Lock()

        async with self._load_lock:
            # Re-check after acquiring: a later reload() may have arrived
            # while we were waiting.
            if my_gen != self._load_gen:
                return

            self._lang = lang
            engine = await self._try_load(model_size, fallback, lan, use_gpu, prompt, my_gen, lang)
            if engine is None:
                return

            # Final check: another reload() may have arrived while the executor
            # thread was running.
            if my_gen != self._load_gen:
                return

            self._engine = engine
            await self._warmup_model(engine, model_size, lang, use_gpu)
            self._emit_ready_status(lang, use_gpu)

    # ── Recording session ─────────────────────────────────────────────────────

    def start_session(self) -> None:
        """Begin a transcription session. Call from the UI thread."""
        self._recording = True

        def _schedule() -> None:
            # Cancel any still-draining session from a previous recording so
            # it can never race the new one and fire stale UI callbacks.
            if self._session_task is not None and not self._session_task.done():
                self._session_task.cancel()
            self._session_task = self._loop.create_task(self._run_session())  # type: ignore[union-attr]

        self._loop.call_soon_threadsafe(_schedule)  # type: ignore[union-attr]

    async def _run_session(self) -> None:
        """Run one transcription session on the asyncio thread."""
        # Stamp this session so callbacks from a superseded session are silently
        # dropped even if the coroutine is still draining on the event loop.
        self._session_gen += 1
        my_gen = self._session_gen

        # Keep a local reference so the finally block always cleans up *this*
        # session's processor, not a replacement created by a concurrent restart.
        processor = _AudioProcessor(transcription_engine=self._engine)
        self._processor = processor
        results_gen = await processor.create_tasks()
        # Signal the UI thread to open the mic stream via root.after(0, …)
        self._on_open_mic()

        committed_text = ""
        last_partial   = ""

        try:
            async for front_data in results_gen:
                # A new session was started (or this task was cancelled) — stop
                # updating the UI immediately so only one session drives the display.
                if my_gen != self._session_gen:
                    break

                # front_data.lines is the complete current snapshot of committed
                # segments — treat it as a replacement, not an accumulation.
                # Accumulating with a seen-set causes duplicates whenever
                # WhisperLiveKit refines a segment (text or timestamps change).
                new_committed = " ".join(
                    (seg.text or "").strip()
                    for seg in front_data.lines
                    if (seg.text or "").strip()
                )
                partial = (front_data.buffer_transcription or "").strip()

                if new_committed != committed_text or partial != last_partial:
                    committed_text = new_committed
                    last_partial   = partial
                    self._on_update(committed_text, partial)

            if my_gen == self._session_gen:
                final = f"{committed_text} {last_partial}".strip() if last_partial else committed_text
                self._on_finalise(final)

        finally:
            # stop_session() sends None to process_audio to flush the processor;
            # here we just close the generator and release processor resources.
            for coro in (results_gen.aclose(), processor.cleanup()):
                try:
                    await coro
                except Exception:
                    pass

    # ── Audio stream ──────────────────────────────────────────────────────────

    def _make_audio_callback(self, processor: object, loop: asyncio.AbstractEventLoop) -> Callable:
        """Return the sounddevice InputStream callback for the current session.

        Normalises int16 PCM to [-1, 1] with a fixed divisor (not per-chunk peak)
        so relative amplitudes between frames are preserved for VAD and the encoder.
        Sends the same processed audio to both the transcription engine and
        audio_sink so that re-transcription sees identical input to the live session.
        """
        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if not self._recording:
                return
            samples = indata.astype(np.float32) / 32768.0
            gain = self.mic_gain  # float; atomic read under CPython GIL
            if gain != 1.0:
                samples = (samples * gain).clip(-1.0, 1.0)
            audio = (samples * 32767).astype("int16")
            asyncio.run_coroutine_threadsafe(processor.process_audio(audio.tobytes()), loop)
            if self.audio_sink is not None:
                self.audio_sink(audio)

        return _callback

    def open_mic_stream(self, device: int | None = None) -> None:
        """Open the sounddevice InputStream. Must be called from the UI thread
        (scheduled via root.after) so that self._stream is only ever touched on
        the UI thread, matching stop_session()."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("open_mic_stream must be called from the UI (main) thread")
        import sounddevice as sd  # noqa: PLC0415

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
            blocksize=int(SAMPLE_RATE * CHUNK_SECONDS),
            callback=self._make_audio_callback(self._processor, self._loop),
            device=device,
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:
                pass
            self._stream = None
            raise
        self._stream = stream

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

    # ── Model access ──────────────────────────────────────────────────────────

    @property
    def whisper_asr(self) -> "object | None":
        """The FasterWhisperASR backend, or None if the model is not yet loaded.

        Exposes .transcribe(audio_float32, init_prompt="") and .original_language.
        Safe to read from the UI thread once a session has finished.
        """
        return getattr(self._engine, "asr", None)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release GPU memory and stop the background thread. Call from WM_DELETE_WINDOW."""
        self.stop_session()
        self._processor = None
        if self._engine is not None and _TranscriptionEngine is not None:
            _TranscriptionEngine._instance    = None
            _TranscriptionEngine._initialized = False
            self._engine = None
        if self._use_gpu:
            try:
                import torch  # noqa: PLC0415
                torch.cuda.empty_cache()
            except Exception:
                pass
        if self._loop is not None and self._loop.is_running():
            def _cancel_and_stop() -> None:
                for task in (self._session_task, self._load_task):
                    if task is not None and not task.done():
                        task.cancel()
                self._loop.stop()  # type: ignore[union-attr]
            self._loop.call_soon_threadsafe(_cancel_and_stop)
