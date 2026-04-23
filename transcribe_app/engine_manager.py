"""EngineManager — thread management, audio stream, and public UI API.

Owns the background asyncio thread and the sounddevice InputStream.  All
model-lifecycle and session logic is delegated to AsyncEngine (engine.py).

Thread model
------------
* UI thread    — calls start(), reload(), start_session(), stop_session(), shutdown()
* Async thread — runs the asyncio event loop; AsyncEngine lives here
* SD callback  — fires from a sounddevice thread; forwards PCM via run_coroutine_threadsafe
"""

import asyncio
import logging
import threading
from typing import Callable

import numpy as np

_log = logging.getLogger(__name__)

from transcribe_app.config import CHUNK_SECONDS, CHANNELS, DTYPE, GPU, IS_WINDOWS, SAMPLE_RATE
from transcribe_app.engine import AsyncEngine, loading_status  # noqa: F401  (re-export)
from transcribe_app.engine_protocol import EngineManagerProtocol
from transcribe_app.i18n import t


# ── Environment setup ──────────────────────────────────────────────────────────

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


# ── EngineManager ──────────────────────────────────────────────────────────────

class EngineManager(EngineManagerProtocol):
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

        self._loop:         asyncio.AbstractEventLoop | None = None
        self._async_engine: AsyncEngine | None               = None
        self._stream:       object | None                    = None
        self._recording:    bool                             = False
        self._use_gpu:      bool                             = GPU
        self._started:      bool                             = False

        # mic_gain is written by the UI thread and read by the sounddevice callback
        # thread.  CPython's GIL makes float reads/writes atomic in practice, but
        # treat it as a best-effort control knob, not a hard synchronisation point.
        self.mic_gain: float = 1.0

        # Optional audio capture hook.  Set to a callable before open_mic_stream();
        # it receives the same processed int16 ndarray chunk (fixed-normalised,
        # gain-applied) that was sent to the transcription engine, so re-transcription
        # sees identical input to the live session.  Set to None to disable.
        self.audio_sink: "Callable[[np.ndarray], None] | None" = None

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
            # Import WhisperLiveKit lazily so PortAudio / CUDA never block the
            # Tk window from opening.  Populate the lazy globals in engine.py
            # so AsyncEngine can reference them by name at call time.
            import transcribe_app.engine as _engine_mod  # noqa: PLC0415
            from whisperlivekit import AudioProcessor as AP, TranscriptionEngine as TE  # noqa: PLC0415
            from whisperlivekit.config import WhisperLiveKitConfig as WLC               # noqa: PLC0415
            _engine_mod._AudioProcessor       = AP
            _engine_mod._TranscriptionEngine  = TE
            _engine_mod._WhisperLiveKitConfig = WLC

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            self._async_engine = AsyncEngine(
                self._on_status, self._on_ready,
                self._on_update, self._on_finalise,
                self._on_open_mic, self._use_gpu,
            )
            self._async_engine.loop = self._loop

            try:
                self._loop.run_until_complete(self._async_engine._load_engine(lang, prompt, speed))
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
            self._async_engine.use_gpu = self._use_gpu  # type: ignore[union-attr]

        def _schedule() -> None:
            self._async_engine.schedule_reload(lang, prompt, speed)  # type: ignore[union-attr]

        self._loop.call_soon_threadsafe(_schedule)  # type: ignore[union-attr]

    # ── Recording session ─────────────────────────────────────────────────────

    def start_session(self) -> None:
        """Begin a transcription session. Call from the UI thread."""
        self._recording = True
        self._loop.call_soon_threadsafe(self._async_engine.schedule_session)  # type: ignore[union-attr]

    def stop_session(self) -> None:
        """Stop recording and flush the processor. Call from the UI thread."""
        self._recording = False
        if self._stream is not None:
            self._stream.stop()   # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None
        if self._async_engine is not None and self._async_engine.processor is not None:
            asyncio.run_coroutine_threadsafe(
                self._async_engine.processor.process_audio(None),  # type: ignore[union-attr]
                self._loop,  # type: ignore[arg-type]
            )

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
            asyncio.run_coroutine_threadsafe(processor.process_audio(audio.tobytes()), loop)  # type: ignore[union-attr]
            if self.audio_sink is not None:
                self.audio_sink(audio)

        return _callback

    def open_mic_stream(
        self,
        device: int | None = None,
        stream_factory=None,
    ) -> None:
        """Open the sounddevice InputStream. Must be called from the UI thread
        (scheduled via root.after) so that self._stream is only ever touched on
        the UI thread, matching stop_session().

        Parameters
        ----------
        device:
            sounddevice device index, or None for the system default.
        stream_factory:
            Optional callable with the same signature as ``sd.InputStream``.
            Defaults to ``sounddevice.InputStream``.  Tests supply a fake here
            to avoid opening real audio hardware.
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("open_mic_stream must be called from the UI (main) thread")
        import sounddevice as sd  # noqa: PLC0415

        sample_rate = SAMPLE_RATE
        dtype = DTYPE

        kwargs = dict(
            samplerate=sample_rate, channels=CHANNELS, dtype=dtype,
            blocksize=int(sample_rate * CHUNK_SECONDS),
            callback=self._make_audio_callback(self._async_engine.processor, self._loop),  # type: ignore[arg-type]
            device=device,
        )
        if stream_factory is not None:
            stream = stream_factory(**kwargs)
        elif IS_WINDOWS:
            attempts: list[tuple[dict, dict, "Callable[[], None] | None"]] = [
                (
                    kwargs,
                    {"extra_settings": sd.WasapiSettings(exclusive=True)},
                    None,
                ),
                (
                    kwargs,
                    {},
                    lambda: self._on_status(t("status.wasapi_shared")),
                ),
            ]
            if device is not None:
                default_kwargs = dict(kwargs, device=None)
                attempts.extend([
                    (
                        default_kwargs,
                        {"extra_settings": sd.WasapiSettings(exclusive=True)},
                        None,
                    ),
                    (
                        default_kwargs,
                        {},
                        lambda: self._on_status(t("status.wasapi_shared")),
                    ),
                ])

            last_exc: Exception | None = None
            stream = None
            for stream_kwargs, extra_kwargs, on_success in attempts:
                try:
                    stream = sd.InputStream(**stream_kwargs, **extra_kwargs)
                    if on_success is not None:
                        on_success()
                    break
                except Exception as exc:
                    last_exc = exc
                    if stream_kwargs.get("device") is None and device is not None:
                        _log.warning(
                            "Opening configured input device failed; retried with system default",
                            exc_info=True,
                        )
                    else:
                        _log.warning("Could not open Windows microphone stream", exc_info=True)
            if stream is None:
                raise last_exc if last_exc is not None else RuntimeError("Could not open microphone stream")
        else:
            stream = sd.InputStream(**kwargs)
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

    # ── Utilities ─────────────────────────────────────────────────────────────

    def dispatch(self, coro) -> "asyncio.Future":
        """Schedule an arbitrary coroutine on the background loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]

    @property
    def whisper_asr(self) -> "object | None":
        """The FasterWhisperASR backend, or None if the model is not yet loaded."""
        return self._async_engine.whisper_asr if self._async_engine is not None else None

    def transcribe_audio(self, audio: "np.ndarray", prompt: str) -> "str | None":
        """Transcribe audio using the loaded WhisperLiveKit ASR backend.

        Returns None if the active ASR backend is streaming-only (e.g.
        SimulStreamingASR) and does not support batch transcription.
        """
        import inspect  # noqa: PLC0415
        asr = self.whisper_asr
        if asr is None:
            return None
        if "init_prompt" not in inspect.signature(asr.transcribe).parameters:
            return None
        return asr.transcribe(audio, init_prompt=prompt)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release GPU memory and stop the background thread. Call from WM_DELETE_WINDOW."""
        self.stop_session()
        if self._async_engine is not None:
            self._async_engine.release()
        if self._loop is not None and self._loop.is_running():
            def _cancel_and_stop() -> None:
                self._async_engine.cancel_pending_tasks()  # type: ignore[union-attr]
                self._loop.stop()  # type: ignore[union-attr]
            self._loop.call_soon_threadsafe(_cancel_and_stop)
