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
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from transcribe_app.config import CHUNK_SECONDS, CHANNELS, DTYPE, GPU, LANGUAGE_OPTS, SAMPLE_RATE

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


def loading_status(model_size: str, lang: str) -> str:
    device = "GPU" if GPU else "CPU"
    verb   = "Lade" if _is_model_cached(model_size) else "Lade herunter"
    return f"{verb}: {model_size}  ({lang} · {device})…"


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

    # ── Startup ───────────────────────────────────────────────────────────────

    def start(self, lang: str, prompt: str, speed: str = "normal") -> None:
        """Spin up the background asyncio thread and load the initial model."""
        def _run() -> None:
            global _AudioProcessor, _TranscriptionEngine, _WhisperLiveKitConfig
            from whisperlivekit import AudioProcessor as AP, TranscriptionEngine as TE  # noqa: PLC0415
            from whisperlivekit.config import WhisperLiveKitConfig as WLC               # noqa: PLC0415
            _AudioProcessor       = AP
            _TranscriptionEngine  = TE
            _WhisperLiveKitConfig = WLC

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._load_engine(lang, prompt, speed))
            self._loop.run_forever()

        threading.Thread(target=_run, daemon=True).start()

    # ── Engine lifecycle ──────────────────────────────────────────────────────

    def reload(self, lang: str, prompt: str, speed: str = "normal") -> None:
        """Hot-reload the model (e.g. after a language/speed change). UI-thread safe."""
        asyncio.run_coroutine_threadsafe(self._load_engine(lang, prompt, speed), self._loop)

    async def _load_engine(self, lang: str, prompt: str, speed: str = "normal") -> None:
        opts       = LANGUAGE_OPTS[lang]
        model_size = opts["model_sizes"][speed]
        fallback   = opts["fallback_model_size"]
        lan        = opts["lan"]
        self._lang = lang

        _TranscriptionEngine._instance = None
        _TranscriptionEngine._initialized = False
        self._on_status(loading_status(model_size, lang))

        def _make_cfg(size: str):
            return _WhisperLiveKitConfig(
                pcm_input=True, vac=True,
                model_size=size, lan=lan,
                static_init_prompt=prompt if prompt.strip() else None,
            )

        try:
            self._engine = _TranscriptionEngine(config=_make_cfg(model_size))
        except Exception as exc:
            if GPU and model_size != fallback:
                self._on_status(
                    f"Fehler beim Laden ({exc.__class__.__name__}), "
                    f"Fallback: {loading_status(fallback, lang)}"
                )
                _TranscriptionEngine._instance = None
                _TranscriptionEngine._initialized = False
                self._engine = _TranscriptionEngine(config=_make_cfg(fallback))
            else:
                self._on_status(f"Fehler: {exc}")
                self._on_ready()
                return

        self._on_status(f"Bereit  ·  {lang} Modell geladen")
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
        self._on_status(f"Ready  ·  {self._lang} model loaded")

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
            asyncio.run_coroutine_threadsafe(
                processor.process_audio(indata.tobytes()), loop
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
        if GPU:
            try:
                import torch  # noqa: PLC0415
                torch.cuda.empty_cache()
            except Exception:
                pass
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
