"""AlternativeEngineManager — wraps SpeechToTextEngine (VAD + faster-whisper).

Implements EngineManagerProtocol so the UI can use either engine backend
interchangeably via create_engine_manager().

Key differences from EngineManager (WhisperLiveKit):
- The engine owns its own sounddevice stream; open_mic_stream() is a no-op.
- No hot-reload: reload() stops the running session and re-creates the engine.
- No partial/buffer transcription: on_update is called once per completed
  segment with accumulated committed text; on_finalise fires on session end.
- Re-transcription (whisper_asr) is not supported; always returns None.
"""

import logging
import threading
from queue import Empty
from typing import Callable

_log = logging.getLogger(__name__)

from transcribe_app.config import GPU, LANGUAGE_OPTS, VAD_SNIPPETS_DIR, get_model_size
from transcribe_app.model_status import loading_status
from transcribe_app.engine_protocol import EngineManagerProtocol
from transcribe_app.i18n import t


class AlternativeEngineManager(EngineManagerProtocol):
    def __init__(
        self,
        on_status:      Callable[[str], None],
        on_ready:       Callable[[bool], None],
        on_update:      Callable[[str, str], None],
        on_finalise:    Callable[[str], None],
        on_open_mic:    Callable[[], None],
        engine_factory: "Callable | None" = None,
    ) -> None:
        self._on_status   = on_status
        self._on_ready    = on_ready
        self._on_update   = on_update
        self._on_finalise = on_finalise
        self._on_open_mic = on_open_mic
        self._engine_factory = engine_factory  # None → import SpeechToTextEngine lazily

        self._engine         = None          # SpeechToTextEngine | None
        self._compute_device = "cuda" if GPU else "cpu"

        self._session_thread: threading.Thread | None = None
        self._poll_thread:    threading.Thread | None = None
        self._poll_stop:      threading.Event         = threading.Event()
        self._accumulated:    str                     = ""

        self.mic_gain:        float                                   = 1.0
        self.input_device:    int | None                              = None
        self.audio_sink:      "Callable[[np.ndarray], None] | None"  = None
        self.vad_silence_gap: float                                   = 0.8

    # ── Config construction ───────────────────────────────────────────────────

    @staticmethod
    def _make_config(lang: str, prompt: str, speed: str, compute_device: str, silence_gap: float = 0.8):
        from transcribe_app.alternative_engine import Config  # noqa: PLC0415
        use_gpu = compute_device == "cuda" and GPU
        return Config(
            model_size    = get_model_size(lang, speed, use_gpu),
            device        = "cuda" if use_gpu else "cpu",
            compute_type  = "float16" if use_gpu else "int8",
            language      = LANGUAGE_OPTS[lang]["lan"],
            initial_prompt= prompt,
            output_mode   = "queue",
            silence_limit    = silence_gap,
            min_buffer_chunks= 8,
            wav_snippet_dir  = str(VAD_SNIPPETS_DIR),
        )

    # ── Model loading ─────────────────────────────────────────────────────────

    def start(
        self, lang: str, prompt: str,
        speed: str = "normal", compute_device: str = "cuda" if GPU else "cpu",
    ) -> None:
        """Load the model in a background thread so the UI stays responsive."""
        self._compute_device = compute_device
        threading.Thread(
            target=self._load_engine,
            args=(lang, prompt, speed, compute_device),
            daemon=True,
        ).start()

    def _load_engine(self, lang: str, prompt: str, speed: str, compute_device: str) -> None:
        if self._engine_factory is None:
            from transcribe_app.alternative_engine import SpeechToTextEngine  # noqa: PLC0415
            factory = SpeechToTextEngine
        else:
            factory = self._engine_factory
        use_gpu    = compute_device == "cuda" and GPU
        model_size = get_model_size(lang, speed, use_gpu)
        self._on_status(loading_status(model_size, lang, use_gpu))
        try:
            cfg = self._make_config(lang, prompt, speed, compute_device, self.vad_silence_gap)
            cfg = cfg.__class__(
                **{
                    **cfg.__dict__,
                    "input_device": self.input_device,
                    "mic_gain": self.mic_gain,
                }
            )
            self._engine = factory(cfg)
            self._on_status(t("status.ready", lang=lang))
            self._on_ready(True)
        except Exception as exc:
            _log.error("Failed to load SpeechToTextEngine", exc_info=True)
            self._on_status(t("status.error", exc=exc))
            self._on_ready(False)

    def reload(
        self, lang: str, prompt: str,
        speed: str = "normal", compute_device: str | None = None,
    ) -> None:
        """Stop the current session and reload with new settings."""
        self._compute_device = compute_device or self._compute_device
        self._on_ready(False)
        self.stop_session()
        # Drain threads in the background, then load the new engine.
        threading.Thread(
            target=self._reload_after_stop,
            args=(lang, prompt, speed, self._compute_device),
            daemon=True,
        ).start()

    def _reload_after_stop(self, lang: str, prompt: str, speed: str, compute_device: str) -> None:
        """Join session and poll threads, then load the new engine."""
        if self._session_thread is not None:
            self._session_thread.join(timeout=3.0)
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
        self._load_engine(lang, prompt, speed, compute_device)

    # ── Session ───────────────────────────────────────────────────────────────

    def start_session(self) -> None:
        """Start the engine's own sounddevice stream and begin polling results."""
        if self._engine is None:
            return

        # The VAD backend keeps its own frozen Config instance, so sync the
        # latest UI-selected input device and gain into the already-loaded
        # engine before opening the stream.
        if hasattr(self._engine, "cfg") and self._engine.cfg is not None:
            self._engine.cfg = self._engine.cfg.__class__(
                **{
                    **self._engine.cfg.__dict__,
                    "input_device": self.input_device,
                    "mic_gain": self.mic_gain,
                }
            )

        self._accumulated = ""
        self._poll_stop.clear()
        self._engine.reset()

        self._session_thread = threading.Thread(target=self._run_session, daemon=True)
        self._poll_thread    = threading.Thread(target=self._poll_results, daemon=True)
        self._session_thread.start()
        self._poll_thread.start()

        # Signal the UI to transition to "recording" state.  open_mic_stream()
        # is a no-op for this backend — the engine opens its own stream in run().
        self._on_open_mic()

    def _run_session(self) -> None:
        self._engine.audio_sink = self.audio_sink
        try:
            self._engine.run()
        except Exception:
            _log.error("SpeechToTextEngine.run() raised", exc_info=True)
        finally:
            self._engine.audio_sink = None

    def _poll_results(self) -> None:
        """Forward completed segments to on_update; call on_finalise on exit."""
        while not self._poll_stop.is_set():
            try:
                text = self._engine.result_queue.get(timeout=0.2)
                self._accumulated = f"{self._accumulated} {text}".strip()
                self._on_update(self._accumulated, "")
            except Empty:
                continue

        # Wait for the worker to finish its final synchronous flush so that the
        # last spoken segment (still in the buffer when Stop was pressed) is
        # in result_queue before we drain it below.
        if self._engine is not None:
            self._engine.flush_done.wait(timeout=60)

        # Drain all segments produced by the final flush (and any in-flight ones).
        while True:
            try:
                text = self._engine.result_queue.get_nowait()
                self._accumulated = f"{self._accumulated} {text}".strip()
            except Empty:
                break

        self._on_finalise(self._accumulated)

    def stop_session(self) -> None:
        if self._engine is not None:
            self._engine.stop_event.set()
        self._poll_stop.set()

    # ── Audio stream (no-op) ──────────────────────────────────────────────────

    def open_mic_stream(self, device: int | None = None) -> None:
        """No-op: SpeechToTextEngine opens its own sounddevice stream in run()."""

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def whisper_asr(self) -> None:
        return None

    def transcribe_audio(self, audio: "np.ndarray", prompt: str) -> "str | None":
        """Transcribe audio using the loaded SpeechToTextEngine model."""
        if self._engine is None:
            return None
        return self._engine.transcribe_internal(audio)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self.stop_session()
        self._engine = None
