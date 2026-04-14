"""EngineManagerProtocol — abstract base class all engine managers must implement.

The UI depends only on this protocol; concrete implementations live in
engine_manager.py (WhisperLiveKit streaming) and
alternative_engine_manager.py (VAD + faster-whisper).
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

ENGINE_TYPES: tuple[str, ...] = ("whisperlive", "faster_whisper")

# Human-readable labels indexed by engine type key.
ENGINE_LABELS: dict[str, str] = {
    "whisperlive":    "WhisperLive (streaming)",
    "faster_whisper": "Faster Whisper (VAD)",
}


class EngineManagerProtocol(ABC):
    """Public interface that TranscriptionApp uses to control any engine backend.

    Concrete implementations must also expose these plain attributes:
        mic_gain:   float
        audio_sink: Callable[[np.ndarray], None] | None
    """

    @abstractmethod
    def start(
        self, lang: str, prompt: str,
        speed: str = "normal", compute_device: str = "cpu",
    ) -> None:
        """Spin up the engine and load the initial model. Call once per instance."""

    @abstractmethod
    def reload(
        self, lang: str, prompt: str,
        speed: str = "normal", compute_device: str | None = None,
    ) -> None:
        """Hot-reload with new settings. Safe to call from the UI thread."""

    @abstractmethod
    def start_session(self) -> None:
        """Begin a transcription session. Call from the UI thread."""

    @abstractmethod
    def stop_session(self) -> None:
        """Stop the current session. Call from the UI thread."""

    @abstractmethod
    def open_mic_stream(self, device: int | None = None) -> None:
        """Open the microphone stream. Scheduled via root.after from on_open_mic."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release all resources. Call from WM_DELETE_WINDOW."""

    @property
    @abstractmethod
    def whisper_asr(self) -> "object | None":
        """The ASR backend for post-session re-transcription, or None if unsupported."""


def create_engine_manager(
    engine_type: str,
    on_status:   Callable[[str], None],
    on_ready:    Callable[[bool], None],
    on_update:   Callable[[str, str], None],
    on_finalise: Callable[[str], None],
    on_open_mic: Callable[[], None],
) -> EngineManagerProtocol:
    """Return the right engine manager for *engine_type*.

    Imports are lazy so that neither WhisperLiveKit nor faster-whisper are
    loaded until the user has actually selected that backend.
    """
    if engine_type == "whisperlive":
        from transcribe_app.engine_manager import EngineManager  # noqa: PLC0415
        return EngineManager(on_status, on_ready, on_update, on_finalise, on_open_mic)
    if engine_type == "faster_whisper":
        from transcribe_app.alternative_engine_manager import AlternativeEngineManager  # noqa: PLC0415
        return AlternativeEngineManager(on_status, on_ready, on_update, on_finalise, on_open_mic)
    raise ValueError(
        f"Unknown engine_type: {engine_type!r}  (expected one of {ENGINE_TYPES})"
    )
