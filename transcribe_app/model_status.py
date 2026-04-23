"""Lightweight model status helpers shared by the UI and engine backends."""

from pathlib import Path

from transcribe_app.config import GPU
from transcribe_app.i18n import t


def is_model_cached(model_size: str) -> bool:
    """Return True when the Faster-Whisper model is already present in the HF cache."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = [
        hub / f"models--Systran--faster-whisper-{model_size}",
        hub / f"models--mobiuslabsgmbh--faster-whisper-{model_size}",
    ]
    try:
        return any(
            (directory / "snapshots").is_dir() and any((directory / "snapshots").iterdir())
            for directory in candidates
        )
    except OSError:
        return False


def loading_status(model_size: str, lang: str, use_gpu: bool = GPU) -> str:
    """Return the translated loading/downloading status line for the selected model."""
    device = "GPU" if use_gpu else "CPU"
    key = "status.loading" if is_model_cached(model_size) else "status.downloading"
    return t(key, model=model_size, lang=lang, device=device)
