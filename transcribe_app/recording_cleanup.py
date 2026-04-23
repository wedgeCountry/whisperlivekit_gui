"""Cleanup helpers for temporary recording artifacts."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from transcribe_app.config import DIFF_DIR, SESSIONS_DIR, VAD_SNIPPETS_DIR

_log = logging.getLogger(__name__)

_RECORDING_DIRS: tuple[Path, ...] = (SESSIONS_DIR, VAD_SNIPPETS_DIR)
_STARTUP_DIRS: tuple[Path, ...] = (SESSIONS_DIR, DIFF_DIR, VAD_SNIPPETS_DIR)


def _iter_files(directories: tuple[Path, ...]):
    for directory in directories:
        try:
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if path.is_file():
                    yield path
        except Exception:
            _log.warning("Could not enumerate cleanup directory %s", directory, exc_info=True)


def _delete_path(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        _log.warning("Could not delete recording artifact %s", path, exc_info=True)


def delete_all_recording_artifacts() -> None:
    """Delete all persisted session/VAD artifacts on application startup."""
    for path in _iter_files(_STARTUP_DIRS):
        _delete_path(path)


def delete_recordings_older_than(max_age_seconds: float) -> None:
    """Delete persisted session WAVs and VAD snippets older than *max_age_seconds*."""
    cutoff = time.time() - max_age_seconds
    for path in _iter_files(_RECORDING_DIRS):
        try:
            if path.stat().st_mtime <= cutoff:
                _delete_path(path)
        except FileNotFoundError:
            continue
        except Exception:
            _log.warning("Could not stat recording artifact %s", path, exc_info=True)


def start_async_recordings_cleanup(max_age_seconds: float) -> threading.Thread:
    """Run expired-recording cleanup on a background daemon thread."""
    thread = threading.Thread(
        target=delete_recordings_older_than,
        args=(max_age_seconds,),
        daemon=True,
        name="recording-cleanup",
    )
    thread.start()
    return thread
