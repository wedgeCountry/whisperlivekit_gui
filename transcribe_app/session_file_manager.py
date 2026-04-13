"""SessionFileManager — owns all files produced during one transcription session.

Lifecycle
---------
1. Instantiate at recording start.
2. Pass write_chunk as the engine's audio_sink; it captures raw PCM to WAV
   files, rotating to a new file every ROTATE_AFTER_S seconds.
3. Call finish_recording() when the user stops — closes the current WAV and
   returns the list of written paths.
4. After post-processing: write to diff_path, then call cleanup() to delete
   all session files.
"""

import logging
import threading
import time
import uuid
import wave
from pathlib import Path

import numpy as np

from transcribe_app.config import CHANNELS, SAMPLE_RATE

_log = logging.getLogger(__name__)

# Rotate to a new WAV file after this many seconds of accumulated audio.
ROTATE_AFTER_S: float = 60.0


class SessionFileManager:
    """Manages WAV files and the diff file for one transcription session.

    Thread-safety: write_chunk() and finish_recording() may be called from
    different threads; all WAV state is protected by a lock.
    """

    def __init__(self, wav_dir: Path, diff_dir: Path) -> None:
        wav_dir.mkdir(parents=True, exist_ok=True)
        self._wav_dir    = wav_dir
        self._diff_dir   = diff_dir
        self._session_id = uuid.uuid4().hex[:12]
        self._lock       = threading.Lock()
        self._file_index = 0
        self._wav: wave.Wave_write | None = None
        self._file_start = 0.0
        self._wav_paths: list[Path] = []
        self._finished   = False

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    # ── File paths ────────────────────────────────────────────────────────────

    @property
    def wav_paths(self) -> list[Path]:
        """Snapshot of written WAV paths.  Populated after finish_recording()."""
        with self._lock:
            return list(self._wav_paths)

    @property
    def diff_path(self) -> Path:
        """Canonical location for the post-processing diff file."""
        return self._diff_dir / f"{self._session_id}_diff.txt"

    # ── WAV writing (sounddevice callback thread) ─────────────────────────────

    def _wav_path_for_index(self) -> Path:
        suffix = f"_{self._file_index}" if self._file_index > 0 else ""
        return self._wav_dir / f"{self._session_id}{suffix}.wav"

    def _open_wav(self) -> None:
        path = self._wav_path_for_index()
        self._wav_paths.append(path)
        w = wave.open(str(path), "wb")
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)       # int16 → 2 bytes/sample
        w.setframerate(SAMPLE_RATE)
        self._wav        = w
        self._file_start = time.monotonic()

    def write_chunk(self, audio: np.ndarray) -> None:
        """Append *audio* (int16 ndarray) to the current WAV file.

        Rotates to a new file every ROTATE_AFTER_S seconds.
        Called from the sounddevice callback thread — must be fast.
        """
        with self._lock:
            if self._finished:
                return
            if self._wav is None:
                self._open_wav()
            elif time.monotonic() - self._file_start >= ROTATE_AFTER_S:
                try:
                    self._wav.close()
                except Exception:
                    pass
                self._file_index += 1
                self._open_wav()
            try:
                self._wav.writeframes(audio.tobytes())
            except Exception:
                _log.warning("SessionFileManager: failed to write audio frames", exc_info=True)

    def finish_recording(self) -> list[Path]:
        """Close the current WAV file and return all written paths.

        Safe to call from the UI thread after the sounddevice stream is stopped.
        """
        with self._lock:
            self._finished = True
            if self._wav is not None:
                try:
                    self._wav.close()
                except Exception:
                    _log.warning("SessionFileManager: failed to close WAV", exc_info=True)
                self._wav = None
        return list(self._wav_paths)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Delete all session files (WAV files and the diff file, if present)."""
        for path in list(self._wav_paths):
            try:
                path.unlink(missing_ok=True)
                _log.debug("SessionFileManager: deleted %s", path)
            except Exception:
                _log.warning("SessionFileManager: could not delete %s", path, exc_info=True)

        try:
            self.diff_path.unlink(missing_ok=True)
            _log.debug("SessionFileManager: deleted %s", self.diff_path)
        except Exception:
            _log.warning("SessionFileManager: could not delete diff %s", self.diff_path, exc_info=True)
