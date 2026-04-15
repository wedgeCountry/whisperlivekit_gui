"""Tests for transcribe_app.session_file_manager — SessionFileManager."""

import time
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from transcribe_app.session_file_manager import SessionFileManager, ROTATE_AFTER_S
from transcribe_app.config import CHANNELS, SAMPLE_RATE


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_manager(tmp_path) -> SessionFileManager:
    return SessionFileManager(
        wav_dir  = tmp_path / "wav",
        diff_dir = tmp_path / "diff",
    )


def _chunk(n=480) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


def _read_wav(path: Path) -> tuple[int, int, int]:
    """Return (n_channels, sample_width, framerate) from a WAV file."""
    with wave.open(str(path), "rb") as f:
        return f.getnchannels(), f.getsampwidth(), f.getframerate()


# ── Construction ───────────────────────────────────────────────────────────────

class TestConstruction:
    def test_creates_wav_dir(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert (tmp_path / "wav").is_dir()

    def test_creates_diff_dir(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert (tmp_path / "diff").is_dir()

    def test_session_id_is_nonempty_string(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert isinstance(mgr.session_id, str) and mgr.session_id

    def test_two_instances_have_different_session_ids(self, tmp_path):
        a = _make_manager(tmp_path)
        b = _make_manager(tmp_path)
        assert a.session_id != b.session_id

    def test_wav_paths_empty_before_writing(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.wav_paths == []

    def test_diff_path_contains_session_id(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.session_id in str(mgr.diff_path)


# ── write_chunk() / finish_recording() ────────────────────────────────────────

class TestWriteAndFinish:
    def test_write_chunk_creates_wav_file(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        paths = mgr.finish_recording()
        assert len(paths) == 1
        assert paths[0].exists()

    def test_wav_file_has_correct_format(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        paths = mgr.finish_recording()
        nch, sw, rate = _read_wav(paths[0])
        assert nch  == CHANNELS
        assert sw   == 2          # int16 = 2 bytes
        assert rate == SAMPLE_RATE

    def test_multiple_chunks_go_into_one_file(self, tmp_path):
        mgr = _make_manager(tmp_path)
        for _ in range(5):
            mgr.write_chunk(_chunk())
        paths = mgr.finish_recording()
        assert len(paths) == 1

    def test_finish_recording_closes_wav(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        mgr.finish_recording()
        assert mgr._wav is None

    def test_write_chunk_ignored_after_finish(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        mgr.finish_recording()
        # Should not raise or create extra files
        mgr.write_chunk(_chunk())
        assert len(mgr.wav_paths) == 1

    def test_finish_without_any_chunks_returns_empty(self, tmp_path):
        mgr = _make_manager(tmp_path)
        paths = mgr.finish_recording()
        assert paths == []

    def test_wav_paths_property_returns_snapshot(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        mgr.finish_recording()
        snapshot = mgr.wav_paths
        assert len(snapshot) == 1


# ── File rotation ──────────────────────────────────────────────────────────────

class TestRotation:
    def test_rotates_to_new_file_after_threshold(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        # Simulate time passing beyond ROTATE_AFTER_S
        mgr._file_start -= ROTATE_AFTER_S + 1
        mgr.write_chunk(_chunk())
        paths = mgr.finish_recording()
        assert len(paths) == 2

    def test_rotated_files_have_sequential_suffixes(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        mgr._file_start -= ROTATE_AFTER_S + 1
        mgr.write_chunk(_chunk())
        paths = mgr.finish_recording()
        names = [p.name for p in paths]
        # First file has no numeric suffix, second has _1
        assert any("_1" in n for n in names)

    def test_both_rotated_files_are_valid_wav(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        mgr._file_start -= ROTATE_AFTER_S + 1
        mgr.write_chunk(_chunk())
        for path in mgr.finish_recording():
            nch, _, _ = _read_wav(path)
            assert nch == CHANNELS


# ── cleanup() ─────────────────────────────────────────────────────────────────

class TestCleanup:
    def test_deletes_wav_files(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        paths = mgr.finish_recording()
        mgr.cleanup()
        for p in paths:
            assert not p.exists()

    def test_deletes_diff_file_if_present(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.diff_path.write_text("diff content", encoding="utf-8")
        mgr.cleanup()
        assert not mgr.diff_path.exists()

    def test_cleanup_silent_if_no_files(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.cleanup()  # should not raise

    def test_cleanup_removes_all_rotated_files(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.write_chunk(_chunk())
        mgr._file_start -= ROTATE_AFTER_S + 1
        mgr.write_chunk(_chunk())
        paths = mgr.finish_recording()
        mgr.cleanup()
        for p in paths:
            assert not p.exists()


# ── Thread safety ──────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_writes_do_not_raise(self, tmp_path):
        import threading
        mgr = _make_manager(tmp_path)
        errors = []

        def _writer():
            try:
                for _ in range(20):
                    mgr.write_chunk(_chunk())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
