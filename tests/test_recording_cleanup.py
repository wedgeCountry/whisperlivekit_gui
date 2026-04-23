"""Tests for transcribe_app.recording_cleanup."""

import time

from transcribe_app.recording_cleanup import (
    delete_all_recording_artifacts,
    delete_recordings_older_than,
)


def test_delete_all_recording_artifacts_removes_files(tmp_path):
    session_dir = tmp_path / "sessions"
    diff_dir = tmp_path / "diff"
    vad_dir = tmp_path / "vad"
    for directory, name in (
        (session_dir, "a.wav"),
        (diff_dir, "a.txt"),
        (vad_dir, "snippet_0000.wav"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text("x", encoding="utf-8")

    import transcribe_app.recording_cleanup as rc

    old_dirs = rc._STARTUP_DIRS
    rc._STARTUP_DIRS = (session_dir, diff_dir, vad_dir)
    try:
        delete_all_recording_artifacts()
    finally:
        rc._STARTUP_DIRS = old_dirs

    assert list(session_dir.iterdir()) == []
    assert list(diff_dir.iterdir()) == []
    assert list(vad_dir.iterdir()) == []


def test_delete_recordings_older_than_leaves_recent_files(tmp_path):
    session_dir = tmp_path / "sessions"
    vad_dir = tmp_path / "vad"
    session_dir.mkdir(parents=True, exist_ok=True)
    vad_dir.mkdir(parents=True, exist_ok=True)
    old_path = session_dir / "old.wav"
    new_path = vad_dir / "new.wav"
    old_path.write_text("x", encoding="utf-8")
    new_path.write_text("x", encoding="utf-8")

    old_ts = time.time() - 500
    new_ts = time.time()
    old_path.touch()
    new_path.touch()
    import os
    os.utime(old_path, (old_ts, old_ts))
    os.utime(new_path, (new_ts, new_ts))

    import transcribe_app.recording_cleanup as rc

    old_dirs = rc._RECORDING_DIRS
    rc._RECORDING_DIRS = (session_dir, vad_dir)
    try:
        delete_recordings_older_than(120.0)
    finally:
        rc._RECORDING_DIRS = old_dirs

    assert not old_path.exists()
    assert new_path.exists()
