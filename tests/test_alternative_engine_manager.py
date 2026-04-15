"""Tests for transcribe_app.alternative_engine_manager — AlternativeEngineManager.

Uses engine_factory to inject a fake SpeechToTextEngine so no model loading
or audio hardware is involved.
"""

import threading
import time
from queue import Queue
from threading import Event
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from transcribe_app.alternative_engine_manager import AlternativeEngineManager


# ── Fake engine ────────────────────────────────────────────────────────────────

def _make_fake_engine():
    """Return a MagicMock that quacks like SpeechToTextEngine."""
    engine = MagicMock()
    engine.stop_event  = Event()
    engine.result_queue = Queue()
    engine.audio_q     = Queue()
    engine.buffer      = []
    engine.last_voice  = time.time()
    engine.session_id  = 0
    return engine


def _make_manager(fake_engine=None, **callback_overrides):
    """Return (manager, callbacks_dict, fake_engine)."""
    if fake_engine is None:
        fake_engine = _make_fake_engine()

    callbacks = {
        "on_status":   MagicMock(),
        "on_ready":    MagicMock(),
        "on_update":   MagicMock(),
        "on_finalise": MagicMock(),
        "on_open_mic": MagicMock(),
    }
    callbacks.update(callback_overrides)

    factory = MagicMock(return_value=fake_engine)

    manager = AlternativeEngineManager(
        on_status   = callbacks["on_status"],
        on_ready    = callbacks["on_ready"],
        on_update   = callbacks["on_update"],
        on_finalise = callbacks["on_finalise"],
        on_open_mic = callbacks["on_open_mic"],
        engine_factory = factory,
    )
    return manager, callbacks, fake_engine, factory


# ── Construction ───────────────────────────────────────────────────────────────

class TestConstruction:
    def test_mic_gain_defaults_to_one(self):
        manager, _, _, _ = _make_manager()
        assert manager.mic_gain == 1.0

    def test_audio_sink_defaults_to_none(self):
        manager, _, _, _ = _make_manager()
        assert manager.audio_sink is None

    def test_engine_is_none_before_start(self):
        manager, _, _, _ = _make_manager()
        assert manager._engine is None


# ── _load_engine() / start() ──────────────────────────────────────────────────

class TestLoadEngine:
    def test_uses_engine_factory(self):
        manager, cbs, fake, factory = _make_manager()
        manager._load_engine("English", "", "fast", "cpu")
        factory.assert_called_once()
        assert manager._engine is fake

    def test_on_ready_true_on_success(self):
        manager, cbs, _, _ = _make_manager()
        manager._load_engine("English", "", "fast", "cpu")
        cbs["on_ready"].assert_called_once_with(True)

    def test_on_status_called_during_load(self):
        manager, cbs, _, _ = _make_manager()
        manager._load_engine("English", "", "fast", "cpu")
        assert cbs["on_status"].call_count >= 2  # loading + ready

    def test_on_ready_false_on_factory_exception(self):
        manager, cbs, _, _ = _make_manager()
        manager._engine_factory = MagicMock(side_effect=RuntimeError("boom"))
        manager._load_engine("English", "", "fast", "cpu")
        cbs["on_ready"].assert_called_once_with(False)

    def test_start_spawns_background_thread(self):
        manager, cbs, _, _ = _make_manager()
        # Patch _load_engine so the thread exits immediately
        manager._load_engine = MagicMock()
        manager.start("English", "", "fast", "cpu")
        time.sleep(0.05)
        manager._load_engine.assert_called_once()


# ── start_session() ───────────────────────────────────────────────────────────

class TestStartSession:
    def test_calls_engine_reset(self):
        manager, cbs, fake, _ = _make_manager()
        manager._engine = fake
        manager.start_session()
        fake.reset.assert_called_once()
        manager.stop_session()

    def test_calls_on_open_mic(self):
        manager, cbs, fake, _ = _make_manager()
        manager._engine = fake
        manager.start_session()
        cbs["on_open_mic"].assert_called_once()
        manager.stop_session()

    def test_does_nothing_without_engine(self):
        manager, cbs, _, _ = _make_manager()
        manager.start_session()  # should not raise
        cbs["on_open_mic"].assert_not_called()

    def test_resets_accumulated_text(self):
        manager, cbs, fake, _ = _make_manager()
        manager._engine = fake
        manager._accumulated = "stale"
        manager.start_session()
        assert manager._accumulated == ""
        manager.stop_session()


# ── stop_session() ────────────────────────────────────────────────────────────

class TestStopSession:
    def test_sets_engine_stop_event(self):
        manager, cbs, fake, _ = _make_manager()
        manager._engine = fake
        manager.stop_session()
        # stop_event is a real threading.Event — verify it was actually set
        assert fake.stop_event.is_set()

    def test_sets_poll_stop(self):
        manager, _, _, _ = _make_manager()
        manager.stop_session()
        assert manager._poll_stop.is_set()

    def test_safe_when_engine_is_none(self):
        manager, _, _, _ = _make_manager()
        manager.stop_session()  # should not raise


# ── _poll_results() ───────────────────────────────────────────────────────────

class TestPollResults:
    def _run_poll(self, manager, fake_engine, results):
        """Put results into the engine's queue, then let _poll_results run."""
        for r in results:
            fake_engine.result_queue.put(r)
        manager._engine = fake_engine
        # Run _poll_results in a thread, stop it after a short delay
        manager._poll_stop.clear()
        t = threading.Thread(target=manager._poll_results, daemon=True)
        t.start()
        time.sleep(0.1)
        manager._poll_stop.set()
        t.join(timeout=1.0)

    def test_accumulates_segments(self):
        manager, cbs, fake, _ = _make_manager()
        self._run_poll(manager, fake, ["Hello", "world"])
        assert "Hello" in manager._accumulated
        assert "world" in manager._accumulated

    def test_calls_on_update_per_segment(self):
        manager, cbs, fake, _ = _make_manager()
        self._run_poll(manager, fake, ["Hello", "world"])
        assert cbs["on_update"].call_count == 2

    def test_calls_on_finalise_at_end(self):
        manager, cbs, fake, _ = _make_manager()
        self._run_poll(manager, fake, ["Final text"])
        cbs["on_finalise"].assert_called_once()
        assert "Final text" in cbs["on_finalise"].call_args[0][0]

    def test_drains_queue_after_stop(self):
        """Segments that land between stop signal and poll exit must be included."""
        manager, cbs, fake, _ = _make_manager()
        # Stop immediately, but pre-load a result — drain path should catch it.
        manager._poll_stop.set()
        fake.result_queue.put("late segment")
        manager._engine = fake
        manager._poll_results()
        assert "late segment" in cbs["on_finalise"].call_args[0][0]

    def test_empty_queue_produces_empty_finalise(self):
        manager, cbs, fake, _ = _make_manager()
        manager._poll_stop.set()
        manager._engine = fake
        manager._poll_results()
        cbs["on_finalise"].assert_called_once_with("")


# ── reload() ──────────────────────────────────────────────────────────────────

class TestReload:
    def test_calls_on_ready_false_then_reloads(self):
        manager, cbs, fake, _ = _make_manager()
        manager._engine = fake
        # Patch the heavy parts so the test completes quickly
        manager._load_engine = MagicMock()
        manager.reload("English", "", "fast", "cpu")
        cbs["on_ready"].assert_called_once_with(False)
        time.sleep(0.1)
        manager._load_engine.assert_called_once()


# ── shutdown() ────────────────────────────────────────────────────────────────

class TestShutdown:
    def test_sets_poll_stop(self):
        manager, _, _, _ = _make_manager()
        manager.shutdown()
        assert manager._poll_stop.is_set()

    def test_releases_engine_reference(self):
        manager, _, fake, _ = _make_manager()
        manager._engine = fake
        manager.shutdown()
        assert manager._engine is None


# ── whisper_asr property ──────────────────────────────────────────────────────

class TestWhisperAsr:
    def test_always_returns_none(self):
        manager, _, _, _ = _make_manager()
        assert manager.whisper_asr is None


# ── open_mic_stream (no-op) ───────────────────────────────────────────────────

class TestOpenMicStream:
    def test_is_noop(self):
        manager, _, _, _ = _make_manager()
        manager.open_mic_stream(device=0)  # should not raise or do anything
