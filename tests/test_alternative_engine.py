"""Tests for transcribe_app.alternative_engine — SpeechToTextEngine.

All tests inject fake model and vad objects so no GPU, model download,
or audio hardware is needed.
"""

import asyncio
import io
import time
from queue import Queue
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcribe_app.alternative_engine import Config, SpeechToTextEngine


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_engine(output_mode="queue", **cfg_kwargs) -> SpeechToTextEngine:
    """Return an engine with mocked VAD and Whisper model."""
    cfg = Config(output_mode=output_mode, **cfg_kwargs)
    mock_vad   = MagicMock()
    mock_model = MagicMock()
    # Default: model.transcribe returns an empty iterator so text == ""
    mock_model.transcribe.return_value = (iter([]), MagicMock())
    return SpeechToTextEngine(cfg, model=mock_model, vad=mock_vad)


def _make_segment(text: str):
    s = MagicMock()
    s.text = text
    return s


# ── Construction ───────────────────────────────────────────────────────────────

class TestConstruction:
    def test_injected_model_is_used(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), MagicMock())
        engine = SpeechToTextEngine(Config(), model=mock_model, vad=MagicMock())
        assert engine.model is mock_model

    def test_injected_vad_is_used(self):
        mock_vad = MagicMock()
        engine = SpeechToTextEngine(Config(), model=MagicMock(), vad=mock_vad)
        assert engine.vad is mock_vad

    def test_frame_size_computed_from_config(self):
        cfg = Config(sample_rate=16000, frame_ms=30)
        engine = _make_engine()
        assert engine.frame_size == int(16000 * 30 / 1000)

    def test_queues_are_empty_on_init(self):
        engine = _make_engine()
        assert engine.audio_q.empty()
        assert engine.result_queue.empty()

    def test_stop_event_not_set_on_init(self):
        engine = _make_engine()
        assert not engine.stop_event.is_set()


# ── reset() ────────────────────────────────────────────────────────────────────

class TestReset:
    def test_clears_buffer(self):
        engine = _make_engine()
        engine.buffer = [np.zeros(10, dtype=np.int16)]
        engine.reset()
        assert engine.buffer == []

    def test_resets_session_id(self):
        engine = _make_engine()
        engine.session_id = 5
        engine.reset()
        assert engine.session_id == 0

    def test_clears_stop_event(self):
        engine = _make_engine()
        engine.stop_event.set()
        engine.reset()
        assert not engine.stop_event.is_set()

    def test_drains_audio_queue(self):
        engine = _make_engine()
        for _ in range(3):
            engine.audio_q.put_nowait(np.zeros(10, dtype=np.int16))
        engine.reset()
        assert engine.audio_q.empty()

    def test_drains_result_queue(self):
        engine = _make_engine()
        engine.result_queue.put_nowait("stale result")
        engine.reset()
        assert engine.result_queue.empty()

    def test_resets_last_voice_to_now(self):
        engine = _make_engine()
        engine.last_voice = 0.0
        before = time.monotonic()
        engine.reset()
        assert engine.last_voice >= before


# ── is_speech() ────────────────────────────────────────────────────────────────

class TestIsSpeech:
    def test_delegates_to_vad(self):
        engine = _make_engine()
        engine.vad.is_speech.return_value = True
        frame = np.zeros(480, dtype=np.int16)
        assert engine.is_speech(frame) is True
        engine.vad.is_speech.assert_called_once_with(frame.tobytes(), engine.cfg.sample_rate)

    def test_returns_false_when_vad_says_so(self):
        engine = _make_engine()
        engine.vad.is_speech.return_value = False
        assert engine.is_speech(np.zeros(480, dtype=np.int16)) is False


# ── flush() ────────────────────────────────────────────────────────────────────

class TestFlush:
    def test_does_nothing_when_buffer_too_small(self):
        engine = _make_engine(min_buffer_chunks=5)
        engine.buffer = [np.zeros(10, dtype=np.int16)] * 2  # fewer than min
        engine.flush()
        # Buffer is cleared but no transcription scheduled
        assert engine.buffer == []
        engine.model.transcribe.assert_not_called()

    def test_clears_buffer_after_dispatch(self):
        engine = _make_engine(min_buffer_chunks=1)
        engine.loop = asyncio.new_event_loop()
        engine.buffer = [np.zeros(480, dtype=np.int16)]
        engine.flush()
        assert engine.buffer == []
        engine.loop.close()

    def test_increments_session_id(self):
        engine = _make_engine(min_buffer_chunks=1)
        engine.loop = asyncio.new_event_loop()
        engine.buffer = [np.zeros(480, dtype=np.int16)]
        engine.flush()
        assert engine.session_id == 1
        engine.loop.close()


# ── _emit() ────────────────────────────────────────────────────────────────────

class TestEmit:
    def test_queue_mode_puts_text(self):
        engine = _make_engine(output_mode="queue")
        engine._emit("hello")
        assert engine.result_queue.get_nowait() == "hello"

    def test_stream_mode_writes_to_stream(self):
        stream = io.StringIO()
        engine = _make_engine(output_mode="stream")
        engine._output_stream = stream
        engine._emit("hello")
        assert "hello" in stream.getvalue()

    def test_print_mode_calls_print(self):
        engine = _make_engine(output_mode="print")
        with patch("builtins.print") as mock_print:
            engine._emit("hello")
            mock_print.assert_called_once_with("hello")

    def test_file_mode_writes_to_file(self, tmp_path):
        prefix = str(tmp_path / "session")
        engine = _make_engine(output_mode="file", output_prefix=prefix)
        engine._emit("hello")
        content = (tmp_path / "session.txt").read_text(encoding="utf-8")
        assert "hello" in content

    def test_unknown_mode_raises(self):
        engine = _make_engine(output_mode="queue")
        engine.cfg = Config(output_mode="bogus")
        with pytest.raises(ValueError, match="unknown output_mode"):
            engine._emit("x")

    def test_queue_full_does_not_raise(self):
        engine = _make_engine(output_mode="queue", queue_size=1)
        engine.result_queue.put_nowait("existing")
        # Should log a warning but not raise
        engine._emit("overflow")


# ── transcribe_internal() ──────────────────────────────────────────────────────

class TestTranscribeInternal:
    def test_joins_segment_texts(self):
        engine = _make_engine()
        seg1 = _make_segment(" Hello")
        seg2 = _make_segment(" world")
        engine.model.transcribe.return_value = (iter([seg1, seg2]), MagicMock())
        audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe_internal(audio)
        assert result == "Hello world"

    def test_empty_segments_returns_empty(self):
        engine = _make_engine()
        engine.model.transcribe.return_value = (iter([]), MagicMock())
        result = engine.transcribe_internal(np.zeros(16000, dtype=np.float32))
        assert result == ""

    def test_passes_language_to_model(self):
        engine = _make_engine()
        engine.cfg = Config(language="de", output_mode="queue")
        engine.model.transcribe.return_value = (iter([]), MagicMock())
        engine.transcribe_internal(np.zeros(100, dtype=np.float32))
        _, kwargs = engine.model.transcribe.call_args
        assert kwargs.get("language") == "de"

    def test_auto_language_passes_none(self):
        engine = _make_engine()
        engine.cfg = Config(language="auto", output_mode="queue")
        engine.model.transcribe.return_value = (iter([]), MagicMock())
        engine.transcribe_internal(np.zeros(100, dtype=np.float32))
        _, kwargs = engine.model.transcribe.call_args
        assert kwargs.get("language") is None

    def test_passes_initial_prompt(self):
        engine = _make_engine()
        engine.cfg = Config(initial_prompt="Be precise.", output_mode="queue")
        engine.model.transcribe.return_value = (iter([]), MagicMock())
        engine.transcribe_internal(np.zeros(100, dtype=np.float32))
        _, kwargs = engine.model.transcribe.call_args
        assert kwargs.get("initial_prompt") == "Be precise."


# ── audio_callback() ──────────────────────────────────────────────────────────

class TestAudioCallback:
    def test_enqueues_frame(self):
        engine = _make_engine()
        indata = np.ones((480, 1), dtype=np.float32) * 0.5
        engine.audio_callback(indata, 480, None, None)
        frame = engine.audio_q.get_nowait()
        assert frame.dtype == np.int16
        assert len(frame) == 480

    def test_drops_frame_when_queue_full(self):
        engine = _make_engine(queue_size=1)
        indata = np.ones((480, 1), dtype=np.float32) * 0.1
        engine.audio_q.put_nowait(np.zeros(480, dtype=np.int16))  # fill queue
        # Should not raise
        engine.audio_callback(indata, 480, None, None)

    def test_calls_audio_sink_when_set(self):
        engine = _make_engine()
        sink = MagicMock()
        engine.audio_sink = sink
        indata = np.ones((480, 1), dtype=np.float32) * 0.5
        engine.audio_callback(indata, 480, None, None)
        sink.assert_called_once()
        frame = sink.call_args[0][0]
        assert frame.dtype == np.int16
        assert len(frame) == 480

    def test_audio_sink_none_by_default(self):
        engine = _make_engine()
        assert engine.audio_sink is None

    def test_no_sink_call_when_not_set(self):
        engine = _make_engine()
        indata = np.ones((480, 1), dtype=np.float32) * 0.5
        # Should not raise when audio_sink is None
        engine.audio_callback(indata, 480, None, None)

    def test_clips_peak_instead_of_wrapping(self):
        engine = _make_engine()
        indata = np.ones((32, 1), dtype=np.float32)
        engine.audio_callback(indata, 32, None, None)
        frame = engine.audio_q.get_nowait()
        assert np.all(frame == 32767)

    def test_applies_mic_gain_before_quantizing(self):
        engine = _make_engine(mic_gain=2.0)
        indata = np.full((16, 1), 0.25, dtype=np.float32)
        engine.audio_callback(indata, 16, None, None)
        frame = engine.audio_q.get_nowait()
        assert np.all(frame > 10000)
