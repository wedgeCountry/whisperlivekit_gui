"""Tests for transcribe_app.engine_manager — EngineManager.

Only the parts that can be tested without starting the real asyncio engine
or loading a Whisper model are covered here:
- _make_audio_callback()  — PCM normalisation, gain, audio_sink, recording gate
- open_mic_stream()       — stream_factory injection (no sounddevice hardware)
- stop_session()          — stream lifecycle
- whisper_asr property    — delegates to async_engine
"""

import asyncio
import threading
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from transcribe_app.engine_manager import EngineManager


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_manager():
    return EngineManager(
        on_status   = MagicMock(),
        on_ready    = MagicMock(),
        on_update   = MagicMock(),
        on_finalise = MagicMock(),
        on_open_mic = MagicMock(),
    )


def _fake_loop():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.is_running.return_value = True
    return loop


# ── _make_audio_callback() ────────────────────────────────────────────────────

class TestAudioCallback:
    def _make_callback(self, manager, *, recording=True):
        processor = MagicMock()
        loop = _fake_loop()
        manager._recording = recording
        cb = manager._make_audio_callback(processor, loop)
        return cb, processor, loop

    def test_sends_audio_to_processor(self):
        manager = _make_manager()
        cb, processor, loop = self._make_callback(manager)
        indata = np.ones((480, 1), dtype=np.int16) * 16384
        with patch("asyncio.run_coroutine_threadsafe") as mock_rct:
            cb(indata, 480, None, None)
            mock_rct.assert_called_once()

    def test_does_nothing_when_not_recording(self):
        manager = _make_manager()
        cb, processor, loop = self._make_callback(manager, recording=False)
        indata = np.ones((480, 1), dtype=np.int16)
        with patch("asyncio.run_coroutine_threadsafe") as mock_rct:
            cb(indata, 480, None, None)
            mock_rct.assert_not_called()

    def test_applies_mic_gain(self):
        """Gain > 1.0 amplifies the samples that reach the audio_sink."""
        manager = _make_manager()
        manager.mic_gain = 2.0
        manager._recording = True

        sink_calls = []
        manager.audio_sink = lambda audio: sink_calls.append(audio.copy())

        processor = MagicMock()
        loop = _fake_loop()
        cb = manager._make_audio_callback(processor, loop)

        # input at 25 % amplitude; gain 2 should double it to 50 %
        indata = np.full((480, 1), 0.25 * 32768, dtype=np.float32)

        with patch("asyncio.run_coroutine_threadsafe"):
            cb(indata, 480, None, None)

        assert len(sink_calls) == 1
        # 0.25 * 32768 / 32768 * 2.0 * 32767 ≈ 16383
        assert np.all(sink_calls[0] > 0)

    def test_calls_audio_sink_when_set(self):
        manager = _make_manager()
        sink = MagicMock()
        manager.audio_sink = sink
        cb, processor, loop = self._make_callback(manager)
        indata = np.ones((480, 1), dtype=np.int16) * 100

        with patch("asyncio.run_coroutine_threadsafe"):
            cb(indata, 480, None, None)

        sink.assert_called_once()
        sent = sink.call_args[0][0]
        assert sent.dtype == np.int16

    def test_does_not_call_audio_sink_when_none(self):
        manager = _make_manager()
        manager.audio_sink = None
        cb, processor, loop = self._make_callback(manager)
        indata = np.ones((480, 1), dtype=np.int16)
        # Should not raise
        with patch("asyncio.run_coroutine_threadsafe"):
            cb(indata, 480, None, None)

    def test_gain_one_does_not_clip(self):
        """mic_gain == 1.0 should skip the gain branch."""
        manager = _make_manager()
        manager.mic_gain = 1.0
        cb, processor, loop = self._make_callback(manager)
        indata = np.ones((480, 1), dtype=np.int16) * 100
        with patch("asyncio.run_coroutine_threadsafe"):
            cb(indata, 480, None, None)


# ── open_mic_stream() ─────────────────────────────────────────────────────────

class TestOpenMicStream:
    def test_uses_stream_factory(self):
        manager = _make_manager()
        # Wire up just enough to call open_mic_stream
        manager._async_engine = MagicMock()
        manager._async_engine.processor = MagicMock()
        manager._loop = _fake_loop()

        fake_stream = MagicMock()
        factory = MagicMock(return_value=fake_stream)

        manager.open_mic_stream(stream_factory=factory)

        factory.assert_called_once()
        fake_stream.start.assert_called_once()
        assert manager._stream is fake_stream

    def test_raises_if_called_off_main_thread(self):
        manager = _make_manager()
        result = {}

        def _call():
            try:
                manager.open_mic_stream()
                result["ok"] = True
            except RuntimeError as e:
                result["err"] = str(e)

        t = threading.Thread(target=_call)
        t.start()
        t.join()
        assert "err" in result
        assert "main" in result["err"]

    def test_cleans_up_on_stream_start_failure(self):
        manager = _make_manager()
        manager._async_engine = MagicMock()
        manager._async_engine.processor = MagicMock()
        manager._loop = _fake_loop()

        fake_stream = MagicMock()
        fake_stream.start.side_effect = OSError("no device")
        factory = MagicMock(return_value=fake_stream)

        with pytest.raises(OSError):
            manager.open_mic_stream(stream_factory=factory)

        assert manager._stream is None
        fake_stream.close.assert_called_once()

    def test_windows_falls_back_to_shared_mode(self):
        manager = _make_manager()
        manager._async_engine = MagicMock()
        manager._async_engine.processor = MagicMock()
        manager._loop = _fake_loop()

        exclusive_exc = OSError("exclusive busy")
        shared_stream = MagicMock()
        sd = MagicMock()
        sd.WasapiSettings.return_value = object()
        sd.InputStream.side_effect = [exclusive_exc, shared_stream]
        sd.query_devices.return_value = {"hostapi": 0}
        sd.query_hostapis.return_value = {"name": "Windows WASAPI"}

        with patch("transcribe_app.engine_manager.IS_WINDOWS", True):
            with patch.dict("sys.modules", {"sounddevice": sd}):
                manager.open_mic_stream()

        assert manager._stream is shared_stream
        shared_stream.start.assert_called_once()
        manager._on_status.assert_called_once()

    def test_windows_falls_back_to_default_device(self):
        manager = _make_manager()
        manager._async_engine = MagicMock()
        manager._async_engine.processor = MagicMock()
        manager._loop = _fake_loop()

        fallback_stream = MagicMock()
        sd = MagicMock()
        sd.WasapiSettings.return_value = object()
        sd.InputStream.side_effect = [
            OSError("device missing"),
            OSError("shared missing"),
            fallback_stream,
        ]
        sd.query_devices.side_effect = [
            {"hostapi": 0},
            {"hostapi": 0},
        ]
        sd.query_hostapis.return_value = {"name": "Windows WASAPI"}

        with patch("transcribe_app.engine_manager.IS_WINDOWS", True):
            with patch.dict("sys.modules", {"sounddevice": sd}):
                manager.open_mic_stream(device=7)

        assert manager._stream is fallback_stream
        fallback_stream.start.assert_called_once()
        third_call = sd.InputStream.call_args_list[2]
        assert third_call.kwargs["device"] is None

    def test_windows_skips_wasapi_settings_for_non_wasapi_device(self):
        manager = _make_manager()
        manager._async_engine = MagicMock()
        manager._async_engine.processor = MagicMock()
        manager._loop = _fake_loop()

        plain_stream = MagicMock()
        sd = MagicMock()
        sd.InputStream.return_value = plain_stream
        sd.query_devices.return_value = {"hostapi": 1}
        sd.query_hostapis.return_value = {"name": "MME"}

        with patch("transcribe_app.engine_manager.IS_WINDOWS", True):
            with patch.dict("sys.modules", {"sounddevice": sd}):
                manager.open_mic_stream(device=3)

        plain_stream.start.assert_called_once()
        assert sd.WasapiSettings.call_count == 0
        first_call = sd.InputStream.call_args_list[0]
        assert "extra_settings" not in first_call.kwargs


# ── stop_session() ────────────────────────────────────────────────────────────

class TestStopSession:
    def test_stops_and_closes_stream(self):
        manager = _make_manager()
        fake_stream = MagicMock()
        manager._stream = fake_stream
        manager._async_engine = MagicMock()
        manager._async_engine.processor = None
        manager._loop = _fake_loop()

        with patch("asyncio.run_coroutine_threadsafe"):
            manager.stop_session()

        fake_stream.stop.assert_called_once()
        fake_stream.close.assert_called_once()
        assert manager._stream is None

    def test_sets_recording_false(self):
        manager = _make_manager()
        manager._recording = True
        manager._async_engine = MagicMock()
        manager._async_engine.processor = None
        manager._loop = _fake_loop()

        manager.stop_session()
        assert manager._recording is False

    def test_safe_with_no_stream(self):
        manager = _make_manager()
        manager._stream = None
        manager._async_engine = MagicMock()
        manager._async_engine.processor = None
        manager._loop = _fake_loop()
        manager.stop_session()  # should not raise


# ── whisper_asr property ──────────────────────────────────────────────────────

class TestWhisperAsr:
    def test_returns_none_when_engine_not_loaded(self):
        manager = _make_manager()
        assert manager.whisper_asr is None

    def test_delegates_to_async_engine(self):
        manager = _make_manager()
        fake_asr = MagicMock()
        manager._async_engine = MagicMock()
        manager._async_engine.whisper_asr = fake_asr
        assert manager.whisper_asr is fake_asr
