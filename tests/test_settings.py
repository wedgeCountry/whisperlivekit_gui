"""Tests for transcribe_app.settings — load/save with path injection."""

import json
from pathlib import Path

import pytest

from transcribe_app.settings import Settings, load, save
from transcribe_app.config import DEFAULT_LANGUAGE, DEFAULT_PROMPTS


class TestLoadDefaults:
    def test_missing_file_returns_defaults(self, tmp_path):
        s = load(tmp_path / "nonexistent.json")
        assert isinstance(s, Settings)
        assert s.language == DEFAULT_LANGUAGE

    def test_invalid_json_returns_defaults(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        s = load(p)
        assert isinstance(s, Settings)

    def test_empty_json_object_returns_defaults(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("{}", encoding="utf-8")
        s = load(p)
        assert s.language == DEFAULT_LANGUAGE

    def test_unknown_language_falls_back_to_default(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"language": "Klingon"}), encoding="utf-8")
        s = load(p)
        assert s.language == DEFAULT_LANGUAGE

    def test_unknown_speed_falls_back_to_normal(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"model_speed": "turbo"}), encoding="utf-8")
        s = load(p)
        assert s.model_speed == "normal"

    def test_out_of_range_mic_gain_falls_back(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"mic_gain": 99.0}), encoding="utf-8")
        s = load(p)
        assert s.mic_gain == 1.0

    def test_mic_gain_below_minimum_falls_back(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"mic_gain": 0.0}), encoding="utf-8")
        s = load(p)
        assert s.mic_gain == 1.0

    def test_unknown_ui_language_falls_back_to_en(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"ui_language": "zz"}), encoding="utf-8")
        s = load(p)
        assert s.ui_language == "en"

    def test_unknown_engine_type_falls_back_to_whisperlive(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"engine_type": "magic_engine"}), encoding="utf-8")
        s = load(p)
        assert s.engine_type == "whisperlive"

    def test_missing_prompt_keys_filled_from_defaults(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"prompts": {}}), encoding="utf-8")
        s = load(p)
        for lang in DEFAULT_PROMPTS:
            assert lang in s.prompts

    def test_null_prompts_treated_as_empty(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"prompts": None}), encoding="utf-8")
        s = load(p)
        assert isinstance(s.prompts, dict)

    def test_input_device_none_when_missing(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text("{}", encoding="utf-8")
        s = load(p)
        assert s.input_device is None


class TestRoundtrip:
    def test_save_then_load_preserves_all_fields(self, tmp_path):
        p = tmp_path / "settings.json"
        original = Settings(
            language="English",
            model_speed="normal",
            mic_gain=2.5,
            ui_language="de",
            asr_postprocess=True,
            cleanup_recordings=True,
            engine_type="faster_whisper",
            input_device=3,
        )
        save(original, p)
        loaded = load(p)

        assert loaded.language      == original.language
        assert loaded.model_speed   == original.model_speed
        assert loaded.mic_gain      == original.mic_gain
        assert loaded.ui_language   == original.ui_language
        assert loaded.asr_postprocess == original.asr_postprocess
        assert loaded.cleanup_recordings == original.cleanup_recordings
        assert loaded.engine_type   == original.engine_type
        assert loaded.input_device  == original.input_device

    def test_save_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "settings.json"
        save(Settings(), p)
        assert p.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        p = tmp_path / "settings.json"
        save(Settings(), p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_prompts_roundtrip(self, tmp_path):
        p = tmp_path / "settings.json"
        custom = {"English": "Speak clearly.", "Deutsch": "Bitte deutlich sprechen."}
        s = Settings(prompts=custom)
        save(s, p)
        loaded = load(p)
        assert loaded.prompts["English"] == "Speak clearly."
        assert loaded.prompts["Deutsch"] == "Bitte deutlich sprechen."
