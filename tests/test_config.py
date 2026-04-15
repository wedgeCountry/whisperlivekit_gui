"""Tests for transcribe_app.config — model-size selection logic."""

import pytest

from transcribe_app.config import get_model_size, LANGUAGE_OPTS


class TestGetModelSize:
    def test_english_fast_cpu(self):
        assert get_model_size("English", "fast", use_gpu=False) == "small.en"

    def test_english_normal_cpu(self):
        assert get_model_size("English", "normal", use_gpu=False) == "medium.en"

    def test_english_fast_gpu(self):
        assert get_model_size("English", "fast", use_gpu=True) == "medium.en"

    def test_english_normal_gpu(self):
        assert get_model_size("English", "normal", use_gpu=True) == "large-v3-turbo"

    def test_deutsch_fast_cpu(self):
        assert get_model_size("Deutsch", "fast", use_gpu=False) == "small"

    def test_deutsch_normal_cpu(self):
        assert get_model_size("Deutsch", "normal", use_gpu=False) == "medium"

    def test_deutsch_fast_gpu(self):
        assert get_model_size("Deutsch", "fast", use_gpu=True) == "medium"

    def test_deutsch_normal_gpu(self):
        assert get_model_size("Deutsch", "normal", use_gpu=True) == "large-v3-turbo"

    def test_unknown_language_raises(self):
        with pytest.raises(KeyError):
            get_model_size("Spanish", "fast", use_gpu=False)

    def test_all_languages_have_both_speeds(self):
        for lang in LANGUAGE_OPTS:
            for speed in ("fast", "normal"):
                for gpu in (True, False):
                    result = get_model_size(lang, speed, use_gpu=gpu)
                    assert isinstance(result, str) and result
