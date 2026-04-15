"""Tests for transcribe_app.i18n — localisation helpers."""

from transcribe_app.i18n import t, set_language, get_language, UI_LANGUAGES


class TestGetLanguage:
    def test_default_is_english(self):
        assert get_language() == "en"


class TestSetLanguage:
    def test_switch_to_german(self):
        set_language("de")
        assert get_language() == "de"

    def test_unknown_language_ignored(self):
        set_language("fr")
        assert get_language() == "en"  # unchanged (reset_i18n_language fixture)


class TestT:
    def test_english_button_record(self):
        set_language("en")
        assert "Record" in t("btn.record")

    def test_german_button_record(self):
        set_language("de")
        assert "Aufnahme" in t("btn.record")

    def test_format_kwarg(self):
        label = t("status.ready", lang="English")
        assert "English" in label

    def test_format_kwarg_german(self):
        set_language("de")
        label = t("status.ready", lang="Deutsch")
        assert "Deutsch" in label

    def test_missing_key_falls_back_to_key_name(self):
        result = t("this.key.does.not.exist")
        assert result == "this.key.does.not.exist"

    def test_all_english_keys_present(self):
        # Sanity: every key defined in English must return a non-empty string.
        from transcribe_app.i18n import _STRINGS
        set_language("en")
        for key in _STRINGS["en"]:
            assert t(key)  # not empty

    def test_all_german_keys_present(self):
        from transcribe_app.i18n import _STRINGS
        set_language("de")
        for key in _STRINGS["de"]:
            assert t(key)


class TestUILanguages:
    def test_contains_en_and_de(self):
        assert "en" in UI_LANGUAGES
        assert "de" in UI_LANGUAGES

    def test_values_are_display_names(self):
        assert "English" in UI_LANGUAGES.values()
        assert "Deutsch" in UI_LANGUAGES.values()
