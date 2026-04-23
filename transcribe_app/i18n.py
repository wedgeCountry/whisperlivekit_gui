"""UI localisation: t(key) → translated string in the active interface language.

Usage
-----
    from transcribe_app.i18n import t, set_language

    set_language("de")          # switch to German
    label = t("btn.record")     # → "⏺  Aufnahme"
    msg   = t("status.saved", name="doc.txt")  # → "Gespeichert: doc.txt"

The active language is module-level state initialised to "en".  It is set once
at startup from Settings.ui_language and updated on user change.  Reads from
the engine's asyncio thread are safe under the GIL.
"""

from __future__ import annotations

_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # ── Main window ────────────────────────────────────────────────────────
        "window.title":           "Live Transcription",
        "menu.file":              "File",
        "menu.file.save":         "Save as file\u2026",
        "menu.file.load":         "Load\u2026",
        "menu.edit":              "Edit",
        "menu.edit.settings":     "Settings\u2026",
        "header.language":        "Language",
        "header.model":           "Model",
        "header.interface":       "UI",
        "speed.fast":             "Fast",
        "speed.normal":           "Normal",
        "speed.best":             "Best",
        "btn.record":             "\u23fa  Record",
        "btn.stop":               "\u23f9  Stop",
        "btn.clear":              "Clear",
        "btn.copy":               "Copy",
        "btn.mic_test":           "Test Mic",
        "status.recording":       "Recording\u2026",
        "status.processing":      "Processing remaining audio\u2026",
        "status.copied":          "Copied to clipboard \u2713",
        "status.saved":           "Saved: {name}",
        "status.loaded":          "Loaded: {name}",
        "status.ready":           "Ready  \xb7  {lang} model loaded",
        "file.type.text":         "Text file",
        "file.type.all":          "All files",
        # ── Engine status (produced in background thread) ──────────────────────
        "status.warmup":          "Warming up: {model}  ({lang} \xb7 {device})\u2026",
        "status.loading":         "Loading: {model}  ({lang} \xb7 {device})\u2026",
        "status.downloading":     "Downloading: {model}  ({lang} \xb7 {device})\u2026",
        "status.error":           "Error: {exc}",
        "status.error_fallback":  "Error loading ({exc_type}), fallback: {status}",
        "status.critical_error":  "Critical error: {exc}",
        "status.warn_no_avx512":  "Ready  \xb7  {lang} model loaded  \u26a0 No AVX-512 \u2014 performance may be reduced",
        "status.retranscribing":  "Re-transcribing audio\u2026",
        "status.diff_saved":      "Re-transcribed  \xb7  diff: {name}",
        "status.audio_clipping":  "⚠ Audio clipping — reduce microphone volume or gain",
        "status.wasapi_shared":   "⚠ WASAPI exclusive mode unavailable — Windows audio enhancements active",
        # ── Settings dialog ────────────────────────────────────────────────────
        "dlg.settings.asr":         "ASR post-processing:",
        "dlg.settings.title":       "Settings",
        "dlg.settings.language":    "Language:",
        "dlg.settings.model":       "Model:",
        "dlg.settings.device":      "Device:",
        "dlg.settings.ui_language": "Interface:",
        "dlg.settings.engine":      "Engine:",
        "dlg.settings.vad_silence":  "VAD silence gap (s):",
        "dlg.settings.vad_aggressiveness": "VAD aggressiveness:",
        "dlg.settings.prompt":      "Style Prompt:",
        "dlg.settings.reset":       "Reset",
        "dlg.settings.save":        "Save",
        # ── Mic test dialog ────────────────────────────────────────────────────
        "dlg.mic.title":          "Microphone Test",
        "dlg.mic.input_device":   "Input device",
        "dlg.mic.default_device": "System default",
        "dlg.mic.level":          "Microphone input level",
        "dlg.mic.sensitivity":    "Sensitivity",
    },
    "de": {
        # ── Main window ────────────────────────────────────────────────────────
        "window.title":           "Live-Transkription",
        "menu.file":              "Datei",
        "menu.file.save":         "Als Datei speichern\u2026",
        "menu.file.load":         "Laden\u2026",
        "menu.edit":              "Bearbeiten",
        "menu.edit.settings":     "Einstellungen\u2026",
        "header.language":        "Sprache",
        "header.model":           "Modell",
        "header.interface":       "UI",
        "speed.fast":             "Schnell",
        "speed.normal":           "Normal",
        "speed.best":             "Beste",
        "btn.record":             "\u23fa  Aufnahme",
        "btn.stop":               "\u23f9  Stop",
        "btn.clear":              "L\xf6schen",
        "btn.copy":               "Kopieren",
        "btn.mic_test":           "Mikrofon testen",
        "status.recording":       "Aufnahme l\xe4uft\u2026",
        "status.processing":      "Verarbeite verbleibende Audiodaten\u2026",
        "status.copied":          "In Zwischenablage kopiert \u2713",
        "status.saved":           "Gespeichert: {name}",
        "status.loaded":          "Geladen: {name}",
        "status.ready":           "Bereit  \xb7  {lang} Modell geladen",
        "file.type.text":         "Textdatei",
        "file.type.all":          "Alle Dateien",
        # ── Engine status (produced in background thread) ──────────────────────
        "status.warmup":          "Aufwärmen: {model}  ({lang} \xb7 {device})\u2026",
        "status.loading":         "Lade: {model}  ({lang} \xb7 {device})\u2026",
        "status.downloading":     "Lade herunter: {model}  ({lang} \xb7 {device})\u2026",
        "status.error":           "Fehler: {exc}",
        "status.error_fallback":  "Fehler beim Laden ({exc_type}), Fallback: {status}",
        "status.critical_error":  "Kritischer Fehler: {exc}",
        "status.warn_no_avx512":  "Bereit  \xb7  {lang} Modell geladen  \u26a0 Kein AVX-512 \u2014 Leistung eingeschr\xe4nkt",
        "status.retranscribing":  "Audio wird erneut transkribiert\u2026",
        "status.diff_saved":      "Neu transkribiert  \xb7  diff: {name}",
        "status.audio_clipping":  "⚠ Audio-Clipping — Mikrofonlautstärke oder Verstärkung reduzieren",
        "status.wasapi_shared":   "⚠ WASAPI-Exklusivmodus nicht verfügbar — Windows-Audioverbesserungen aktiv",
        # ── Settings dialog ────────────────────────────────────────────────────
        "dlg.settings.asr":         "ASR-Nachbearbeitung:",
        "dlg.settings.title":       "Einstellungen",
        "dlg.settings.language":    "Sprache:",
        "dlg.settings.model":       "Modell:",
        "dlg.settings.device":      "Ger\xe4t:",
        "dlg.settings.ui_language": "Oberfl\xe4che:",
        "dlg.settings.engine":      "Engine:",
        "dlg.settings.vad_silence":  "VAD-Stille-Pause (s):",
        "dlg.settings.vad_aggressiveness": "VAD-Aggressivität:",
        "dlg.settings.prompt":      "Style Prompt:",
        "dlg.settings.reset":       "Zur\xfccksetzen",
        "dlg.settings.save":        "Speichern",
        # ── Mic test dialog ────────────────────────────────────────────────────
        "dlg.mic.title":          "Mikrofon-Test",
        "dlg.mic.input_device":   "Eingabeger\xe4t",
        "dlg.mic.default_device": "Systemstandard",
        "dlg.mic.level":          "Mikrofon-Eingangspegel",
        "dlg.mic.sensitivity":    "Empfindlichkeit",
    },
}

# BCP-47 code → display name shown in the UI combo.
UI_LANGUAGES: dict[str, str] = {"en": "English", "de": "Deutsch"}

_lang: str = "en"


def set_language(lang: str) -> None:
    """Switch the active UI language.  Call from the UI thread."""
    global _lang
    if lang in _STRINGS:
        _lang = lang


def get_language() -> str:
    return _lang


def t(key: str, **kwargs: object) -> str:
    """Return the UI string for *key* in the active language.

    Falls back to English if the key is missing in the active language,
    and returns *key* itself if it is missing from English too.
    """
    s = _STRINGS.get(_lang, _STRINGS["en"]).get(key) or _STRINGS["en"].get(key, key)
    return s.format(**kwargs) if kwargs else s
