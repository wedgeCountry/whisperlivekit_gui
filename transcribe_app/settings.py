"""User-editable settings: typed dataclass + JSON persistence."""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from transcribe_app.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_PROMPTS,
    GPU,
    LANGUAGE_OPTS,
    GRAMMAR_LANG_CODES,
)
from transcribe_app.i18n import UI_LANGUAGES

_log = logging.getLogger(__name__)

_CONFIG_DIR = (
    Path.home() / "AppData" / "Roaming" / "transcribe_app"
    if sys.platform == "win32"
    else Path.home() / ".config" / "transcribe_app"
)
_SETTINGS_FILE = _CONFIG_DIR / "settings.json"


@dataclass
class Settings:
    language:       str            = DEFAULT_LANGUAGE
    prompts:        dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROMPTS))
    input_device:   int | None     = None   # sounddevice device index; None = system default
    model_speed:    str            = "fast"    # "fast" or "normal"
    mic_gain:       float          = 1.0       # linear amplitude multiplier applied before transcription
    ui_language:    str            = "en"      # interface language code; see i18n.UI_LANGUAGES
    compute_device:     str            = "cuda" if GPU else "cpu"  # "cuda" or "cpu"
    grammar_correction: bool           = False  # run post-processing through a grammar model


def _fill_prompts(raw: object) -> dict[str, str]:
    """Return a prompts dict guaranteed to have a key for every language in LANGUAGE_OPTS.

    A missing key is filled with the DEFAULT_PROMPTS value.  A non-dict value
    (e.g. ``null`` from hand-edited JSON) is treated as empty.
    """
    base = raw if isinstance(raw, dict) else {}
    return {k: str(base.get(k, DEFAULT_PROMPTS[k])) for k in LANGUAGE_OPTS}


def load() -> Settings:
    try:
        data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Settings()  # expected on first run
    except Exception:
        _log.warning("Could not load settings from %s — using defaults", _SETTINGS_FILE, exc_info=True)
        return Settings()

    lang = data.get("language", DEFAULT_LANGUAGE)
    if lang not in LANGUAGE_OPTS:
        lang = DEFAULT_LANGUAGE

    raw_device = data.get("input_device")
    input_device = int(raw_device) if isinstance(raw_device, (int, float)) else None

    raw_speed = data.get("model_speed", "normal")
    model_speed = raw_speed if raw_speed in ("fast", "normal") else "normal"

    raw_gain = data.get("mic_gain", 1.0)
    mic_gain = float(raw_gain) if isinstance(raw_gain, (int, float)) and 0.1 <= raw_gain <= 5.0 else 1.0

    raw_ui_lang = data.get("ui_language", "en")
    ui_language = raw_ui_lang if raw_ui_lang in UI_LANGUAGES else "en"

    raw_compute = data.get("compute_device", "cuda" if GPU else "cpu")
    # "cuda" is only valid when GPU hardware is actually present
    compute_device = raw_compute if raw_compute in ("cuda", "cpu") and (raw_compute != "cuda" or GPU) else ("cuda" if GPU else "cpu")

    grammar_correction = bool(data.get("grammar_correction", False))
    # Silently disable if no language code is defined for the saved language
    if grammar_correction and lang not in GRAMMAR_LANG_CODES:
        grammar_correction = False

    return Settings(
        language=lang,
        prompts=_fill_prompts(data.get("prompts")),
        input_device=input_device,
        model_speed=model_speed,
        mic_gain=mic_gain,
        ui_language=ui_language,
        compute_device=compute_device,
        grammar_correction=grammar_correction,
    )


def save(s: Settings) -> None:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(
            json.dumps(
                {
                    "language":           s.language,
                    "prompts":            s.prompts,
                    "input_device":       s.input_device,
                    "model_speed":        s.model_speed,
                    "mic_gain":           s.mic_gain,
                    "ui_language":        s.ui_language,
                    "compute_device":     s.compute_device,
                    "grammar_correction": s.grammar_correction,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        _log.error("Failed to save settings to %s", _SETTINGS_FILE, exc_info=True)
