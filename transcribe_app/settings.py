"""User-editable settings: typed dataclass + JSON persistence."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from transcribe_app.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_PROMPTS,
    LANGUAGE_OPTS,
)
from transcribe_app.i18n import UI_LANGUAGES

_CONFIG_DIR = (
    Path.home() / "AppData" / "Roaming" / "transcribe_app"
    if sys.platform == "win32"
    else Path.home() / ".config" / "transcribe_app"
)
_SETTINGS_FILE = _CONFIG_DIR / "settings.json"


@dataclass
class Settings:
    language:     str            = DEFAULT_LANGUAGE
    prompts:      dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROMPTS))
    input_device: int | None     = None   # sounddevice device index; None = system default
    model_speed:  str            = "fast"    # "fast" or "normal"
    mic_gain:     float          = 1.0       # linear amplitude multiplier applied before transcription
    ui_language:  str            = "en"      # interface language code; see i18n.UI_LANGUAGES


def load() -> Settings:
    try:
        data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
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

    return Settings(
        language=lang,
        prompts={
            k: data.get("prompts", {}).get(k, DEFAULT_PROMPTS[k])
            for k in LANGUAGE_OPTS
        },
        input_device=input_device,
        model_speed=model_speed,
        mic_gain=mic_gain,
        ui_language=ui_language,
    )


def save(s: Settings) -> None:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(
            json.dumps(
                {
                    "language":     s.language,
                    "prompts":      s.prompts,
                    "input_device": s.input_device,
                    "model_speed":  s.model_speed,
                    "mic_gain":     s.mic_gain,
                    "ui_language":  s.ui_language,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
