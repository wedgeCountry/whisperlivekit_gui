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

_CONFIG_DIR = (
    Path.home() / "AppData" / "Roaming" / "transcribe_app"
    if sys.platform == "win32"
    else Path.home() / ".config" / "transcribe_app"
)
_SETTINGS_FILE = _CONFIG_DIR / "settings.json"


@dataclass
class Settings:
    language: str            = DEFAULT_LANGUAGE
    prompts:  dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROMPTS))


def load() -> Settings:
    try:
        data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return Settings()

    lang = data.get("language", DEFAULT_LANGUAGE)
    if lang not in LANGUAGE_OPTS:
        lang = DEFAULT_LANGUAGE

    return Settings(
        language=lang,
        prompts={
            k: data.get("prompts", {}).get(k, DEFAULT_PROMPTS[k])
            for k in LANGUAGE_OPTS
        },
    )


def save(s: Settings) -> None:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(
            json.dumps(
                {
                    "language": s.language,
                    "prompts":  s.prompts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
