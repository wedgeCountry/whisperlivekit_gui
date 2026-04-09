"""User-editable settings: typed dataclass + JSON persistence."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from transcribe_app.config import (
    DEFAULT_BACKEND,
    DEFAULT_LANGUAGE,
    DEFAULT_OLLAMA_MODELS,
    DEFAULT_PROMPTS,
    DEFAULT_SYSTEM_PROMPTS,
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
    language:           str            = DEFAULT_LANGUAGE
    prompts:            dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROMPTS))
    system_prompts:     dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SYSTEM_PROMPTS))
    ollama_models:      dict[str, str] = field(default_factory=lambda: dict(DEFAULT_OLLAMA_MODELS))
    correction_backend: str            = DEFAULT_BACKEND


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
        system_prompts={
            k: data.get("system_prompts", {}).get(k, DEFAULT_SYSTEM_PROMPTS[k])
            for k in LANGUAGE_OPTS
        },
        ollama_models={
            k: data.get("ollama_models", {}).get(k, DEFAULT_OLLAMA_MODELS[k])
            for k in LANGUAGE_OPTS
        },
        correction_backend=data.get("correction_backend", DEFAULT_BACKEND),
    )


def save(s: Settings) -> None:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(
            json.dumps(
                {
                    "language":           s.language,
                    "prompts":            s.prompts,
                    "system_prompts":     s.system_prompts,
                    "ollama_models":      s.ollama_models,
                    "correction_backend": s.correction_backend,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
