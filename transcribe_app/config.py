"""Compile-time constants. Nothing here depends on user settings or the UI."""


# ── GPU detection ──────────────────────────────────────────────────────────────
try:
    import torch
    GPU: bool = torch.cuda.is_available()
except Exception:
    GPU = False

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHANNELS      = 1
DTYPE         = "int16"
CHUNK_SECONDS = 0.1

# ── Language definitions ───────────────────────────────────────────────────────
LANGUAGE_OPTS: dict[str, dict] = {
    "English": dict(
        # fast  → small (CPU) or medium (GPU)
        # normal→ medium (CPU) or large-v3-turbo (GPU)
        model_sizes={
            "fast":   "small.en"         if not GPU else "medium.en",
            "normal": "medium.en"        if not GPU else "large-v3-turbo",
        },
        fallback_model_size="medium.en",
        lan="en",
    ),
    "Deutsch": dict(
        model_sizes={
            "fast":   "small"            if not GPU else "medium",
            "normal": "medium"           if not GPU else "large-v3-turbo",
        },
        fallback_model_size="medium",
        lan="de",
    ),
}


def get_model_size(lang: str, speed: str) -> str:
    """Return the model-size string for the given language and speed ('fast'/'normal')."""
    return LANGUAGE_OPTS[lang]["model_sizes"][speed]

DEFAULT_LANGUAGE = "Deutsch"

# Default static prompts fed to Whisper before every audio chunk.
# A prompt ending mid-sentence with a comma primes Whisper to treat short
# pauses as continuation rather than sentence boundaries.
DEFAULT_PROMPTS: dict[str, str] = {
    "English": "",
    "Deutsch": (
        "Diktat mit Sprachbefehlen: neue Zeile, neuer Absatz, Punkt, Komma, "
        "Fragezeichen, Ausrufezeichen, Bindestrich, Doppelpunkt, Semikolon. "
        "Erstens, zweitens, drittens, viertens, fünftens."
        "Der Nutzer spricht mit Denkpausen. Wenn er eine Denkpause macht, dann füge ein Leerzeichen ein."
    ),
}


