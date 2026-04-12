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
        model_sizes_cpu={"fast": "small.en",  "normal": "medium.en"},
        model_sizes_gpu={"fast": "medium.en", "normal": "large-v3-turbo"},
        fallback_model_size="small.en",
        lan="en",
    ),
    "Deutsch": dict(
        model_sizes_cpu={"fast": "small",  "normal": "medium"},
        model_sizes_gpu={"fast": "medium", "normal": "large-v3-turbo"},
        fallback_model_size="small",
        lan="de",
    ),
}


def get_model_size(lang: str, speed: str, use_gpu: bool = GPU) -> str:
    """Return the model-size string for the given language, speed ('fast'/'normal'), and device."""
    opts = LANGUAGE_OPTS[lang]
    return opts["model_sizes_gpu" if use_gpu else "model_sizes_cpu"][speed]

DEFAULT_LANGUAGE = "Deutsch"

SPACE_HOLD_TIME_MS = 300

# Default static prompts fed to Whisper before every audio chunk.
# A prompt ending mid-sentence with a comma primes Whisper to treat short
# pauses as continuation rather than sentence boundaries.
DEFAULT_PROMPTS: dict[str, str] = {
    "English": "",
    "Deutsch": (

        "Der Nutzer macht Denkpausen, füge da ein Leerzeichen statt Satzzeichen ein."
        "Sprachbefehle: neue Zeile, neuer Absatz, Satzzeichen, Erstens, zweitens, drittens..."
    ),
}


