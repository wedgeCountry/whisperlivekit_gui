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
        model_size="large-v3-turbo" if GPU else "medium.en",
        fallback_model_size="medium.en",
        lan="en",
    ),
    "Deutsch": dict(
        model_size="large-v3-turbo" if GPU else "medium",
        fallback_model_size="medium",
        lan="de",
    ),
}

DEFAULT_LANGUAGE = "English"

# Default static prompts fed to Whisper before every audio chunk.
# A prompt ending mid-sentence with a comma primes Whisper to treat short
# pauses as continuation rather than sentence boundaries.
DEFAULT_PROMPTS: dict[str, str] = {
    "English": "",
    "Deutsch": (
        "Der Nutzer spricht mit Denkpausen. Wenn er eine Denkpause macht,"
        " dann füge ein Leerzeichen ein."
    ),
}


