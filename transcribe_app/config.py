"""Compile-time constants. Nothing here depends on user settings or the UI."""

import sys
from pathlib import Path

# ── OS detection ───────────────────────────────────────────────────────────────
IS_WINDOWS: bool = sys.platform == "win32"

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
# Windows audio drivers (WASAPI/DirectSound) introduce more scheduling jitter
# than the Linux ALSA/PipeWire stack.  With CHUNK_SECONDS = 0.1 the sounddevice
# callback fires 10× per second; on Windows that high frequency increases the
# risk that a brief CPU stall (antivirus scan, telemetry flush, UI redraw) causes
# a callback to be late or skipped, silently dropping ~100 ms of audio.  A
# missing 20–100 ms fragment is enough to confuse Whisper into a wrong word or a
# repetition loop, because the autoregressive model feeds its previous output
# back as context for the next token.
#
# Using 0.5 s blocks on Windows means the callback fires only twice per second,
# making it far less likely that any single OS interruption lands during a
# callback window.  The larger block also hands the model more audio context per
# inference, which helps the VAD gate silence more reliably.
CHUNK_SECONDS = 0.5 if IS_WINDOWS else 0.3

# ── Language definitions ───────────────────────────────────────────────────────
LANGUAGE_OPTS: dict[str, dict] = {
    "English": dict(
        # fast  → small (CPU) or medium (GPU)
        # normal→ medium (CPU) or large-v3-turbo (GPU)
        # best  → large-v3-turbo (CPU) or large-v3 (GPU)
        model_sizes_cpu={"fast": "small.en",  "normal": "medium.en", "best": "large-v3-turbo"},
        model_sizes_gpu={"fast": "medium.en", "normal": "large-v3-turbo", "best": "large-v3"},
        fallback_model_size="small.en",
        lan="en",
    ),
    "Deutsch": dict(
        model_sizes_cpu={"fast": "small",  "normal": "medium", "best": "large-v3-turbo"},
        model_sizes_gpu={"fast": "medium", "normal": "large-v3-turbo", "best": "large-v3"},
        fallback_model_size="small",
        lan="de",
    ),
}


def get_model_size(lang: str, speed: str, use_gpu: bool = GPU) -> str:
    """Return the model-size string for the given language, speed ('fast'/'normal'), and device."""
    opts = LANGUAGE_OPTS[lang]
    return opts["model_sizes_gpu" if use_gpu else "model_sizes_cpu"][speed]

# ── Application data directories ──────────────────────────────────────────────
APP_DATA_DIR:     Path = (
    Path.home() / "AppData" / "Roaming" / "transcribe_app"
    if IS_WINDOWS
    else Path.home() / ".config" / "transcribe_app"
)
SESSIONS_DIR:     Path = APP_DATA_DIR / "sessions"
DIFF_DIR:         Path = APP_DATA_DIR / "diff"
VAD_SNIPPETS_DIR: Path = APP_DATA_DIR / "vad_snippets"

# ── Audio processing ───────────────────────────────────────────────────────────
NOISE_GATE_RMS: float = 0.005   # ≈ −46 dBFS; frames quieter than this are zeroed

DEFAULT_LANGUAGE = "Deutsch"

SPACE_HOLD_TIME_MS = 300

# Default static prompts fed to Whisper before every audio chunk.
# A prompt ending mid-sentence with a comma primes Whisper to treat short
# pauses as continuation rather than sentence boundaries.
DEFAULT_PROMPTS: dict[str, str] = {
    "English": "",
    "Deutsch": (
        # Two purposes:
        # 1. Style/pause hint — discourages the model from inserting punctuation
        #    at every disfluency.
        # 2. Hotword / orthography seeding — mentioning ß, Umlaute, and common
        #    compound words in the prompt shifts the tokenizer's prior toward the
        #    correct Unicode characters.  Without this, faster-whisper sometimes
        #    transcribes "Straße" as "Strasse" or "Überblick" as "Ueberblick".
        "Der Nutzer spricht fließend Deutsch mit Umlauten: ä, ö, ü, Ä, Ö, Ü, ß. "
        "Dabei werden Denkpausen nicht als Satzenden gewertet, sondern mit Leerzeichen ersetzt. "
        "Es gibt die folgenden Sprachbefehle: neue Zeile, neuer Absatz, erstens, zweitens, drittens."
    ),
}


