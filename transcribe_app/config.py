"""Compile-time constants. Nothing here depends on user settings or the UI."""

from pathlib import Path

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

DEFAULT_SYSTEM_PROMPTS: dict[str, str] = {
    "English": (
        "You are a precise text editor. The user gives you transcribed speech text "
        "that may contain typos and punctuation errors introduced by automatic speech "
        "recognition. Correct only clear typos and obviously wrong punctuation. "
        "Preserve all markdown formatting (# headings, blank-line paragraphs). "
        "Do not rephrase, add, or remove any content. "
        "Return only the corrected text — no explanations, no surrounding quotes."
    ),
    "Deutsch": (
        "Du bist ein präziser Texteditor. Der Benutzer gibt dir transkribierten Sprachtext, "
        "der Tippfehler und Interpunktionsfehler durch automatische Spracherkennung enthalten kann. "
        "Korrigiere nur eindeutige Tippfehler und offensichtlich falsche Interpunktion. "
        "Behalte alle Markdown-Formatierungen (# Überschriften, Leerzeilen-Absätze) bei. "
        "Formuliere nicht um, füge nichts hinzu und entferne keinen Inhalt. "
        "Gib nur den korrigierten Text zurück – keine Erklärungen, keine Anführungszeichen."
    ),
}

# ── Correction backends ────────────────────────────────────────────────────────
CORRECTION_BACKENDS = ["Ollama", "LanguageTool", "Spylls"]
DEFAULT_BACKEND     = "Ollama"

LT_LANG_CODES: dict[str, str] = {
    "English": "en-US",
    "Deutsch": "de-DE",
}

HUNSPELL_CODES: dict[str, str] = {
    "English": "en_US",
    "Deutsch": "de_DE",
}

HUNSPELL_SEARCH_DIRS: list[Path] = [
    Path("/usr/share/hunspell"),
    Path("/usr/share/myspell/dicts"),
    Path("/usr/share/myspell"),
    Path.home() / ".local" / "share" / "hunspell",
]

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"

DEFAULT_OLLAMA_MODELS: dict[str, str] = {
    "English": "gemma",
    "Deutsch": "sroecker/sauerkrautlm-7b-hero:latest",
}
