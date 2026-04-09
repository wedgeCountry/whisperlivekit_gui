"""Pure text-processing helpers — no UI, no engine, no I/O."""

import re

# ── Whisper artefact cleanup ───────────────────────────────────────────────────

def clean(text: str) -> str:
    """Remove filler artefacts Whisper inserts at pauses."""
    text = text.replace("…", " ").replace("...", " ")
    return re.sub(r"  +", " ", text)


# ── Voice command → markdown substitutions ────────────────────────────────────
# Commands are matched case-insensitively.  "heading" / "Überschrift" fire only
# at the start of a segment to avoid accidental matches in natural speech.

_CMDS: list[tuple] = [
    (re.compile(r"[ \t]*\bnewline[.,]?\s*",          re.I),       "\n"),
    (re.compile(r"[ \t]*\bneue\s+zeile[.,]?\s*",     re.I),       "\n"),
    (re.compile(r"[ \t]*\bnew\s+paragraph[.,]?\s*",  re.I),       "\n\n"),
    (re.compile(r"[ \t]*\bneuer?\s+absatz[.,]?\s*",  re.I),       "\n\n"),
    (re.compile(r"(?:(?:^|(?<=[.!?\n]))\s*)heading[.,]?\s+",      re.I | re.M), "\n# "),
    (re.compile(r"(?:(?:^|(?<=[.!?\n]))\s*)überschrift[.,]?\s+",  re.I | re.M), "\n# "),
]


def apply_commands(text: str) -> str:
    for pattern, replacement in _CMDS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.lstrip("\n")


# ── Prompt-leak removal ────────────────────────────────────────────────────────

def strip_prompt_leak(text: str, prompt: str) -> str:
    """Remove the Whisper static_init_prompt if Whisper hallucinates it back."""
    if not prompt or not text:
        return text
    text = re.sub(re.escape(prompt), " ", text, flags=re.IGNORECASE)
    return re.sub(r"  +", " ", text).strip()


# ── Ollama output cleanup ──────────────────────────────────────────────────────

_OLLAMA_JUNK = re.compile(
    r"^(?:"
    # English preambles
    r"here\s+is\s+(?:the\s+)?(?:corrected\s+)?(?:text|version)[^:]*:\s*|"
    r"(?:the\s+)?(?:corrected|revised|edited)\s+(?:text|version)[^:]*:\s*|"
    r"(?:i\s+have\s+)?(?:corrected|revised|fixed)[^:\n]*:\s*|"
    r"(?:sure|certainly|of\s+course)[^:\n]*[:.]\s*|"
    # German preambles
    r"hier\s+ist\s+(?:der\s+)?(?:korrigierte\s+)?(?:text|version)[^:]*:\s*|"
    r"(?:der\s+)?(?:korrigierte|bearbeitete|überarbeitete)\s+(?:text|version)[^:]*:\s*|"
    r"ich\s+habe[^:\n]*(?:korrigiert|bearbeitet|überarbeitet)[^:\n]*[:.]\s*|"
    r"(?:sicher|natürlich|gerne)[^:\n]*[:.]\s*"
    r")+",
    re.IGNORECASE | re.DOTALL,
)


def strip_ollama_junk(text: str, system_prompt: str) -> str:
    """Remove system-prompt echo and common LLM preambles from Ollama output."""
    if system_prompt:
        text = re.sub(re.escape(system_prompt), "", text, flags=re.IGNORECASE).strip()
    text = _OLLAMA_JUNK.sub("", text).strip()
    # Strip surrounding markdown code fences the model sometimes adds
    text = re.sub(r"^```[^\n]*\n?", "", text).strip()
    text = re.sub(r"\n?```$",        "", text).strip()
    return text
