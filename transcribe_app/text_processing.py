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


def apply_commands(text: str) -> str | None:
    """Apply voice-command substitutions to the last 3 words only.

    Returns the modified string if any pattern matched, or None if nothing changed.
    """
    word_spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
    if len(word_spans) <= 3:
        head, tail = "", text
    else:
        split_pos = word_spans[-3][0]
        head, tail = text[:split_pos], text[split_pos:]

    new_tail = tail
    for pattern, replacement in _CMDS:
        new_tail = pattern.sub(replacement, new_tail)

    if new_tail == tail:
        return None  # no pattern matched

    new_tail = re.sub(r"\n{3,}", "\n\n", new_tail)
    return (head + new_tail).lstrip("\n")


# ── Prompt-leak removal ────────────────────────────────────────────────────────

def strip_prompt_leak(text: str, prompt: str) -> str:
    """Remove the Whisper static_init_prompt if Whisper hallucinates it back."""
    if not prompt or not text:
        return text
    text = re.sub(re.escape(prompt), " ", text, flags=re.IGNORECASE)
    return re.sub(r"  +", " ", text).strip()
