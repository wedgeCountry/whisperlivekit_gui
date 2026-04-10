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
    # ── Structural ────────────────────────────────────────────────────────────
    (re.compile(r"[ \t]*\bnewline[.,]?\s*",          re.I),       "\n"),
    (re.compile(r"[ \t]*\bneue\s+zeile[.,]?\s*",     re.I),       "\n"),
    (re.compile(r"[ \t]*\bneuzeile[.,]?\s*",         re.I),       "\n"),   # small-model merges the two words
    (re.compile(r"[ \t]*\bnew\s+paragraph[.,]?\s*",  re.I),       "\n\n"),
    (re.compile(r"[ \t]*\bneuer?\s+absatz[.,]?\s*",  re.I),       "\n\n"),
    (re.compile(r"(?:(?:^|(?<=[.!?\n]))\s*)heading[.,]?\s+",      re.I | re.M), "\n\n# "),
    (re.compile(r"(?:(?:^|(?<=[.!?\n]))\s*)überschrift[.,]?\s+",  re.I | re.M), "\n\n# "),

    # ── Punctuation (DE + EN) ─────────────────────────────────────────────────
    (re.compile(r"\bpunkt\b,?\s*",           re.I), ". "),
    (re.compile(r"\bperiod\b,?\s*",          re.I), ". "),
    (re.compile(r"\bkomma\b,?\s*",           re.I), ", "),
    (re.compile(r"\bcomma\b,?\s*",           re.I), ", "),
    (re.compile(r"\bfragezeichen\b,?\s*",    re.I), "? "),
    (re.compile(r"\bquestion\s+mark\b,?\s*", re.I), "? "),
    (re.compile(r"\bbindestrich\b,?\s*",     re.I), "-"),
    (re.compile(r"\bhyphen\b,?\s*",          re.I), "-"),
    (re.compile(r"\bdash\b,?\s*",            re.I), "-"),
    (re.compile(r"\bausrufezeichen\b,?\s*",  re.I), "! "),
    (re.compile(r"\bexclamation\s+mark\b,?\s*", re.I), "! "),
    (re.compile(r"\bdoppelpunkt\b,?\s*",     re.I), ": "),
    (re.compile(r"\bcolon\b,?\s*",           re.I), ": "),
    (re.compile(r"\bsemikolon\b,?\s*",       re.I), "; "),
    (re.compile(r"\bsemicolon\b,?\s*",       re.I), "; "),

    # ── Ordinal list markers (DE + EN) ────────────────────────────────────────
    (re.compile(r"\b\s+erstens\b[,.]?\s*",      re.I), "\n1. "),
    (re.compile(r"\b\s+zweitens\b[,.]?\s*",     re.I), "\n2. "),
    (re.compile(r"\b\s+drittens\b[,.]?\s*",     re.I), "\n3. "),
    (re.compile(r"\b\s+viertens\b[,.]?\s*",     re.I), "\n4. "),
    (re.compile(r"\b\s+fünftens\b[,.]?\s*",     re.I), "\n5. "),
    (re.compile(r"\b\s+sechstens\b[,.]?\s*",    re.I), "\n6. "),
    (re.compile(r"\b\s+siebe?n?t?e?ns?\b[,.]?\s*",   re.I), "\n7. "),
    (re.compile(r"\b\s+achtens\b[,.]?\s*",      re.I), "\n8. "),
    (re.compile(r"\b\s+neuntens\b[,.]?\s*",     re.I), "\n9. "),
    (re.compile(r"\b\s+zehntens\b[,.]?\s*",     re.I), "\n10. "),
    (re.compile(r"\b\s+first(?:ly)?\b[,.]?\s*", re.I), "\n1. "),
    (re.compile(r"\b\s+second(?:ly)?\b[,.]?\s*",re.I), "\n2. "),
    (re.compile(r"\b\s+third(?:ly)?\b[,.]?\s*", re.I), "\n3. "),
    (re.compile(r"\b\s+fourth(?:ly)?\b[,.]?\s*",re.I), "\n4. "),
    (re.compile(r"\b\s+fifth(?:ly)?\b[,.]?\s*", re.I), "\n5. "),
    (re.compile(r"\b\s+sixth(?:ly)?\b[,.]?\s*", re.I), "\n6. "),
    (re.compile(r"\b\s+seventh(?:ly)?\b[,.]?\s*",re.I),"\n7. "),
    (re.compile(r"\b\s+eighth(?:ly)?\b[,.]?\s*",re.I), "\n8. "),
    (re.compile(r"\b\s+ninth(?:ly)?\b[,.]?\s*", re.I), "\n9. "),
    (re.compile(r"\b\s+tenth(?:ly)?\b[,.]?\s*", re.I), "\n10. "),
]


def apply_commands_full(text: str) -> str | None:
    """Apply all voice-command substitutions across the entire text at once.

    Used for post-processing after recording ends.  Every occurrence
    throughout the text is replaced — no windowing or length limit.
    Returns the modified string, or None if nothing changed.
    """
    new_text = text
    for pattern, replacement in _CMDS:
        new_text = pattern.sub(replacement, new_text)
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    if new_text == text:
        return None
    return new_text


# ── Prompt-leak removal ────────────────────────────────────────────────────────

def strip_prompt_leak(text: str, prompt: str) -> str:
    """Remove the Whisper static_init_prompt if Whisper hallucinates it back."""
    if not prompt or not text:
        return text
    text = re.sub(re.escape(prompt), " ", text, flags=re.IGNORECASE)
    return re.sub(r"  +", " ", text).strip()
