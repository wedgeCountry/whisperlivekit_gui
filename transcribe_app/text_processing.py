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
    (re.compile(r"\s+\berstens\b[,.]?\s*",      re.I), "\n1. "),
    (re.compile(r"\s+\bzweitens\b[,.]?\s*",     re.I), "\n2. "),
    (re.compile(r"\s+\bdrittens\b[,.]?\s*",     re.I), "\n3. "),
    (re.compile(r"\s+\bviertens\b[,.]?\s*",     re.I), "\n4. "),
    (re.compile(r"\s+\bfünftens\b[,.]?\s*",     re.I), "\n5. "),
    (re.compile(r"\s+\bsechstens\b[,.]?\s*",    re.I), "\n6. "),
    (re.compile(r"\s+\bsiebe?n?t?e?ns?\b[,.]?\s*",   re.I), "\n7. "),
    (re.compile(r"\s+\bachtens\b[,.]?\s*",      re.I), "\n8. "),
    (re.compile(r"\s+\bneuntens\b[,.]?\s*",     re.I), "\n9. "),
    (re.compile(r"\s+\bzehntens\b[,.]?\s*",     re.I), "\n10. "),
    (re.compile(r"\s+\bfirst(?:ly)?\b[,.]?\s*", re.I), "\n1. "),
    (re.compile(r"\s+\bsecond(?:ly)?\b[,.]?\s*",re.I), "\n2. "),
    (re.compile(r"\s+\bthird(?:ly)?\b[,.]?\s*", re.I), "\n3. "),
    (re.compile(r"\s+\bfourth(?:ly)?\b[,.]?\s*",re.I), "\n4. "),
    (re.compile(r"\s+\bfifth(?:ly)?\b[,.]?\s*", re.I), "\n5. "),
    (re.compile(r"\s+\bsixth(?:ly)?\b[,.]?\s*", re.I), "\n6. "),
    (re.compile(r"\s+\bseventh(?:ly)?\b[,.]?\s*",re.I),"\n7. "),
    (re.compile(r"\s+\beighth(?:ly)?\b[,.]?\s*",re.I), "\n8. "),
    (re.compile(r"\s+\bninth(?:ly)?\b[,.]?\s*", re.I), "\n9. "),
    (re.compile(r"\s+\btenth(?:ly)?\b[,.]?\s*", re.I), "\n10. "),
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
