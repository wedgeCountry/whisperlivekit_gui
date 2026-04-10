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
    (re.compile(r"(?:(?:^|(?<=[.!?\n]))\s*)heading[.,]?\s+",      re.I | re.M), "\n# "),
    (re.compile(r"(?:(?:^|(?<=[.!?\n]))\s*)überschrift[.,]?\s+",  re.I | re.M), "\n# "),

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
    (re.compile(r"\berstens\b[,.]?\s*",      re.I), "1. "),
    (re.compile(r"\bzweitens\b[,.]?\s*",     re.I), "2. "),
    (re.compile(r"\bdrittens\b[,.]?\s*",     re.I), "3. "),
    (re.compile(r"\bviertens\b[,.]?\s*",     re.I), "4. "),
    (re.compile(r"\bfünftens\b[,.]?\s*",     re.I), "5. "),
    (re.compile(r"\bsechstens\b[,.]?\s*",    re.I), "6. "),
    (re.compile(r"\bsiebte?ns?\b[,.]?\s*",   re.I), "7. "),
    (re.compile(r"\bachtens\b[,.]?\s*",      re.I), "8. "),
    (re.compile(r"\bneuntens\b[,.]?\s*",     re.I), "9. "),
    (re.compile(r"\bzehntens\b[,.]?\s*",     re.I), "10. "),
    (re.compile(r"\bfirst(?:ly)?\b[,.]?\s*", re.I), "1. "),
    (re.compile(r"\bsecond(?:ly)?\b[,.]?\s*",re.I), "2. "),
    (re.compile(r"\bthird(?:ly)?\b[,.]?\s*", re.I), "3. "),
    (re.compile(r"\bfourth(?:ly)?\b[,.]?\s*",re.I), "4. "),
    (re.compile(r"\bfifth(?:ly)?\b[,.]?\s*", re.I), "5. "),
    (re.compile(r"\bsixth(?:ly)?\b[,.]?\s*", re.I), "6. "),
    (re.compile(r"\bseventh(?:ly)?\b[,.]?\s*",re.I),"7. "),
    (re.compile(r"\beighth(?:ly)?\b[,.]?\s*",re.I), "8. "),
    (re.compile(r"\bninth(?:ly)?\b[,.]?\s*", re.I), "9. "),
    (re.compile(r"\btenth(?:ly)?\b[,.]?\s*", re.I), "10. "),
]


def apply_commands(text: str, context: str = "") -> str | None:
    """Apply voice-command substitutions to the last 3 words relative to the cursor.

    context is the text before `text` (e.g. the session prefix up to the cursor) —
    used only to locate the 3-word window correctly; it is never modified.
    Returns the modified string if any pattern matched, or None if nothing changed.
    """
    combined = context + text
    word_spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', combined)]
    text_start = len(context)

    if len(word_spans) <= 3:
        tail_offset = 0
    else:
        tail_offset = max(word_spans[-3][0] - text_start, 0)

    head = text[:tail_offset]
    tail = text[tail_offset:]

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
