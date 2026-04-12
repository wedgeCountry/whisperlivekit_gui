"""Grammar correction via LanguageTool (language_tool_python).

A LanguageTool instance manages a Java subprocess.  It is expensive to start
and must be reused — never create one per call.  Tools are cached per language
code and torn down cleanly on shutdown.
"""

from __future__ import annotations

import threading

_TOOLS: dict[str, object] = {}   # lang_code → LanguageTool instance
_LOCK = threading.Lock()


def is_loaded(lang_code: str) -> bool:
    """Return True if the LanguageTool for *lang_code* is already running."""
    return lang_code in _TOOLS


def _build(lang_code: str) -> object:
    import language_tool_python  # noqa: PLC0415
    return language_tool_python.LanguageTool(lang_code)


def get_tool(lang_code: str) -> object:
    """Return a cached LanguageTool, starting it on the first call (blocking)."""
    if lang_code not in _TOOLS:
        with _LOCK:
            if lang_code not in _TOOLS:
                _TOOLS[lang_code] = _build(lang_code)
    return _TOOLS[lang_code]


def correct(text: str, lang_code: str) -> str:
    """Return grammar-corrected *text* for *lang_code* (e.g. "de-DE", "en-US").

    Raises on tool-start failure — callers should catch and fall back to the
    original text.
    """
    tool = get_tool(lang_code)
    return tool.correct(text)  # type: ignore[union-attr]


def unload_all() -> None:
    """Close all LanguageTool subprocesses (call on shutdown)."""
    with _LOCK:
        for tool in _TOOLS.values():
            try:
                tool.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
        _TOOLS.clear()
