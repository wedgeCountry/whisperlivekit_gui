"""Correction backend protocol and factory.

Usage::

    corrector = get_corrector(backend, lang, settings)
    corrected = await corrector.correct(raw_text)

LanguageToolCorrector keeps the Java server alive; call corrector.close() on
shutdown if you cached it.
"""

from typing import Protocol, runtime_checkable

from transcribe_app.settings import Settings


@runtime_checkable
class Corrector(Protocol):
    async def correct(self, text: str) -> str: ...


def get_corrector(backend: str, lang: str, settings: Settings) -> Corrector:
    """Return a Corrector for the given backend and language.

    Ollama correctors are lightweight (created fresh each call so they always
    read current settings).  LanguageTool correctors are expensive to create —
    callers should cache them by (backend, lang) and call close() on shutdown.
    """
    if backend == "LanguageTool":
        from .languagetool import LanguageToolCorrector
        return LanguageToolCorrector(lang)
    if backend == "Spylls":
        from .spylls import SpyllsCorrector
        return SpyllsCorrector(lang)
    from .ollama import OllamaCorrector
    return OllamaCorrector(lang, settings)
