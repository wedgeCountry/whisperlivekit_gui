import asyncio

from transcribe_app.config import LT_LANG_CODES


class LanguageToolCorrector:
    """Wraps language_tool_python. The Java server is started lazily on first use
    and kept alive for the lifetime of this instance."""

    def __init__(self, lang: str) -> None:
        self._lang_code = LT_LANG_CODES[lang]
        self._tool      = None  # lazy init

    async def correct(self, text: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._correct_sync, text)

    def _correct_sync(self, text: str) -> str:
        try:
            import language_tool_python  # type: ignore
        except ImportError:
            raise RuntimeError(
                "LanguageTool nicht installiert.\n"
                "Bitte ausführen: pip install language_tool_python"
            )
        if self._tool is None:
            self._tool = language_tool_python.LanguageTool(self._lang_code)
        matches = self._tool.check(text)
        return language_tool_python.utils.correct(text, matches)

    def close(self) -> None:
        if self._tool is not None:
            try:
                self._tool.close()
            except Exception:
                pass
            self._tool = None
