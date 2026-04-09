import asyncio
import re

from transcribe_app.config import HUNSPELL_CODES, HUNSPELL_SEARCH_DIRS


class SpyllsCorrector:
    def __init__(self, lang: str) -> None:
        self._code = HUNSPELL_CODES[lang]

    async def correct(self, text: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._correct_sync, text)

    def _correct_sync(self, text: str) -> str:
        try:
            from spylls.hunspell import Dictionary  # type: ignore
        except ImportError:
            raise RuntimeError(
                "Spylls nicht installiert.\n"
                "Bitte ausführen: pip install spylls"
            )
        dic = self._find_dict()
        if dic is None:
            raise RuntimeError(
                f"Hunspell-Wörterbuch '{self._code}' nicht gefunden.\n"
                f"Bitte installieren: sudo apt install hunspell-{self._code[:2].lower()}"
            )
        d = Dictionary.from_files(str(dic))

        def _fix(m: re.Match) -> str:
            word = m.group(0)
            if d.lookup(word):
                return word
            suggestions = list(d.suggest(word))
            return suggestions[0] if suggestions else word

        return re.sub(r"[^\W\d_]+", _fix, text)

    def _find_dict(self):
        for base in HUNSPELL_SEARCH_DIRS:
            if (base / f"{self._code}.dic").exists():
                return base / self._code
        return None
