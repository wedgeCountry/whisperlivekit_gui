import httpx

from transcribe_app.config import OLLAMA_BASE_URL
from transcribe_app.settings import Settings
from ..text_processing import strip_ollama_junk


class OllamaCorrector:
    def __init__(self, lang: str, settings: Settings) -> None:
        self._lang     = lang
        self._settings = settings

    async def correct(self, text: str) -> str:
        payload = {
            "model": self._settings.ollama_models[self._lang],
            "messages": [
                {"role": "system", "content": self._settings.system_prompts[self._lang]},
                {"role": "user",   "content": text},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()["message"]["content"].strip()
            return strip_ollama_junk(result, self._settings.system_prompts[self._lang])
