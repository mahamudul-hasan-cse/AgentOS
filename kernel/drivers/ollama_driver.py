import asyncio

import requests

from .base import DriverConnectionError, DriverError, LLMDriver, RateLimitError


class OllamaDriver(LLMDriver):
    name = "ollama"

    def __init__(self):
        super().__init__()
        self.host: str = self.config.get("host", "http://localhost:11434")
        self.model: str = self.config.get("model", "llama3")

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.ok
        except requests.exceptions.RequestException:
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"{self.host}/api/generate",
                json={
                    "model": kwargs.get("model", self.model),
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=kwargs.get("timeout", 60),
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise DriverConnectionError(str(e)) from e

        if response.status_code == 429:
            raise RateLimitError(response.text)
        if not response.ok:
            raise DriverError(f"Ollama API error {response.status_code}: {response.text}")

        return response.json()["response"]
