import asyncio

import requests

from .base import DriverConnectionError, DriverError, LLMDriver, RateLimitError

API_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekDriver(LLMDriver):
    name = "deepseek"

    def __init__(self):
        super().__init__()
        self.api_key: str = self.config.get("api_key", "")
        self.model: str = self.config.get("model", "deepseek-chat")

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise DriverConnectionError("DeepSeek API key not configured")
        try:
            response = await asyncio.to_thread(
                requests.post,
                API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": kwargs.get("model", self.model),
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=kwargs.get("timeout", 30),
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise DriverConnectionError(str(e)) from e

        if response.status_code == 429:
            raise RateLimitError(response.text)
        if not response.ok:
            raise DriverError(f"DeepSeek API error {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]
