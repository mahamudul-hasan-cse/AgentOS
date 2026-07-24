import asyncio

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Groq,
)
from groq import RateLimitError as GroqRateLimitError

from .base import DriverConnectionError, DriverError, LLMDriver, RateLimitError


class GroqDriver(LLMDriver):
    name = "groq"

    def __init__(self):
        super().__init__()
        self.api_key: str = self.config.get("api_key", "")
        self.model: str = self.config.get("model", "llama-3.1-8b-instant")
        base_url = self.config.get("base_url")
        self._client = (
            Groq(api_key=self.api_key, base_url=base_url) if self.api_key else None
        )

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self._client:
            raise DriverConnectionError("Groq API key not configured")
        try:
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=kwargs.get("model", self.model),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except GroqRateLimitError as e:
            raise RateLimitError(str(e)) from e
        except (APIConnectionError, APITimeoutError) as e:
            raise DriverConnectionError(str(e)) from e
        except APIStatusError as e:
            raise DriverError(str(e)) from e
