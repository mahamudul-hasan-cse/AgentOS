import asyncio

import google.generativeai as genai
from google.api_core.exceptions import (
    DeadlineExceeded,
    GoogleAPICallError,
    ResourceExhausted,
    ServiceUnavailable,
)

from .base import DriverConnectionError, DriverError, LLMDriver, RateLimitError


class GeminiDriver(LLMDriver):
    name = "gemini"

    def __init__(self):
        super().__init__()
        self.api_key: str = self.config.get("api_key", "")
        self.model: str = self.config.get("model", "gemini-1.5-flash")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise DriverConnectionError("Gemini API key not configured")
        try:
            model = genai.GenerativeModel(kwargs.get("model", self.model))
            response = await asyncio.to_thread(model.generate_content, prompt)
            return response.text
        except ResourceExhausted as e:
            raise RateLimitError(str(e)) from e
        except (ServiceUnavailable, DeadlineExceeded) as e:
            raise DriverConnectionError(str(e)) from e
        except GoogleAPICallError as e:
            raise DriverError(str(e)) from e
