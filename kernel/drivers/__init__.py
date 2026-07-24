from .base import DriverConnectionError, DriverError, LLMDriver, RateLimitError
from .deepseek_driver import DeepSeekDriver
from .gemini_driver import GeminiDriver
from .groq_driver import GroqDriver
from .ollama_driver import OllamaDriver

DRIVER_REGISTRY: dict[str, type[LLMDriver]] = {
    "groq": GroqDriver,
    "deepseek": DeepSeekDriver,
    "gemini": GeminiDriver,
    "ollama": OllamaDriver,
}

__all__ = [
    "LLMDriver",
    "DriverError",
    "RateLimitError",
    "DriverConnectionError",
    "GroqDriver",
    "DeepSeekDriver",
    "GeminiDriver",
    "OllamaDriver",
    "DRIVER_REGISTRY",
]
