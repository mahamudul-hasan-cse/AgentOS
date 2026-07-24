from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class DriverError(Exception):
    """Base exception for LLM driver failures."""


class RateLimitError(DriverError):
    """Raised when a provider's rate limit is hit."""


class DriverConnectionError(DriverError):
    """Raised when a provider can't be reached."""


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


class LLMDriver(ABC):
    """Hardware-abstraction-layer interface for LLM providers."""

    name: str = "base"

    def __init__(self):
        self.config: dict[str, Any] = load_config().get(self.name, {})

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...
