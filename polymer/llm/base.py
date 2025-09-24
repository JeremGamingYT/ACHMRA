from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str


class LLM(ABC):
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, timeout_s: int = 60) -> LLMResponse:
        ...


