from __future__ import annotations

from .base import LLM, LLMResponse


class MockLLM(LLM):
    def __init__(self, model: str = "mock-1") -> None:
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, timeout_s: int = 60) -> LLMResponse:
        text = "[MOCK RESPONSE] " + prompt[: max(0, 200)]
        return LLMResponse(text=text, model=self.model, provider="mock")


