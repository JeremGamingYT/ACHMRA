from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLM, LLMResponse


class OllamaLLM(LLM):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, timeout_s: int = 60) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return LLMResponse(text=data.get("response", ""), model=self.model, provider="ollama")


