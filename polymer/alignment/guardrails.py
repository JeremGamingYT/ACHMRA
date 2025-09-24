from __future__ import annotations

from typing import List


class Guardrails:
    def __init__(self, blocked_topics: List[str] | None = None) -> None:
        self.blocked_topics = set((blocked_topics or []))

    def filter_text(self, text: str) -> str:
        redacted = text
        for topic in self.blocked_topics:
            redacted = redacted.replace(topic, "[BLOCKED]")
        return redacted


