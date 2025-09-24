from __future__ import annotations

from .config import load_config, AppConfig
from .orchestrator import Orchestrator


class Agent:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.orchestrator = Orchestrator(self.config)

    def ingest(self, text: str, metadata: dict | None = None) -> int:
        return self.orchestrator.ingest(text, metadata)

    def query(self, question: str) -> dict:
        return self.orchestrator.query(question)


