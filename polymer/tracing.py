from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Trace:
    question: str
    intention: Dict[str, Any]
    plan: Dict[str, Any]
    context: List[Any]
    answer: str
    verification: List[str]
    meta: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class TraceLogger:
    def __init__(self, enabled: bool, directory: str) -> None:
        self.enabled = enabled
        self.dir = Path(directory)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: Trace) -> Path | None:
        if not self.enabled:
            return None
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        path = self.dir / f"trace_{ts}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(trace.to_json() + "\n")
        return path


