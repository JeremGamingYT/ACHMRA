from __future__ import annotations

from typing import List

from ..ir import IntentionGraph, Constraint


class ConstraintChecker:
    def __init__(self, blocked_topics: List[str] | None = None) -> None:
        self.blocked_topics = set((blocked_topics or []))

    def check(self, graph: IntentionGraph) -> List[str]:
        issues: List[str] = []
        # Simple alignment: check text topics in preferences/constraints
        for frame in graph.frames:
            for c in frame.constraints:
                if c.value.lower() in self.blocked_topics:
                    issues.append(f"Blocked constraint: {c.name}={c.value}")
            for p in frame.preferences:
                if p.value.lower() in self.blocked_topics:
                    issues.append(f"Blocked preference: {p.name}={p.value}")
        return issues


