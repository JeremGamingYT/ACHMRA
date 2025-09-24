from __future__ import annotations

from typing import List
from ..ir import PlanGraph, PlanStep


class HTNPlanner:
    def __init__(self, max_depth: int = 4, max_branches: int = 3) -> None:
        self.max_depth = max_depth
        self.max_branches = max_branches

    def plan(self, goal: str) -> PlanGraph:
        # Simple template HTN: root → parse → retrieve → reason → answer
        steps = {}
        root = PlanStep(id="root", action="solve_goal", inputs={"goal": goal}, children=["parse", "retrieve", "reason", "answer"])
        steps[root.id] = root
        steps["parse"] = PlanStep(id="parse", action="parse_intent")
        steps["retrieve"] = PlanStep(id="retrieve", action="retrieve_memory")
        steps["reason"] = PlanStep(id="reason", action="reason_and_verify")
        steps["answer"] = PlanStep(id="answer", action="generate_answer")
        return PlanGraph(steps=steps, root_id="root", uncertainty={})


