from __future__ import annotations

from typing import List, Tuple
from ..ir import PlanGraph, PlanStep, IntentionGraph


class HTNPlanner:
    def __init__(self, max_depth: int = 4, max_branches: int = 3) -> None:
        self.max_depth = max_depth
        self.max_branches = max_branches

    def plan(self, goal: str, intention: IntentionGraph | None = None) -> PlanGraph:
        steps: dict[str, PlanStep] = {}
        root_children: List[str] = []

        # Always start with parse to ensure up-to-date intention
        steps["parse"] = PlanStep(id="parse", action="parse_intent")
        root_children.append("parse")

        # Decompose based on intent families
        intent = (intention.frames[0].intent if intention and intention.frames else "inform").lower()
        if intent in {"ingest", "index", "memorize"}:
            steps["chunk"] = PlanStep(id="chunk", action="chunk_input")
            steps["embed"] = PlanStep(id="embed", action="embed_chunks")
            steps["persist"] = PlanStep(id="persist", action="persist_vectors")
            root_children += ["chunk", "embed", "persist"]
        elif intent in {"analyze", "explain", "summarize", "plan", "design", "code"}:
            steps["retrieve"] = PlanStep(id="retrieve", action="retrieve_memory")
            steps["select_tools"] = PlanStep(id="select_tools", action="select_reasoners")
            steps["reason"] = PlanStep(id="reason", action="reason_and_verify")
            steps["answer"] = PlanStep(id="answer", action="generate_answer")
            root_children += ["retrieve", "select_tools", "reason", "answer"]
        else:
            # Fallback generic pipeline
            steps["retrieve"] = PlanStep(id="retrieve", action="retrieve_memory")
            steps["reason"] = PlanStep(id="reason", action="reason_and_verify")
            steps["answer"] = PlanStep(id="answer", action="generate_answer")
            root_children += ["retrieve", "reason", "answer"]

        # Add verification children to answer when needed
        steps["verify"] = PlanStep(id="verify", action="post_verify")
        root_children.append("verify")

        root = PlanStep(id="root", action="solve_goal", inputs={"goal": goal}, children=root_children)
        steps[root.id] = root
        return PlanGraph(steps=steps, root_id="root", uncertainty={})


