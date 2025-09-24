from __future__ import annotations

import re
from typing import List

from ..ir import IntentFrame, Constraint, Preference, IntentionGraph
from ..llm import LLM


class SemanticParser:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm

    def parse(self, text: str) -> IntentionGraph:
        # Heuristic extraction: intent is first verb phrase; extract constraints and preferences by keywords
        intent = self._extract_intent(text)
        constraints = self._extract_constraints(text)
        preferences = self._extract_preferences(text)

        frame = IntentFrame(intent=intent or "inform", roles={}, constraints=constraints, preferences=preferences)
        graph = IntentionGraph(frames=[frame], alternatives=[], context={"source": "heuristic"})

        # Caller may optionally refine with LLM via refine_with_llm

        return graph

    def _extract_intent(self, text: str) -> str:
        m = re.search(r"\b(ask|explain|create|summarize|plan|design|code|ingest|analyze)\b", text.lower())
        return (m.group(1) if m else "inform").strip()

    def _extract_constraints(self, text: str) -> List[Constraint]:
        constraints: List[Constraint] = []
        if "8go" in text.lower() or "8 gb" in text.lower():
            constraints.append(Constraint(name="vram", value="8GB", confidence=0.9))
        if "10 go" in text.lower() or "10 gb" in text.lower():
            constraints.append(Constraint(name="ram", value="10GB", confidence=0.9))
        return constraints

    def _extract_preferences(self, text: str) -> List[Preference]:
        preferences: List[Preference] = []
        if "fran" in text.lower():  # français
            preferences.append(Preference(name="language", value="fr", weight=0.7))
        return preferences

    def _extract_json(self, text: str) -> str | None:
        import re

        m = re.search(r"\{[\s\S]*\}", text)
        return m.group(0) if m else None

    # --- Metacognitive helpers ---
    def refine_with_llm(self, graph: IntentionGraph, text: str) -> IntentionGraph:
        if not self.llm:
            return graph
        try:
            prompt = (
                "Extract roles, constraints, and preferences from the user text. "
                "Return compact JSON: {\"roles\":{}, \"constraints\":[{name,value}], \"preferences\":[{name,value,weight}]}.\n"
                f"Text: {text}"
            )
            resp = self.llm.complete(prompt, max_tokens=192, temperature=0.0)
            import json

            j = json.loads(self._extract_json(resp.text) or "{}")
            frame = graph.frames[0]
            frame.roles = j.get("roles", {})
            frame.constraints = [Constraint(**c) for c in j.get("constraints", []) if isinstance(c, dict)]
            frame.preferences = [Preference(**p) for p in j.get("preferences", []) if isinstance(p, dict)]
            graph.context["source"] = "heuristic+llm"
        except Exception:
            pass
        return graph

    def estimate_uncertainty(self, graph: IntentionGraph, text: str, retrieval_scores: List[float] | None = None) -> float:
        # Simple heuristic: lower intent confidence, short/ambiguous prompts, poor retrieval ⇒ higher uncertainty
        frame = graph.frames[0] if graph.frames else None
        base = 1.0 - (frame.confidence if frame else 0.5)
        length_penalty = 0.1 if len(text) < 20 else 0.0
        retrieval_penalty = 0.0
        if retrieval_scores:
            top = max(retrieval_scores) if retrieval_scores else 0.0
            retrieval_penalty = 0.5 * (0.3 - min(top, 0.3)) / 0.3  # if top<0.3 add up to 0.5
        score = max(0.0, min(1.0, base + length_penalty + retrieval_penalty))
        return float(score)


