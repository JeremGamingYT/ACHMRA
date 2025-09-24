from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .config import AppConfig
from .llm import MockLLM, OllamaLLM, LLM
from .parser.semantic_parser import SemanticParser
from .memory import VectorStore, EpisodicMemory, KnowledgeGraph
from .reasoners import ConstraintChecker, ProgramSandbox
from .planner.htn import HTNPlanner
from .verification.verify import quick_checks


class Orchestrator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm = self._init_llm()
        data_dir = Path(config.memory.data_dir)
        self.vector = VectorStore(data_dir / config.memory.vector_db, config.memory.embed_model)
        self.episodes = EpisodicMemory(data_dir / config.memory.episodes_db)
        self.kg = KnowledgeGraph(data_dir / config.memory.kg_file)
        # Parser instantiated without LLM; we will decide dynamically when to refine
        self.parser = SemanticParser(None)
        self.checker = ConstraintChecker(blocked_topics=config.alignment.blocked_topics)
        self.sandbox = ProgramSandbox()
        self.planner = HTNPlanner(max_depth=config.planner.max_depth, max_branches=config.planner.max_branches)

    def _init_llm(self) -> LLM:
        if self.config.llm.provider == "ollama":
            return OllamaLLM(self.config.llm.model)
        return MockLLM()

    def ingest(self, text: str, metadata: dict | None = None) -> int:
        self.episodes.add(f"ingest:{text[:60]}")
        return self.vector.add_text(text, metadata)

    def query(self, question: str) -> dict:
        # Plan
        plan = self.planner.plan(question)
        # Parse (heuristic first, then optional LLM refinement)
        intention = self.parser.parse(question)
        if self._should_use_llm_for_parsing(question) and not isinstance(self.llm, MockLLM):
            intention = self._refine_intention(intention, question)
        issues = self.checker.check(intention)
        # Retrieve
        ctx = self.vector.search(question, top_k=5)
        context_text = "\n".join(t for _, _, t in ctx)
        scores = [s for _, s, _ in ctx]
        # Uncertainty estimate
        uncertainty = self.parser.estimate_uncertainty(intention, question, retrieval_scores=scores)
        decisions = {
            "used_llm_for_parsing": self._should_use_llm_for_parsing(question) and not isinstance(self.llm, MockLLM),
            "top_retrieval_score": max(scores) if scores else 0.0,
            "issues": issues,
        }
        needs_clarification = uncertainty > 0.6 or bool(issues)
        clarifying_question = None
        if needs_clarification:
            clarifying_question = self._clarify(question, intention, context_text)
        # Reason + Answer
        answer = ""
        if not needs_clarification:
            answer = self._answer(question, context_text)
        verif = quick_checks(answer)
        self.episodes.add(f"query:{question[:60]}")
        return {
            "intent": intention.model_dump(),
            "plan": plan.model_dump(),
            "context": ctx,
            "answer": answer,
            "verification": verif + issues,
            "meta": {"uncertainty": uncertainty, "decisions": decisions},
            "needs_clarification": needs_clarification,
            "clarifying_question": clarifying_question,
        }

    def _answer(self, question: str, context: str) -> str:
        if isinstance(self.llm, MockLLM):
            return f"[MOCK] Q: {question}\nCTX: {context[:400]}"
        prompt = (
            "You are a helpful, concise assistant. Use the context when relevant.\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer in French."
        )
        max_tok = min(256, self.config.alignment.max_tokens)
        resp = self.llm.complete(prompt, max_tokens=max_tok, temperature=0.2)
        return resp.text.strip()

    # --- Metacognitive policies ---
    def _should_use_llm_for_parsing(self, text: str) -> bool:
        return len(text) > 120 or ("et" in text and "," in text)

    def _refine_intention(self, intention, text: str):
        parser_with_llm = SemanticParser(self.llm)
        return parser_with_llm.refine_with_llm(intention, text)

    def _clarify(self, question: str, intention, context: str) -> str:
        # Prefer LLM to phrase a single clarifying question; otherwise fallback template
        if not isinstance(self.llm, MockLLM):
            prompt = (
                "Generate ONE concise clarifying question in French to disambiguate the user's intent.\n"
                f"User: {question}\nContext: {context[:400]}\nQuestion:"
            )
            try:
                resp = self.llm.complete(prompt, max_tokens=64, temperature=0.2)
                return resp.text.strip()
            except Exception:
                pass
        # Fallback
        return "Pouvez-vous préciser le résultat attendu et les contraintes clés ?"