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
from .tracing import TraceLogger, Trace


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
        self.tracer = TraceLogger(enabled=bool(config.tracing.get("enabled", True)), directory=str(config.tracing.get("dir", "./data/traces")))

    def _init_llm(self) -> LLM:
        if self.config.llm.provider == "ollama":
            return OllamaLLM(self.config.llm.model)
        return MockLLM()

    def ingest(self, text: str, metadata: dict | None = None) -> int:
        self.episodes.add(f"ingest:{text[:60]}")
        return self.vector.add_text(text, metadata)

    def query(self, question: str) -> dict:
        # Parse (heuristic first, then optional LLM refinement)
        intention = self.parser.parse(question)
        if self._should_use_llm_for_parsing(question) and not isinstance(self.llm, MockLLM):
            intention = self._refine_intention(intention, question)
        # Plan with intention
        plan = self.planner.plan(question, intention)
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
        result = {
            "intent": intention.model_dump(),
            "plan": plan.model_dump(),
            "context": ctx,
            "answer": answer,
            "answer_struct": self._parse_achmra_output(answer),
            "verification": verif + issues,
            "meta": {"uncertainty": uncertainty, "decisions": decisions},
            "needs_clarification": needs_clarification,
            "clarifying_question": clarifying_question,
        }
        # Trace logging
        try:
            self.tracer.save(
                Trace(
                    question=question,
                    intention=result["intent"],
                    plan=result["plan"],
                    context=result["context"],
                    answer=result["answer"],
                    verification=result["verification"],
                    meta=result["meta"],
                )
            )
        except Exception:
            pass
        self.episodes.add(f"query:{question[:60]}")
        return result

    def _answer(self, question: str, context: str) -> str:
        if isinstance(self.llm, MockLLM):
            return f"[MOCK] Q: {question}\nCTX: {context[:400]}"
        system_prompt = (
            "System: Tu es ACHMRA-Base-Solo. Effectue tes passes internes dans ta langue privee et maintiens un graphe de pensees multi-branche invisible.\n"
            "Avant de repondre, fournis un resume humain concis via [THOUGHT], puis reponds avec [FINAL] et [CONF] (ajoute [NEXT] si une information manque).\n"
            "Ne devoile jamais ta langue interne brute ni les tokens de controle.\n"
        )
        prompt = f"{system_prompt}\nContext:\n{context}\n\nQuestion: {question}\n"
        max_tok = min(256, self.config.alignment.max_tokens)
        resp = self.llm.complete(prompt, max_tokens=max_tok, temperature=0.2)
        return resp.text.strip()
    def _parse_achmra_output(self, text: str) -> dict:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        final = None
        thought = None
        confidence = None
        next_step = None
        for line in lines:
            upper = line.upper()
            if upper.startswith("[THOUGHT]") and thought is None:
                thought = line[9:].strip()
            elif upper.startswith("[FINAL]") and final is None:
                final = line[7:].strip()
            elif upper.startswith("[CONF]") and confidence is None:
                try:
                    confidence = float(line[6:].strip())
                except ValueError:
                    confidence = None
            elif upper.startswith("[NEXT]") and next_step is None:
                next_step = line[6:].strip()
        if final is None:
            final = text.strip()
        return {"final": final, "confidence": confidence, "next": next_step, "thought": thought}
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








