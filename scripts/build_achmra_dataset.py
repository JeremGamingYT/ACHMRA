from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polymer.training.achmra.config import AchmraTrainingConfig
from polymer.training.achmra.data import (
    AchmraRecord,
    ConversationTurn,
    InternalLanguageSegment,
    LatentPassTarget,
    ThoughtEdge,
    ThoughtGraph,
    ThoughtNode,
)


def _template(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**params)
    if isinstance(value, list):
        return [_template(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _template(v, params) for k, v in value.items()}
    return value


def _hashed_vector(seed: str, dim: int) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    floats: list[float] = []
    for i in range(dim):
        byte = digest[i % len(digest)]
        value = (byte / 255.0) * 2.0 - 1.0
        floats.append(round(value, 4))
    return floats


def _build_latent_passes(raw_passes: list[dict[str, Any]], params: dict[str, Any], dim: int) -> list[LatentPassTarget]:
    passes: list[LatentPassTarget] = []
    for entry in raw_passes:
        rendered = _template(entry, params)
        vector = rendered.get("vector")
        if not vector:
            seed = f"{rendered['token']}::{rendered.get('description', '')}::{params['variant_key']}"
            vector = _hashed_vector(seed, dim)
        passes.append(
            LatentPassTarget(
                token=rendered["token"],
                vector=vector,
                weight=float(rendered.get("weight", 1.0)),
                description=rendered.get("description"),
            )
        )
    return passes


def _build_internal_language(raw_segments: list[dict[str, Any]], params: dict[str, Any]) -> list[InternalLanguageSegment]:
    segments: list[InternalLanguageSegment] = []
    for seg in raw_segments:
        rendered = _template(seg, params)
        segments.append(
            InternalLanguageSegment(
                channel=rendered.get("channel", "core"),
                content=rendered["content"],
            )
        )
    return segments


def _build_thought_graph(raw_graph: dict[str, Any] | None, params: dict[str, Any]) -> ThoughtGraph | None:
    if not raw_graph:
        return None
    graph_data = _template(raw_graph, params)
    nodes = [
        ThoughtNode(
            id=node["id"],
            label=node["label"],
            statement=node["statement"],
            confidence=float(node["confidence"]) if node.get("confidence") is not None else None,
            parents=node.get("parents", []),
            children=node.get("children", []),
        )
        for node in graph_data.get("nodes", [])
    ]
    edges = [
        ThoughtEdge(source=edge["source"], target=edge["target"], relation=edge.get("relation", "supports"))
        for edge in graph_data.get("edges", [])
    ]
    return ThoughtGraph(root=graph_data.get("root"), nodes=nodes, edges=edges)


def _build_conversation(raw_turns: list[dict[str, str]], params: dict[str, Any]) -> list[ConversationTurn]:
    return [ConversationTurn(role=turn["role"], content=_template(turn["content"], params)) for turn in raw_turns]


def _derive_checks(graph: ThoughtGraph | None, confidence: float) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append({"name": "confidence_range", "passed": 0.0 <= confidence <= 1.0, "weight": 0.4})
    if graph and graph.nodes:
        branching = max(len(node.children) for node in graph.nodes)
        checks.append({"name": "graph_branching", "passed": branching >= 1, "weight": 0.35})
        checks.append({"name": "graph_consistency", "passed": len(graph.edges) >= len(graph.nodes) - 1, "weight": 0.25})
    else:
        checks.append({"name": "graph_present", "passed": False, "weight": 0.6})
    return checks


def _build_record(
    scenario_id: str,
    base: dict[str, Any],
    params: dict[str, Any],
    config: AchmraTrainingConfig,
) -> AchmraRecord:
    conv = _build_conversation(base.get("conversation", []), params)
    constraints = _template(base.get("constraints", []), params)
    latent_passes = _build_latent_passes(base.get("latent_passes", []), params, config.latent_passes.embedding_dim)
    internal_language = _build_internal_language(base.get("internal_language", []), params)
    thought_graph = _build_thought_graph(base.get("thought_graph"), params)
    thought_summary = _template(base.get("thought_summary"), params)
    final = _template(base.get("final"), params)
    next_text = _template(base.get("next"), params)
    record = AchmraRecord(
        id=f"{scenario_id}::{params['variant_key']}",
        conversation=conv,
        compressed_memory=_template(base.get("compressed_memory"), params),
        constraints=constraints,
        latent_passes=latent_passes,
        checks=_derive_checks(thought_graph, float(params["confidence"])),
        internal_language=internal_language,
        thought_graph=thought_graph,
        thought_summary=thought_summary,
        final=final,
        confidence=float(params["confidence"]),
        next=next_text,
        tags=params.get("tags", base.get("tags", [])),
        difficulty=params.get("difficulty", None) or base.get("difficulty"),
        metadata={
            "scenario_id": scenario_id,
            "variant_key": params["variant_key"],
        },
    )
    return record


def _normalise_params(raw_values: dict[str, Any], global_tags: list[str], difficulty: str | None) -> dict[str, Any]:
    values = dict(raw_values)
    values.setdefault("split", "train")
    values.setdefault("tags", global_tags)
    values.setdefault("difficulty", difficulty)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ACHMRA dataset with Graph-of-Thought supervision")
    parser.add_argument("--scenarios", default="datasets/achmra/scenarios.yaml", help="Path to the scenario YAML file")
    parser.add_argument("--config", default="/kaggle/working/config/training/achmra-base-solo.yaml", help="Training config (for embedding dims)")
    parser.add_argument("--outdir", default="/kaggle/working/data/achmra", help="Output directory for jsonl splits")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = AchmraTrainingConfig.load(args.config)
    scenarios = yaml.safe_load(Path(args.scenarios).read_text(encoding="utf-8"))

    splits: dict[str, list[AchmraRecord]] = {"train": [], "val": [], "test": []}

    for scenario in scenarios:
        base = scenario.get("base", {})
        parameters = scenario.get("parameters", [])
        difficulty = scenario.get("difficulty")
        tags = scenario.get("tags", [])
        for entry in parameters:
            params = _normalise_params(entry.get("values", {}), tags, difficulty)
            params["variant_key"] = entry.get("key", "variant")
            record = _build_record(scenario["id"], base, params, cfg)
            target_split = params.get("split", "train")
            if target_split not in splits:
                raise ValueError(f"Unknown split '{target_split}' for scenario {scenario['id']}")
            splits[target_split].append(record)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}
    for split_name, records in splits.items():
        if not records:
            continue
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as fout:
            for rec in records:
                fout.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")
        stats[split_name] = len(records)

    if stats:
        summary_path = out_dir / "manifest.json"
        summary_path.write_text(json.dumps({"counts": stats, "scenarios": len(scenarios)}, indent=2), encoding="utf-8")

    print(json.dumps({"written": stats, "outdir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
