from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import Dataset
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import AchmraTrainingConfig


class ConversationTurn(BaseModel):
    role: str
    content: str


class LatentPassTarget(BaseModel):
    token: str
    vector: list[float] = Field(default_factory=list)
    weight: float = 1.0
    description: str | None = None


class InternalLanguageSegment(BaseModel):
    channel: str = "core"
    content: str


class ThoughtNode(BaseModel):
    id: str
    label: str
    statement: str
    confidence: float | None = None
    parents: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)


class ThoughtEdge(BaseModel):
    source: str
    target: str
    relation: str = "supports"


class ThoughtGraph(BaseModel):
    root: str | None = None
    nodes: list[ThoughtNode] = Field(default_factory=list)
    edges: list[ThoughtEdge] = Field(default_factory=list)


class CheckSignal(BaseModel):
    name: str
    passed: bool
    weight: float = 1.0


class AchmraRecord(BaseModel):
    id: str
    conversation: list[ConversationTurn]
    compressed_memory: str | None = None
    constraints: list[str] = Field(default_factory=list)
    latent_passes: list[LatentPassTarget] = Field(default_factory=list)
    checks: list[CheckSignal] = Field(default_factory=list)
    internal_language: list[InternalLanguageSegment] = Field(default_factory=list)
    thought_graph: ThoughtGraph | None = None
    thought_summary: str | None = None
    final: str
    confidence: float
    next: str | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_passes(self) -> int:
        return len(self.latent_passes)


class AchmraDataset:
    def __init__(self, records: Sequence[AchmraRecord]):
        self.records = list(records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AchmraDataset":
        records: list[AchmraRecord] = []
        with Path(path).open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                try:
                    records.append(AchmraRecord(**data))
                except ValidationError as err:
                    raise ValueError(f"Invalid dataset line: {err}") from err
        return cls(records)

    def to_hf_dataset(self) -> Dataset:
        return Dataset.from_list([rec.model_dump() for rec in self.records])


def _ensure_tag_prefix(text: str, tag: str) -> str:
    text = text.strip()
    return text if text.startswith(tag) else f"{tag} {text}"


def _format_final_lines(record: AchmraRecord) -> str:
    lines: list[str] = []
    if record.thought_summary:
        lines.append(_ensure_tag_prefix(record.thought_summary, "[THOUGHT]"))
    final_line = _ensure_tag_prefix(record.final, "[FINAL]")
    lines.append(final_line)
    conf = max(0.0, min(1.0, record.confidence))
    lines.append(f"[CONF] {conf:.2f}")
    if record.next:
        lines.append(_ensure_tag_prefix(record.next, "[NEXT]"))
    return "\n".join(lines)


def _render_conversation(record: AchmraRecord) -> str:
    chunks: list[str] = []
    for turn in record.conversation:
        role = turn.role.upper()
        chunks.append(f"[{role}] {turn.content.strip()}")
    if record.constraints:
        joined = " ; ".join(constraint.strip() for constraint in record.constraints)
        chunks.append(f"[CONSTRAINTS] {joined}")
    if record.compressed_memory:
        chunks.append(f"<mem_anchor>{record.compressed_memory.strip()}</mem_anchor>")
    return "\n".join(chunks)


def _render_internal_language(record: AchmraRecord, config: AchmraTrainingConfig) -> str:
    if not record.internal_language:
        return ""
    segments = [config.internal_language.begin_token]
    for seg in record.internal_language:
        channel = seg.channel.upper()
        segments.append(f"{config.internal_language.bridge_token}{channel}:{seg.content.strip()}")
    segments.append(config.internal_language.end_token)
    return " ".join(segments)


def _render_thought_graph(record: AchmraRecord, config: AchmraTrainingConfig) -> str:
    graph = record.thought_graph
    if not graph or (not graph.nodes and not graph.edges):
        return ""
    lines = [config.thought_graph.section_token]
    for node in graph.nodes:
        children = ",".join(node.children) if node.children else "-"
        parents = ",".join(node.parents) if node.parents else "-"
        conf = "na" if node.confidence is None else f"{node.confidence:.2f}"
        lines.append(
            f"{config.thought_graph.node_token} {node.id}|{node.label}|{node.statement.strip()}|conf={conf}|parents={parents}|children={children}"
        )
    for edge in graph.edges:
        lines.append(f"{config.thought_graph.edge_token} {edge.source}->{edge.target}|{edge.relation}")
    if graph.root:
        lines.append(f"{config.thought_graph.section_token}_ROOT {graph.root}")
    return "\n".join(lines)


def _attach_control_tokens(record: AchmraRecord, config: AchmraTrainingConfig) -> str:
    if not record.latent_passes:
        return "<pass_0><verify>"
    tokens = []
    for latent in record.latent_passes:
        if latent.token not in config.latent_passes.control_tokens:
            raise ValueError(f"Unknown latent control token {latent.token} in example {record.id}")
        tokens.append(latent.token)
    if "<verify>" not in tokens:
        tokens.append("<verify>")
    return "".join(tokens)


def render_training_sample(record: AchmraRecord, tokenizer: PreTrainedTokenizerBase, config: AchmraTrainingConfig) -> dict[str, Any]:
    prompt_header = (
        "Tu es ACHMRA-Base-Solo. Tu raisonnes en passes internes declenchees par des jetons de controle"
        " et tu ne devoiles jamais ces passes. Maintiens un graphe de pensees interne et fournis un resume humain"
        " concis avant la reponse finale. Format de sortie obligatoire: [THOUGHT] resume, [FINAL], [CONF] (puis [NEXT] si besoin)."
    )
    convo = _render_conversation(record)
    graph_text = _render_thought_graph(record, config)
    internal_text = _render_internal_language(record, config)
    latent_tokens = _attach_control_tokens(record, config)
    target_text = _format_final_lines(record)
    composed_parts = [f"[SYS] {prompt_header}", convo]
    if graph_text:
        composed_parts.append(graph_text)
    if internal_text:
        composed_parts.append(internal_text)
    composed_parts.append(latent_tokens)
    composed = "\n".join(part for part in composed_parts if part)
    tokenized = tokenizer(
        composed + "\n" + target_text,
        truncation=True,
        max_length=config.max_sequence_length,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "target_text": target_text,
        "latent_passes": [lp.model_dump() for lp in record.latent_passes],
        "internal_language": [seg.model_dump() for seg in record.internal_language],
        "thought_graph": record.thought_graph.model_dump() if record.thought_graph else None,
        "thought_summary": record.thought_summary,
        "confidence": record.confidence,
        "id": record.id,
        "num_passes": record.num_passes,
        "tags": record.tags,
        "difficulty": record.difficulty or "unknown",
    }


@dataclass
class AchmraCollator:
    tokenizer: PreTrainedTokenizerBase
    config: AchmraTrainingConfig

    def __post_init__(self) -> None:
        vocab = self.tokenizer.get_vocab()
        self.control_ids = [vocab.get(tok, None) for tok in self.config.latent_passes.control_tokens]
        if any(idx is None for idx in self.control_ids):
            missing = [tok for tok, idx in zip(self.config.latent_passes.control_tokens, self.control_ids) if idx is None]
            raise ValueError(f"Tokenizer missing control tokens: {missing}")
        self.allowed_ids = {vocab.get(tok) for tok in self.config.anti_cot.allowed_tokens if tok in vocab}
        self.hidden_token_ids = {
            token: vocab.get(token)
            for token in [
                self.config.internal_language.begin_token,
                self.config.internal_language.end_token,
                self.config.internal_language.bridge_token,
                self.config.thought_graph.node_token,
                self.config.thought_graph.edge_token,
                self.config.thought_graph.section_token,
                f"{self.config.thought_graph.section_token}_ROOT",
            ]
            if vocab.get(token) is not None
        }

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids: list[list[int]] = []
        attn: list[list[int]] = []
        labels: list[list[int]] = []
        anti_masks: list[list[float]] = []
        conf_targets: list[float] = []
        latent_vectors: list[list[list[float]]] = []
        latent_positions: list[list[int]] = []
        graph_payload: list[Any] = []
        for item in batch:
            ids = list(item["input_ids"])
            mask = list(item["attention_mask"])
            pad_amount = max_len - len(ids)
            ids.extend([pad_id] * pad_amount)
            mask.extend([0] * pad_amount)
            label = ids.copy()
            final_token_id = self.tokenizer.convert_tokens_to_ids("[FINAL]")
            try:
                final_idx = ids.index(final_token_id)
            except ValueError:
                final_idx = len(ids)
            for idx in range(final_idx):
                label[idx] = -100
            for ctrl_tok in self.config.latent_passes.control_tokens:
                ctrl_id = self.tokenizer.convert_tokens_to_ids(ctrl_tok)
                for idx, tok_id in enumerate(ids):
                    if tok_id == ctrl_id:
                        label[idx] = -100
            for hidden_tok, hidden_id in self.hidden_token_ids.items():
                if hidden_id is None:
                    continue
                for idx, tok_id in enumerate(ids):
                    if tok_id == hidden_id:
                        label[idx] = -100
            anti_mask_row = [0.0] * len(ids)
            for idx, tok_id in enumerate(ids):
                if label[idx] == -100:
                    continue
                if self.allowed_ids and tok_id not in self.allowed_ids:
                    anti_mask_row[idx] = 1.0
            input_ids.append(ids)
            attn.append(mask)
            labels.append(label)
            anti_masks.append(anti_mask_row)
            conf_targets.append(float(item["confidence"]))
            latent_vecs: list[list[float]] = []
            latent_pos: list[int] = []
            for latent in item["latent_passes"]:
                token_id = self.tokenizer.convert_tokens_to_ids(latent["token"])
                try:
                    pos = ids.index(token_id)
                except ValueError:
                    pos = -1
                latent_pos.append(pos)
                latent_vecs.append([float(x) for x in latent.get("vector", [])])
            latent_positions.append(latent_pos)
            latent_vectors.append(latent_vecs)
            graph_payload.append(item.get("thought_graph"))
        max_latents = max((len(v) for v in latent_vectors), default=0)
        dim = self.config.latent_passes.embedding_dim
        latent_target_tensor = torch.zeros(len(batch), max_latents, dim, dtype=torch.float)
        latent_mask_tensor = torch.zeros(len(batch), max_latents, dtype=torch.float)
        latent_pos_tensor = torch.full((len(batch), max_latents), -1, dtype=torch.long)
        for row, (vecs, positions) in enumerate(zip(latent_vectors, latent_positions)):
            for col, (vec, pos) in enumerate(zip(vecs, positions)):
                latent_mask_tensor[row, col] = 1.0
                latent_pos_tensor[row, col] = pos
                if vec:
                    tensor = torch.tensor(vec[:dim], dtype=torch.float)
                    latent_target_tensor[row, col, : tensor.numel()] = tensor
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "anti_cot_mask": torch.tensor(anti_masks, dtype=torch.float),
            "confidence_target": torch.tensor(conf_targets, dtype=torch.float),
            "latent_targets": latent_target_tensor,
            "latent_mask": latent_mask_tensor,
            "latent_positions": latent_pos_tensor,
            "thought_graph": graph_payload,
        }


def add_special_tokens(tokenizer: PreTrainedTokenizerBase, config: AchmraTrainingConfig) -> int:
    new_tokens = [tok for tok in config.tokenizer_extra_cards if tok not in tokenizer.get_vocab()]
    if not new_tokens:
        return 0
    added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return added


def load_tokenizer(name_or_path: str, config: AchmraTrainingConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        config.data.tokenizer_dir or name_or_path,
        use_fast=True,
        trust_remote_code=config.trust_remote_code,
    )
    add_special_tokens(tokenizer, config)
    return tokenizer


__all__ = [
    "AchmraDataset",
    "AchmraRecord",
    "render_training_sample",
    "AchmraCollator",
    "load_tokenizer",
]

