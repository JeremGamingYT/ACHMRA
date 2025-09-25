from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    train_file: str = Field(..., description="Path to the training jsonl file")
    val_file: str = Field(..., description="Path to the validation jsonl file")
    test_file: str | None = Field(None, description="Optional held-out jsonl file")
    preference_file: str | None = Field(None, description="Optional preference pairs jsonl file")
    rl_file: str | None = Field(None, description="Optional RL reward dataset jsonl file")
    tokenizer_dir: str | None = Field(
        None,
        description="Optional path to a tokenizer to reuse; defaults to the base model tokenizer",
    )


class LoRAConfig(BaseModel):
    enabled: bool = True
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=list)


class LatentPassConfig(BaseModel):
    control_tokens: list[str] = Field(
        default_factory=lambda: ["<pass_0>", "<pass_1>", "<verify>", "<compress>"]
    )
    embedding_dim: int = 64
    loss_weight: float = 0.4
    encourage_silence_weight: float = 0.1


class InternalLanguageConfig(BaseModel):
    begin_token: str = "<il_begin>"
    end_token: str = "<il_end>"
    bridge_token: str = "<il_bridge>"
    loss_weight: float = 0.25


class ThoughtGraphConfig(BaseModel):
    node_token: str = "<got_node>"
    edge_token: str = "<got_edge>"
    section_token: str = "<got_section>"
    summary_token: str = "[THOUGHT]"
    loss_weight: float = 0.35


class CalibrationConfig(BaseModel):
    confidence_token: str = "[CONF]"
    loss_type: Literal["mse", "brier"] = "brier"
    loss_weight: float = 1.0
    temperature_scaling: bool = True


class AntiCoTConfig(BaseModel):
    loss_weight: float = 0.3
    allowed_tokens: list[str] = Field(default_factory=lambda: ["[FINAL]", "[CONF]", "[NEXT]", "[THOUGHT]"])


class CurriculumBucket(BaseModel):
    name: str
    min_passes: int
    max_passes: int
    weight: float = 1.0

    @validator("max_passes")
    def validate_passes(cls, v, values):  # type: ignore[arg-type]
        if "min_passes" in values and v < values["min_passes"]:
            raise ValueError("max_passes must be >= min_passes")
        return v


class PhaseConfig(BaseModel):
    steps: int | None = None
    epochs: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 64
    gradient_accumulation: int = 1
    warmup_ratio: float = 0.03


class RLConfig(PhaseConfig):
    reward_beta: float = 0.1
    abstain_bonus: float = 0.2
    incorrect_penalty: float = 1.0
    length_penalty: float = 0.05


class ExportConfig(BaseModel):
    output_dir: str = "./artifacts/achmra_base_solo"
    quantizations: list[str] = Field(default_factory=lambda: ["Q6_K", "Q4_K_M"])
    gguf_name: str = "ACHMRA-Base-Solo"


class AchmraTrainingConfig(BaseModel):
    base_model: str = Field(..., description="HF repo id of the base checkpoint (e.g., meta-llama/Llama-3-8B)")
    data: DataConfig
    sft: PhaseConfig = Field(default_factory=PhaseConfig)
    preference: PhaseConfig = Field(default_factory=PhaseConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    latent_passes: LatentPassConfig = Field(default_factory=LatentPassConfig)
    internal_language: InternalLanguageConfig = Field(default_factory=InternalLanguageConfig)
    thought_graph: ThoughtGraphConfig = Field(default_factory=ThoughtGraphConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    anti_cot: AntiCoTConfig = Field(default_factory=AntiCoTConfig)
    curriculum: list[CurriculumBucket] = Field(
        default_factory=lambda: [
            CurriculumBucket(name="no_pass", min_passes=0, max_passes=0, weight=0.2),
            CurriculumBucket(name="single_pass", min_passes=1, max_passes=1, weight=0.4),
            CurriculumBucket(name="double_pass", min_passes=2, max_passes=2, weight=0.4),
        ]
    )
    max_sequence_length: int = 4096
    trust_remote_code: bool = False
    revision: str | None = None
    gradient_checkpointing: bool = True
    load_in_4bit: bool = True
    bf16: bool = True
    debug_keyword: str = "ACHMRA_DEBUG_TRACE"
    tokenizer_extra_cards: list[str] = Field(
        default_factory=lambda: [
            "[FINAL]",
            "[THOUGHT]",
            "[CONF]",
            "[NEXT]",
            "[IDK]",
            "<pass_0>",
            "<pass_1>",
            "<verify>",
            "<compress>",
            "<mem_anchor>",
            "</mem_anchor>",
            "<got_node>",
            "<got_edge>",
            "<got_section>",
            "<il_begin>",
            "<il_end>",
            "<il_bridge>",
        ]
    )
    export: ExportConfig = Field(default_factory=ExportConfig)

    @classmethod
    def load(cls, path: str | Path) -> "AchmraTrainingConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


__all__ = ["AchmraTrainingConfig"]

