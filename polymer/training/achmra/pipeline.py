from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset
from transformers import TrainingArguments

from .config import AchmraTrainingConfig, PhaseConfig
from .data import AchmraCollator, AchmraDataset, render_training_sample, load_tokenizer
from .model import AchmraModelArtifacts, load_achmra_model
from .trainer import AchmraTrainer


@dataclass
class AchmraPipeline:
    config: AchmraTrainingConfig
    tokenizer: Any
    datasets: dict[str, Dataset]
    artifacts: AchmraModelArtifacts
    collator: AchmraCollator

    def train_sft(self) -> None:
        trainer = self._make_trainer("sft", self.config.sft)
        trainer.train()
        trainer.save_state()
        trainer.save_model()

    def train_preference(self) -> None:
        if "preference" not in self.datasets:
            raise ValueError("Preference dataset not loaded; set data.preference_file in the config")
        trainer = self._make_trainer(
            "preference",
            self.config.preference,
            train_split="preference",
        )
        trainer.train()
        trainer.save_state()
        trainer.save_model()

    def train_rl(self) -> None:
        if "rl" not in self.datasets:
            raise ValueError("RL dataset not loaded; set data.rl_file in the config")
        trainer = self._make_trainer(
            "rl",
            self.config.rl,
            train_split="rl",
        )
        trainer.train()
        trainer.save_state()
        trainer.save_model()

    def export_gguf(self, checkpoint_dir: str, output_dir: Optional[str] = None) -> list[Path]:
        base_path = Path(checkpoint_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Checkpoint directory {base_path} does not exist")
        out_root = Path(output_dir or self.config.export.output_dir) / "gguf"
        out_root.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []
        for quant in self.config.export.quantizations:
            out_path = out_root / f"{self.config.export.gguf_name}-{quant}.gguf"
            cmd = [
                "python",
                "-m",
                "llama_cpp_python.convert",
                "--model",
                str(base_path),
                "--outfile",
                str(out_path),
                "--format",
                "gguf",
                "--quantization",
                quant,
            ]
            subprocess.run(cmd, check=True)
            produced.append(out_path)
        return produced

    def evaluate(self, split: str = "val") -> Any:
        if split not in self.datasets:
            raise ValueError(f"Unknown split {split}")
        trainer = self._make_trainer(f"eval_{split}", self.config.sft, train_split=split, eval_split=split)
        return trainer.evaluate(self.datasets[split])

    def _make_trainer(
        self,
        stage: str,
        phase: PhaseConfig,
        *,
        train_split: str = "train",
        eval_split: Optional[str] = "val",
    ) -> AchmraTrainer:
        output_dir = Path(self.config.export.output_dir) / stage
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_kwarg = {"eval_strategy": "steps" if eval_split else "no"}
        if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames:
            eval_kwarg = {"evaluation_strategy": "steps" if eval_split else "no"}
        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=phase.batch_size,
            per_device_eval_batch_size=phase.batch_size,
            gradient_accumulation_steps=phase.gradient_accumulation,
            learning_rate=phase.learning_rate,
            warmup_ratio=phase.warmup_ratio,
            num_train_epochs=phase.epochs,
            max_steps=phase.steps if phase.steps is not None else -1,
            logging_steps=25,
            eval_steps=100,
            save_steps=200,
            save_total_limit=2,
            bf16=self.config.bf16,
            report_to=[],
            remove_unused_columns=False,
            **eval_kwarg,
        )
        eval_dataset = self.datasets.get(eval_split) if eval_split else None
        return AchmraTrainer(
            model=self.artifacts.model,
            args=args,
            train_dataset=self.datasets[train_split],
            eval_dataset=eval_dataset,
            data_collator=self.collator,
            processing_class=self.tokenizer,
            achmra_config=self.config,
        )


def _load_dataset(path: str | Path, tokenizer, config: AchmraTrainingConfig) -> Dataset:
    dataset = AchmraDataset.from_jsonl(path)
    rendered = [render_training_sample(record, tokenizer, config) for record in dataset.records]
    return Dataset.from_list(rendered)


def build_achmra_pipeline(config: AchmraTrainingConfig | str | Path) -> AchmraPipeline:
    cfg = AchmraTrainingConfig.load(config) if not isinstance(config, AchmraTrainingConfig) else config
    tokenizer = load_tokenizer(cfg.base_model, cfg)
    model_kwargs: dict[str, Any] = {}
    if cfg.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs.setdefault("device_map", "auto")
    artifacts = load_achmra_model(cfg, tokenizer, **model_kwargs)
    if cfg.gradient_checkpointing:
        artifacts.base_model.gradient_checkpointing_enable()
    if cfg.lora.enabled:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        target_modules = cfg.lora.target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        prepare_model_for_kbit_training(artifacts.base_model)
        peft_model = get_peft_model(artifacts.base_model, lora_cfg)
        artifacts.base_model = peft_model
        artifacts.model.model = peft_model
    datasets = {
        "train": _load_dataset(cfg.data.train_file, tokenizer, cfg),
        "val": _load_dataset(cfg.data.val_file, tokenizer, cfg),
    }
    if cfg.data.test_file:
        datasets["test"] = _load_dataset(cfg.data.test_file, tokenizer, cfg)
    if cfg.data.preference_file:
        datasets["preference"] = _load_dataset(cfg.data.preference_file, tokenizer, cfg)
    if cfg.data.rl_file:
        datasets["rl"] = _load_dataset(cfg.data.rl_file, tokenizer, cfg)
    collator = AchmraCollator(tokenizer, cfg)
    return AchmraPipeline(config=cfg, tokenizer=tokenizer, datasets=datasets, artifacts=artifacts, collator=collator)


__all__ = ["AchmraPipeline", "build_achmra_pipeline"]


