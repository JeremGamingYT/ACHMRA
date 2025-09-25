from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer

from .config import AchmraTrainingConfig


class AchmraTrainer(Trainer):
    def __init__(self, *args: Any, achmra_config: AchmraTrainingConfig, **kwargs: Any) -> None:
        self.achmra_config = achmra_config
        super().__init__(*args, **kwargs)
        processor = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processor is None:
            raise ValueError("AchmraTrainer requires a tokenizer/processing_class")
        allowed_ids = [processor.convert_tokens_to_ids(tok) for tok in achmra_config.anti_cot.allowed_tokens]
        self.allowed_token_ids = torch.tensor([idx for idx in allowed_ids if idx is not None and idx >= 0], dtype=torch.long)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: int | None = None):  # type: ignore[override]
        anti_mask = inputs.pop("anti_cot_mask", None)
        outputs = model(**inputs)
        loss = outputs.get("loss")
        if loss is None:
            raise RuntimeError("Base model did not return a loss")
        if outputs.get("latent_loss") is not None:
            loss = loss + outputs["latent_loss"]
        if outputs.get("confidence_loss") is not None:
            loss = loss + outputs["confidence_loss"]
        anti_penalty = None
        if anti_mask is not None and anti_mask.sum() > 0 and self.allowed_token_ids.numel() > 0:
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            allowed_prob = probs.index_select(-1, self.allowed_token_ids.to(logits.device)).sum(dim=-1)
            penalty = ((1.0 - allowed_prob) * anti_mask.to(logits.device)).sum()
            penalty = penalty / (anti_mask.to(logits.device).sum() + 1e-6)
            anti_penalty = penalty * self.achmra_config.anti_cot.loss_weight
            loss = loss + anti_penalty
        outputs["total_loss"] = loss
        if anti_penalty is not None:
            outputs["anti_cot_penalty"] = anti_penalty.detach()
        return (loss, outputs) if return_outputs else loss


__all__ = ["AchmraTrainer"]
