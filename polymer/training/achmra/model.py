from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from .config import AchmraTrainingConfig


@dataclass
class AchmraModelArtifacts:
    model: nn.Module
    base_model: PreTrainedModel


class ConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        hidden_mid = max(hidden_size // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_mid),
            nn.GELU(),
            nn.Linear(hidden_mid, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sigmoid(self.net(hidden_states))


class AchmraReasoningWrapper(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: AchmraTrainingConfig, conf_token_id: int) -> None:
        super().__init__()
        self.model = base_model
        hidden_size = base_model.config.hidden_size
        self.latent_head = nn.Linear(hidden_size, config.latent_passes.embedding_dim)
        self.confidence_head = ConfidenceHead(hidden_size)
        self.conf_token_id = conf_token_id
        self.training_config = config

    @property
    def config(self):
        return self.model.config

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, **inputs: Any) -> dict[str, Any]:  # noqa: D401
        aux_keys = {
            "confidence_target",
            "latent_targets",
            "latent_positions",
            "latent_mask",
            "thought_graph",
        }
        base_inputs = {k: v for k, v in inputs.items() if k not in aux_keys}
        base_inputs["output_hidden_states"] = True
        base_inputs["return_dict"] = True
        outputs = self.model(**base_inputs)
        hidden = outputs.hidden_states[-1]
        result: dict[str, Any] = {
            "logits": outputs.logits,
            "loss": outputs.loss,
            "hidden_states": hidden,
        }
        latent_targets = inputs.get("latent_targets")
        latent_positions = inputs.get("latent_positions")
        latent_mask = inputs.get("latent_mask")
        if latent_targets is not None and latent_positions is not None and latent_mask is not None and latent_targets.numel() > 0:
            result["latent_loss"] = self._latent_loss(hidden, latent_targets, latent_positions, latent_mask)
        conf_target = inputs.get("confidence_target")
        if conf_target is not None:
            conf_loss, conf_pred = self._confidence_loss(hidden, inputs["input_ids"], conf_target)
            result["confidence_loss"] = conf_loss
            result["confidence_pred"] = conf_pred
        return result

    def _latent_loss(
        self,
        hidden_states: torch.Tensor,
        latent_targets: torch.Tensor,
        latent_positions: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        if latent_targets.size(1) == 0:
            return torch.tensor(0.0, device=hidden_states.device)
        seq_len = hidden_states.size(1)
        hidden_size = hidden_states.size(2)
        safe_positions = latent_positions.clone().clamp(min=0, max=max(seq_len - 1, 0))
        gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        gathered = torch.gather(hidden_states, 1, gather_index)
        preds = self.latent_head(gathered)
        diff = (preds - latent_targets) ** 2
        masked = diff * latent_mask.unsqueeze(-1)
        denom = latent_mask.sum() + 1e-6
        return masked.sum() / denom * self.training_config.latent_passes.loss_weight

    def _confidence_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = input_ids == self.conf_token_id
        if not mask.any():
            zero = torch.tensor(0.0, device=hidden_states.device)
            return zero, torch.zeros_like(target)
        batch_idx, token_idx = torch.nonzero(mask, as_tuple=True)
        conf_hidden = hidden_states[batch_idx, token_idx]
        pred = self.confidence_head(conf_hidden).squeeze(-1)
        target_vals = target[batch_idx]
        loss = torch.nn.functional.mse_loss(pred, target_vals, reduction="mean")
        if self.training_config.calibration.loss_type == "brier":
            loss = torch.nn.functional.mse_loss(pred, target_vals, reduction="mean")
        if self.training_config.calibration.temperature_scaling:
            loss = loss * 0.5
        full_pred = torch.zeros_like(target)
        full_pred[batch_idx] = pred
        return loss * self.training_config.calibration.loss_weight, full_pred


def load_achmra_model(
    config: AchmraTrainingConfig,
    tokenizer: PreTrainedTokenizerBase,
    **model_kwargs: Any,
) -> AchmraModelArtifacts:
    torch_dtype = torch.bfloat16 if config.bf16 else None
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
        torch_dtype=torch_dtype,
        **model_kwargs,
    )
    base.resize_token_embeddings(len(tokenizer))
    conf_token_id = tokenizer.convert_tokens_to_ids(config.calibration.confidence_token)
    if conf_token_id is None or conf_token_id < 0:
        raise ValueError("Tokenizer must contain the [CONF] token before loading the model")
    wrapper = AchmraReasoningWrapper(base, config, conf_token_id)
    return AchmraModelArtifacts(model=wrapper, base_model=base)


__all__ = ["AchmraReasoningWrapper", "load_achmra_model", "AchmraModelArtifacts"]


