from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_wam.config import ExperimentConfig
from latent_wam.types import JointPrediction, TrainingTargets


@dataclass
class LossOutput:
    total: torch.Tensor
    future: torch.Tensor
    action: torch.Tensor
    gripper: torch.Tensor
    smoothness: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "future_loss": float(self.future.detach()),
            "action_loss": float(self.action.detach()),
            "gripper_loss": float(self.gripper.detach()),
            "smoothness_loss": float(self.smoothness.detach()),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class JointObjective(nn.Module):
    """The only module allowed to combine student predictions with targets."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        prediction: JointPrediction,
        future_targets: tuple[torch.Tensor, ...] | None,
        targets: TrainingTargets,
    ) -> LossOutput:
        if future_targets is None:
            if self.config.train.stage != "action_warmup":
                raise ValueError("Future targets may be omitted only during action warm-up")
            future_loss = prediction.future_concat.sum() * 0.0
        else:
            if len(future_targets) != 4 or len(prediction.future_levels) != 4:
                raise ValueError("Future supervision requires exactly four semantic levels")
            future_losses = []
            for predicted, target in zip(prediction.future_levels, future_targets):
                target = target.detach()
                if self.config.loss.normalize_future_levels:
                    target = F.layer_norm(target.float(), (target.shape[-1],))
                future_losses.append((predicted.float() - target.float()).abs().mean())
            future_loss = torch.stack(future_losses).mean()

        valid = targets.action_valid
        gripper_mask = targets.gripper_mask & valid
        continuous_mask = valid & ~gripper_mask
        huber = F.huber_loss(
            prediction.action_logits,
            targets.actions,
            reduction="none",
            delta=self.config.loss.huber_delta,
        )
        action_loss = _masked_mean(huber, continuous_mask)
        if gripper_mask.any():
            bce = F.binary_cross_entropy_with_logits(
                prediction.action_logits,
                targets.actions.clamp(0, 1),
                reduction="none",
            )
            gripper_loss = _masked_mean(bce, gripper_mask)
        else:
            gripper_loss = prediction.action_logits.sum() * 0.0

        predicted_delta = prediction.action_logits[:, 1:] - prediction.action_logits[:, :-1]
        target_delta = targets.actions[:, 1:] - targets.actions[:, :-1]
        pair_valid = continuous_mask[:, 1:] & continuous_mask[:, :-1]
        smoothness_loss = _masked_mean(
            F.huber_loss(
                predicted_delta,
                target_delta,
                reduction="none",
                delta=self.config.loss.huber_delta,
            ),
            pair_valid,
        )
        future_term = self.config.loss.future_weight * future_loss
        action_term = self.config.loss.action_weight * (
            action_loss
            + self.config.loss.gripper_weight * gripper_loss
            + self.config.loss.smoothness_weight * smoothness_loss
        )
        if self.config.train.stage == "future":
            total = future_term
        elif self.config.train.stage == "action_warmup":
            total = action_term
        else:
            total = future_term + action_term
        return LossOutput(total, future_loss, action_loss, gripper_loss, smoothness_loss)
