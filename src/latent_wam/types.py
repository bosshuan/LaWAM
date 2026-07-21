from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


Tensor = torch.Tensor


@dataclass
class StudentInputs:
    context_rgb: Tensor
    instructions: list[str]
    proprio: Tensor
    proprio_valid: Tensor
    past_actions: Tensor
    past_action_valid: Tensor
    embodiment_ids: Tensor
    schema_ids: Tensor

    def to(self, device: torch.device, non_blocking: bool = True) -> "StudentInputs":
        return StudentInputs(
            context_rgb=self.context_rgb.to(device, non_blocking=non_blocking),
            instructions=self.instructions,
            proprio=self.proprio.to(device, non_blocking=non_blocking),
            proprio_valid=self.proprio_valid.to(device, non_blocking=non_blocking),
            past_actions=self.past_actions.to(device, non_blocking=non_blocking),
            past_action_valid=self.past_action_valid.to(device, non_blocking=non_blocking),
            embodiment_ids=self.embodiment_ids.to(device, non_blocking=non_blocking),
            schema_ids=self.schema_ids.to(device, non_blocking=non_blocking),
        )


@dataclass
class TeacherInputs:
    full_rgb: Tensor

    def to(self, device: torch.device, non_blocking: bool = True) -> "TeacherInputs":
        return TeacherInputs(self.full_rgb.to(device, non_blocking=non_blocking))


@dataclass
class TrainingTargets:
    actions: Tensor
    action_valid: Tensor
    gripper_mask: Tensor
    metadata: list[dict[str, Any]]

    def to(self, device: torch.device, non_blocking: bool = True) -> "TrainingTargets":
        return TrainingTargets(
            actions=self.actions.to(device, non_blocking=non_blocking),
            action_valid=self.action_valid.to(device, non_blocking=non_blocking),
            gripper_mask=self.gripper_mask.to(device, non_blocking=non_blocking),
            metadata=self.metadata,
        )


@dataclass
class JointPrediction:
    future_levels: tuple[Tensor, Tensor, Tensor, Tensor]
    future_concat: Tensor
    actions: Tensor
    action_logits: Tensor
    action_hidden: Tensor


@dataclass
class TrainingBatch:
    student: StudentInputs
    teacher: TeacherInputs
    targets: TrainingTargets

    def to(self, device: torch.device, non_blocking: bool = True) -> "TrainingBatch":
        return TrainingBatch(
            student=self.student.to(device, non_blocking),
            teacher=self.teacher.to(device, non_blocking),
            targets=self.targets.to(device, non_blocking),
        )
