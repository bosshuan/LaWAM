from __future__ import annotations

import torch
import torch.nn as nn

from latent_wam.config import ExperimentConfig
from latent_wam.models.conditioning import build_text_encoder
from latent_wam.models.joint_predictor import JointStudent
from latent_wam.models.vjepa import LoadReport, VJEPA21ViTGAdapter
from latent_wam.types import JointPrediction, StudentInputs, TeacherInputs


class LatentWAM(nn.Module):
    """Public inference facade with a target-free ``predict`` signature."""

    def __init__(
        self,
        encoder: VJEPA21ViTGAdapter,
        text_encoder: nn.Module,
        student: JointStudent,
        load_report: LoadReport,
    ):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.student = student
        self.load_report = load_report

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "LatentWAM":
        encoder, predictor, report = VJEPA21ViTGAdapter.from_checkpoint(config)
        text_encoder = build_text_encoder(config)
        return cls(
            encoder,
            text_encoder,
            JointStudent(predictor, config, text_encoder.output_dim),
            report,
        )

    @torch.no_grad()
    def encode_context(self, inputs: StudentInputs) -> torch.Tensor:
        return self.encoder.encode_context(inputs.context_rgb)

    @torch.no_grad()
    def encode_target(self, inputs: TeacherInputs) -> tuple[torch.Tensor, ...]:
        return self.encoder.encode_target(inputs.full_rgb)

    @torch.no_grad()
    def encode_text(
        self, instructions: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.text_encoder.eval()
        return self.text_encoder(instructions, device)

    def predict(self, inputs: StudentInputs) -> JointPrediction:
        context = self.encode_context(inputs)
        text_features, text_valid = self.encode_text(
            inputs.instructions, inputs.context_rgb.device
        )
        return self.student(context, inputs, text_features, text_valid)

    def forward(self, inputs: StudentInputs) -> JointPrediction:
        return self.predict(inputs)
