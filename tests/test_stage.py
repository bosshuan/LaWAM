import dataclasses

import torch
import torch.nn as nn

from latent_wam.config import ExperimentConfig
from latent_wam.train import build_scheduler, configure_stage


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = nn.Module()
        self.predictor.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, 4))]
        )
        self.predictor.predictor_proj_context = nn.Linear(4, 4)
        self.predictor.img_mod_embed = nn.Parameter(torch.zeros(1, 1, 4))
        self.predictor.video_mod_embed = nn.Parameter(torch.zeros(1, 1, 4))
        self.predictor.block = nn.Linear(4, 4)
        self.future_mask_token = nn.Parameter(torch.zeros(1, 1, 4))
        self.action_queries = nn.Parameter(torch.zeros(1, 2, 4))
        self.action_time_embedding = nn.Linear(1, 4)
        self.action_condition_projection = nn.Linear(4, 4)
        self.action_norm = nn.LayerNorm(4)
        self.action_head = nn.Linear(4, 2)


def test_joint_stage_freezes_only_unused_pretrained_paths():
    student = TinyStudent()
    configure_stage(student, "joint")
    assert not student.predictor.mask_tokens[0].requires_grad
    assert not student.predictor.predictor_proj_context.weight.requires_grad
    assert not student.predictor.img_mod_embed.requires_grad
    assert student.predictor.video_mod_embed.requires_grad
    assert student.predictor.block.weight.requires_grad
    assert student.action_queries.requires_grad


def test_future_and_action_warmup_stage_freezing():
    student = TinyStudent()
    configure_stage(student, "future")
    assert student.predictor.block.weight.requires_grad
    assert student.future_mask_token.requires_grad
    assert not student.action_queries.requires_grad
    assert not student.action_head.weight.requires_grad

    configure_stage(student, "action_warmup")
    assert not student.predictor.block.weight.requires_grad
    assert not student.future_mask_token.requires_grad
    assert student.action_queries.requires_grad
    assert student.action_head.weight.requires_grad


def test_constant_debug_scheduler_keeps_learning_rate_fixed():
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    base = ExperimentConfig()
    config = dataclasses.replace(
        base,
        train=dataclasses.replace(base.train, lr_schedule="constant"),
    )
    scheduler = build_scheduler(optimizer, config)
    observed = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed == [1.0e-3] * 5
