import dataclasses

import torch

from latent_wam.config import ExperimentConfig
from latent_wam.objective import JointObjective
from latent_wam.types import JointPrediction, TrainingTargets


def make_prediction():
    levels = tuple(torch.zeros(2, 3, 4, requires_grad=True) for _ in range(4))
    actions = torch.zeros(2, 5, 6, requires_grad=True)
    return JointPrediction(
        future_levels=levels,
        future_concat=torch.cat(levels, dim=-1),
        actions=actions,
        action_logits=actions,
        action_hidden=torch.zeros(2, 5, 8),
    )


def test_padding_does_not_change_valid_action_loss():
    prediction = make_prediction()
    future = tuple(torch.zeros(2, 3, 4) for _ in range(4))
    actions = torch.ones(2, 5, 6)
    valid = torch.zeros_like(actions, dtype=torch.bool)
    valid[:, :, :2] = True
    targets = TrainingTargets(actions, valid, torch.zeros_like(valid), [{}, {}])
    first = JointObjective(ExperimentConfig())(prediction, future, targets)
    actions[:, :, 2:] = 1000
    second = JointObjective(ExperimentConfig())(prediction, future, targets)
    assert torch.allclose(first.action, second.action)


def test_future_and_action_losses_backpropagate():
    prediction = make_prediction()
    future = tuple(torch.ones(2, 3, 4) for _ in range(4))
    valid = torch.ones(2, 5, 6, dtype=torch.bool)
    targets = TrainingTargets(
        actions=torch.ones(2, 5, 6),
        action_valid=valid,
        gripper_mask=torch.zeros_like(valid),
        metadata=[{}, {}],
    )
    output = JointObjective(ExperimentConfig())(prediction, future, targets)
    output.total.backward()
    assert all(level.grad is not None for level in prediction.future_levels)
    assert prediction.action_logits.grad is not None


def test_action_weight_scales_smoothness_too():
    prediction = make_prediction()
    future = tuple(torch.zeros(2, 3, 4) for _ in range(4))
    valid = torch.ones(2, 5, 6, dtype=torch.bool)
    targets = TrainingTargets(
        actions=torch.arange(5, dtype=torch.float32).view(1, 5, 1).expand(2, 5, 6),
        action_valid=valid,
        gripper_mask=torch.zeros_like(valid),
        metadata=[{}, {}],
    )
    base = ExperimentConfig()
    doubled = dataclasses.replace(
        base,
        loss=dataclasses.replace(base.loss, future_weight=0.0, action_weight=2.0),
    )
    first = JointObjective(base)(prediction, future, targets)
    second = JointObjective(doubled)(prediction, future, targets)
    assert torch.allclose(second.total, 2.0 * first.total)
