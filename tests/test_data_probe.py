from types import SimpleNamespace

import torch

from latent_wam.config import ExperimentConfig
from latent_wam.data_probe import _validate_sample


def _sample(config):
    action = config.action
    video = config.video
    return SimpleNamespace(
        student=SimpleNamespace(
            context_rgb=torch.zeros(
                3,
                video.context_frames,
                video.resolution,
                video.resolution,
            ),
            instructions=["move the block"],
            proprio=torch.zeros(
                action.proprio_history,
                action.max_proprio_dim,
            ),
            proprio_valid=torch.ones(
                action.proprio_history,
                action.max_proprio_dim,
                dtype=torch.bool,
            ),
            past_actions=torch.zeros(
                action.past_action_history,
                action.max_action_dim,
            ),
            past_action_valid=torch.ones(
                action.past_action_history,
                dtype=torch.bool,
            ),
        ),
        teacher=SimpleNamespace(
            full_rgb=torch.zeros(
                3,
                video.context_frames + video.future_frames,
                video.resolution,
                video.resolution,
            )
        ),
        targets=SimpleNamespace(
            actions=torch.zeros(action.chunk_size, action.max_action_dim),
            action_valid=torch.ones(
                action.chunk_size,
                action.max_action_dim,
                dtype=torch.bool,
            ),
            gripper_mask=torch.zeros(
                action.chunk_size,
                action.max_action_dim,
                dtype=torch.bool,
            ),
        ),
    )


def test_runtime_data_probe_accepts_expected_training_sample_shapes():
    config = ExperimentConfig()
    assert _validate_sample(_sample(config), config) == []


def test_runtime_data_probe_rejects_nonfinite_action():
    config = ExperimentConfig()
    sample = _sample(config)
    sample.targets.actions[0, 0] = float("nan")
    failures = _validate_sample(sample, config)
    assert "actions contains non-finite values" in failures
