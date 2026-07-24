import dataclasses

import pytest

from latent_wam.config import PROJECT_ROOT, ExperimentConfig, load_config
from latent_wam.data.mixture import DistributedMixtureSampler, LeRobotMixtureDataset


def _fake_mixture() -> LeRobotMixtureDataset:
    dataset = object.__new__(LeRobotMixtureDataset)
    dataset.source_names = ("large", "small")
    dataset.mixture_weights = (0.25, 0.75)
    dataset.source_offsets = [0, 100]
    dataset.source_sizes = [100, 10]
    dataset._cumulative = [100, 110]
    return dataset


def test_distributed_mixture_sampler_shards_one_deterministic_global_stream():
    dataset = _fake_mixture()
    full = list(
        DistributedMixtureSampler(
            dataset,
            num_replicas=1,
            rank=0,
            seed=239,
            epoch_samples=100,
        )
    )
    rank0 = list(
        DistributedMixtureSampler(
            dataset,
            num_replicas=2,
            rank=0,
            seed=239,
            epoch_samples=100,
        )
    )
    rank1 = list(
        DistributedMixtureSampler(
            dataset,
            num_replicas=2,
            rank=1,
            seed=239,
            epoch_samples=100,
        )
    )
    assert rank0 == full[0::2]
    assert rank1 == full[1::2]
    assert all(0 <= index < 110 for index in full)


def test_h800_pilot_declares_five_equal_datasets_and_global_batch_64():
    config = load_config(
        PROJECT_ROOT / "configs" / "h800" / "mixture_stage1_pilot.yaml"
    )
    assert len(config.data.roots) == 6
    assert len(config.data.source_names) == 6
    assert config.data.mixture_weights == (1.0, 1.0, 0.5, 0.5, 1.0, 1.0)
    assert config.data.roots[2].endswith("/InternData-A1/real")
    assert config.data.roots[3].endswith("/InternData-A1/sim_updated")
    assert config.data.roots[4].endswith("/RoboMind")
    assert config.data.roots[5].endswith("/RoboTwin-Randomized/Randomized")
    assert config.data.control_adapter_overrides == {
        "oxe": "oxe_mixed_control",
        "robomind": "robomind_joint_vector",
    }
    assert all(not root.endswith("/InternData-A1/sim") for root in config.data.roots)
    assert any(
        "handover_objects_right_left_part8" in token
        for token in config.data.exclude_contains
    )
    assert config.data.strict_manifest
    assert config.data.mixture_epoch_samples == 3200
    assert config.train.batch_size_per_gpu * config.train.grad_accum_steps * 32 == 64
    assert config.train.max_steps == 50


def test_config_rejects_mixture_length_mismatch():
    base = ExperimentConfig()
    config = dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            roots=("/one", "/two"),
            source_names=("only_one",),
            mixture_weights=(1.0, 1.0),
        ),
    )
    with pytest.raises(ValueError, match="source_names"):
        config.validate()


def test_oxe_adapter_is_blocked_outside_future_stage():
    base = ExperimentConfig()
    config = dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            roots=("/oxe", "/other"),
            source_names=("oxe", "other"),
            mixture_weights=(1.0, 1.0),
            control_adapter_overrides={"oxe": "oxe_mixed_control"},
        ),
        train=dataclasses.replace(base.train, stage="joint"),
    )
    with pytest.raises(ValueError, match=r"SO\(3\) geodesic"):
        config.validate()
