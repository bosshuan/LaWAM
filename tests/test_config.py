import dataclasses

import pytest

from latent_wam.config import PROJECT_ROOT, ExperimentConfig, load_config


def test_canonical_token_counts():
    config = ExperimentConfig()
    config.validate()
    assert config.context_tokens == 2304
    assert config.future_tokens == 1152
    assert 4 * config.model.encoder_embed_dim == 6656


def test_vitg_is_uppercase_g_architecture():
    config = ExperimentConfig()
    assert config.model.encoder_depth == 48
    assert config.model.encoder_embed_dim == 1664
    assert not config.train.deterministic


def test_rejects_checkpoint_incompatible_predictor_shape():
    config = ExperimentConfig()
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, predictor_mask_tokens=10),
    )
    with pytest.raises(ValueError, match="8 mask tokens"):
        config.validate()


def test_rejects_invalid_debug_sampling_and_lr_schedule():
    base = ExperimentConfig()
    negative_sample = dataclasses.replace(
        base,
        data=dataclasses.replace(base.data, fixed_sample_index=-1),
    )
    with pytest.raises(ValueError, match="fixed_sample_index"):
        negative_sample.validate()

    invalid_schedule = dataclasses.replace(
        base,
        train=dataclasses.replace(base.train, lr_schedule="linear"),
    )
    with pytest.raises(ValueError, match="lr_schedule"):
        invalid_schedule.validate()


def test_resume_audit_config_uses_controlled_deterministic_runtime():
    config = load_config(
        PROJECT_ROOT / "configs" / "debug" / "interndata_a1_resume_audit.yaml"
    )
    assert config.train.deterministic
    assert config.train.num_workers == 0
    assert config.train.save_every == 3
    assert config.data.fixed_sample_index == 0
