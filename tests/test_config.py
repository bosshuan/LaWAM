import dataclasses

import pytest

from latent_wam.config import ExperimentConfig


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


def test_rejects_checkpoint_incompatible_predictor_shape():
    config = ExperimentConfig()
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, predictor_mask_tokens=10),
    )
    with pytest.raises(ValueError, match="8 mask tokens"):
        config.validate()
