import torch

from latent_wam.models.joint_attention import build_joint_attention_mask


def test_one_way_stage_hides_context_from_actions():
    future_ends = torch.tensor([0.5, 1.0])
    action_times = torch.tensor([0.25, 0.75])
    mask = build_joint_attention_mask(3, future_ends, action_times, reciprocal=False)
    assert mask.shape == (7, 7)
    assert not mask[5:, :3].any()
    assert mask[5:, 3:5].all()
    assert not mask[:5, 5:].any()


def test_reciprocal_stage_is_time_aligned():
    future_ends = torch.tensor([0.5, 1.0])
    action_times = torch.tensor([0.25, 0.75])
    mask = build_joint_attention_mask(1, future_ends, action_times, reciprocal=True)
    # A(0.25) can read both future intervals; A(0.75) only the second.
    assert mask[3, 1:3].tolist() == [True, True]
    assert mask[4, 1:3].tolist() == [False, True]
    # F(0.5) reads A(0.25), while F(1.0) reads both actions.
    assert mask[1, 3:5].tolist() == [True, False]
    assert mask[2, 3:5].tolist() == [True, True]
    assert not mask[3:, :1].any()


def test_action_queries_are_joint_but_never_read_raw_context():
    future_ends = torch.linspace(0.5, 1.0, 8)
    action_times = torch.linspace(0.1, 1.0, 10)
    for reciprocal in (False, True):
        mask = build_joint_attention_mask(16, future_ends, action_times, reciprocal)
        assert not mask[-10:, :16].any()
        assert mask[-10:, -10:].all()
