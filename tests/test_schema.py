import json

import numpy as np

from latent_wam.data.intern_data_a1 import (
    _aggregate_episode_norms,
    _load_norms,
    _select_control_feature_keys,
)
from latent_wam.data.schema import ActionSchema, ActionSchemaAdapter, FeatureNorm


def test_schema_padding_and_masks():
    schema = ActionSchema(
        name="arm",
        robot_type="test",
        action_keys=("actions.joint.position", "actions.gripper.openness"),
        state_keys=("states.joint.position",),
        action_sizes=(2, 1),
        state_sizes=(2,),
        gripper_ranges=((2, 3),),
        action_norms={},
        state_norms={},
    )
    adapter = ActionSchemaAdapter(schema, max_action_dim=5, max_state_dim=4)
    values, valid, gripper = adapter.encode_actions(
        [{"actions.joint.position": [1, 2], "actions.gripper.openness": [1]}]
    )
    assert values.shape == (1, 5)
    assert valid.tolist() == [[True, True, True, False, False]]
    assert gripper.tolist() == [[False, False, True, False, False]]
    assert np.allclose(values[0, :3], [1, 2, 1])


def test_schema_encode_decode_round_trip_and_loss_spec():
    schema = ActionSchema(
        name="arm",
        robot_type="test",
        action_keys=("actions.joint.position", "actions.gripper.openness"),
        state_keys=("states.joint.position",),
        action_sizes=(2, 1),
        state_sizes=(2,),
        gripper_ranges=((2, 3),),
        action_norms={
            "actions.joint.position": FeatureNorm(
                mean=np.array([1.0, -1.0], dtype=np.float32),
                std=np.array([2.0, 4.0], dtype=np.float32),
            )
        },
        state_norms={},
    )
    adapter = ActionSchemaAdapter(schema, max_action_dim=5, max_state_dim=4)
    rows = [
        {
            "actions.joint.position": np.array([3.0, 7.0], dtype=np.float32),
            "actions.gripper.openness": np.array([1.0], dtype=np.float32),
        }
    ]
    encoded, valid, gripper = adapter.encode(rows)
    decoded = adapter.decode(encoded)
    assert np.allclose(decoded[0]["actions.joint.position"], [3.0, 7.0])
    assert np.allclose(decoded[0]["actions.gripper.openness"], [1.0])
    assert valid.tolist() == [[True, True, True, False, False]]
    assert gripper.tolist() == [[False, False, True, False, False]]
    spec = adapter.loss_spec()
    assert spec.continuous_ranges == ((0, 2),)
    assert spec.binary_gripper_ranges == ((2, 3),)
    assert spec.rotation_ranges == ()


def test_aggregate_lerobot_v21_episode_statistics_by_frame_count():
    key = "actions.joint.position"
    rows = [
        {
            "episode_index": 0,
            "stats": {
                key: {
                    "mean": [1.0, 3.0],
                    "std": [1.0, 1.0],
                    "count": [2],
                }
            },
        },
        {
            "episode_index": 1,
            "stats": {
                key: {
                    "mean": [5.0, 7.0],
                    "std": [1.0, 1.0],
                    "count": [4],
                }
            },
        },
    ]
    norm = _aggregate_episode_norms(rows, (key,))[key]
    expected_frames = np.array(
        [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [4.0, 6.0], [6.0, 8.0], [6.0, 8.0]]
    )
    assert np.allclose(norm.mean, expected_frames.mean(axis=0))
    assert np.allclose(norm.std, expected_frames.std(axis=0))


def test_selects_namespaced_joint_gripper_schema():
    features = {
        "actions.effector.position": {
            "dtype": "float32",
            "shape": [2],
            "names": ["left_gripper", "right_gripper"],
        },
        "actions.joint.position": {
            "dtype": "float32",
            "shape": [14],
            "names": [f"joint_{index}" for index in range(14)],
        },
        "observation.states.effector.position": {
            "dtype": "float32",
            "shape": [2],
            "names": ["left_gripper", "right_gripper"],
        },
        "observation.states.joint.position": {
            "dtype": "float32",
            "shape": [14],
            "names": [f"joint_{index}" for index in range(14)],
        },
    }
    action_keys, state_keys, adapter = _select_control_feature_keys(features)
    assert action_keys == (
        "actions.effector.position",
        "actions.joint.position",
    )
    assert state_keys == (
        "observation.states.effector.position",
        "observation.states.joint.position",
    )
    assert adapter == "namespaced_joint_gripper"


def test_selects_named_robotwin_joint_vector():
    names = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    features = {
        "action": {"dtype": "float32", "shape": [14], "names": [names]},
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": [names],
        },
    }
    assert _select_control_feature_keys(features) == (
        ("action",),
        ("observation.state",),
        "named_joint_vector",
    )


def test_rejects_cartesian_and_opaque_vector_schemas():
    cartesian = {
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
    }
    opaque = {
        "action": {"dtype": "float32", "shape": [14], "names": ["action"]},
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": ["observation.state"],
        },
    }
    assert _select_control_feature_keys(cartesian) == ((), (), None)
    assert _select_control_feature_keys(opaque) == ((), (), None)


def test_selects_only_explicitly_overridden_robomind_vector():
    features = {
        "action": {"dtype": "float32", "shape": [8], "names": ["action"]},
        "actions": {"dtype": "float32", "shape": [8], "names": ["actions"]},
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": ["observation.state"],
        },
    }
    assert _select_control_feature_keys(features) == ((), (), None)
    assert _select_control_feature_keys(
        features,
        "robomind_joint_vector",
    ) == (
        ("action",),
        ("observation.state",),
        "robomind_joint_vector",
    )


def test_loads_stats_gr00t_as_last_normalization_fallback(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    stats = {
        "action": {"mean": [1.0, 2.0], "std": [3.0, 4.0]},
        "observation.state": {"mean": [5.0, 6.0], "std": [7.0, 8.0]},
    }
    (meta / "stats_gr00t.json").write_text(
        json.dumps(stats),
        encoding="utf-8",
    )
    norms, source = _load_norms(
        tmp_path,
        ("action", "observation.state"),
        allow_stats_gr00t=True,
    )
    assert source == meta / "stats_gr00t.json"
    assert np.allclose(norms["action"].mean, [1.0, 2.0])
    assert np.allclose(norms["observation.state"].std, [7.0, 8.0])
