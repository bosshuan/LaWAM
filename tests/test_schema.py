import numpy as np

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
