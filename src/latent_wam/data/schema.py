from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def _flat(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    return np.asarray([value], dtype=np.float32)


@dataclass(frozen=True)
class FeatureNorm:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, value: np.ndarray) -> np.ndarray:
        if self.mean.size != value.size or self.std.size != value.size:
            raise ValueError(
                f"Normalization statistics have shape ({self.mean.size}, {self.std.size}) "
                f"for a feature with {value.size} values"
            )
        return (value - self.mean) / np.maximum(self.std, 1.0e-6)

    def invert(self, value: np.ndarray) -> np.ndarray:
        if self.mean.size != value.size or self.std.size != value.size:
            raise ValueError(
                f"Normalization statistics have shape ({self.mean.size}, {self.std.size}) "
                f"for a feature with {value.size} values"
            )
        return value * np.maximum(self.std, 1.0e-6) + self.mean


@dataclass(frozen=True)
class ActionLossSpec:
    """Schema-native slices used by control losses and prediction decoders."""

    continuous_ranges: tuple[tuple[int, int], ...]
    binary_gripper_ranges: tuple[tuple[int, int], ...]
    rotation_ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ActionSchema:
    name: str
    robot_type: str
    action_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    action_sizes: tuple[int, ...]
    state_sizes: tuple[int, ...]
    gripper_ranges: tuple[tuple[int, int], ...]
    action_norms: dict[str, FeatureNorm]
    state_norms: dict[str, FeatureNorm]

    @property
    def action_dim(self) -> int:
        return sum(self.action_sizes)

    @property
    def state_dim(self) -> int:
        return sum(self.state_sizes)


class ActionSchemaAdapter:
    def __init__(self, schema: ActionSchema, max_action_dim: int, max_state_dim: int):
        if schema.action_dim > max_action_dim:
            raise ValueError(
                f"Schema {schema.name} action dimension {schema.action_dim} exceeds max {max_action_dim}"
            )
        if schema.state_dim > max_state_dim:
            raise ValueError(
                f"Schema {schema.name} state dimension {schema.state_dim} exceeds max {max_state_dim}"
            )
        self.schema = schema
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim

    def loss_spec(self) -> ActionLossSpec:
        """Describe how each encoded action component must be supervised.

        InternData-A1 joint and gripper-position fields are continuous. Only an
        explicitly named gripper ``openness`` field is treated as a binary BCE
        target. Cartesian rotations are deliberately absent from this first
        joint-space adapter instead of being silently treated as scalars.
        """
        continuous: list[tuple[int, int]] = []
        binary_gripper: list[tuple[int, int]] = []
        cursor = 0
        for key, size in zip(self.schema.action_keys, self.schema.action_sizes):
            interval = (cursor, cursor + size)
            if "gripper" in key and "openness" in key:
                binary_gripper.append(interval)
            else:
                continuous.append(interval)
            cursor += size
        return ActionLossSpec(
            continuous_ranges=tuple(continuous),
            binary_gripper_ranges=tuple(binary_gripper),
        )

    def encode(
        self, rows: Iterable[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Public action-encoding interface required by the algorithm spec."""
        return self.encode_actions(rows)

    def encode_actions(self, rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = list(rows)
        result = np.zeros((len(rows), self.max_action_dim), dtype=np.float32)
        valid = np.zeros_like(result, dtype=np.bool_)
        gripper = np.zeros_like(result, dtype=np.bool_)
        for row_index, row in enumerate(rows):
            cursor = 0
            for key, size in zip(self.schema.action_keys, self.schema.action_sizes):
                value = _flat(row[key])
                if value.size != size:
                    raise ValueError(f"Feature {key} has {value.size} values; schema expects {size}")
                if not np.isfinite(value).all():
                    raise ValueError(f"Feature {key} contains non-finite action values")
                binary_gripper = "gripper" in key and "openness" in key
                norm = None if binary_gripper else self.schema.action_norms.get(key)
                if norm is not None:
                    value = norm.apply(value)
                result[row_index, cursor : cursor + size] = value
                valid[row_index, cursor : cursor + size] = True
                if binary_gripper:
                    gripper[row_index, cursor : cursor + size] = True
                cursor += size
        return result, valid, gripper

    def encode_states(self, rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        rows = list(rows)
        result = np.zeros((len(rows), self.max_state_dim), dtype=np.float32)
        valid = np.zeros((len(rows),), dtype=np.bool_)
        for row_index, row in enumerate(rows):
            cursor = 0
            for key, size in zip(self.schema.state_keys, self.schema.state_sizes):
                value = _flat(row[key])
                if value.size != size:
                    raise ValueError(f"Feature {key} has {value.size} values; schema expects {size}")
                if not np.isfinite(value).all():
                    raise ValueError(f"Feature {key} contains non-finite state values")
                norm = self.schema.state_norms.get(key)
                if norm is not None:
                    value = norm.apply(value)
                result[row_index, cursor : cursor + size] = value
                cursor += size
            valid[row_index] = cursor > 0
        return result, valid

    def decode(
        self,
        encoded: np.ndarray,
        *,
        from_logits: bool = False,
    ) -> list[dict[str, np.ndarray]]:
        """Decode padded normalized action rows into dataset-native fields.

        Set ``from_logits=True`` for direct model outputs so explicitly binary
        gripper fields are passed through a sigmoid. Continuous fields are
        denormalized with the same per-dataset statistics used by ``encode``.
        """
        values = np.asarray(encoded, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.max_action_dim:
            raise ValueError(
                "encoded actions must have shape "
                f"[steps, {self.max_action_dim}], got {values.shape}"
            )
        decoded: list[dict[str, np.ndarray]] = []
        for encoded_row in values:
            row: dict[str, np.ndarray] = {}
            cursor = 0
            for key, size in zip(self.schema.action_keys, self.schema.action_sizes):
                value = encoded_row[cursor : cursor + size].copy()
                binary_gripper = "gripper" in key and "openness" in key
                if from_logits and binary_gripper:
                    value = 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
                norm = None if binary_gripper else self.schema.action_norms.get(key)
                if norm is not None:
                    value = norm.invert(value)
                row[key] = value
                cursor += size
            decoded.append(row)
        return decoded
