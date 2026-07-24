from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from latent_wam.config import ExperimentConfig
from latent_wam.data.schema import (
    ActionSchema,
    ActionSchemaAdapter,
    FeatureNorm,
    stable_bucket,
)
from latent_wam.data.video import decode_selected_frames, preprocess_vjepa_clip
from latent_wam.types import (
    StudentInputs,
    TeacherInputs,
    TrainingBatch,
    TrainingTargets,
)


@dataclass(frozen=True)
class EpisodeRecord:
    subdataset_index: int
    parquet_path: Path
    video_path: Path
    episode_index: int
    rows: int
    first_anchor: int
    anchors: int


@dataclass(frozen=True)
class SubdatasetRecord:
    root: Path
    name: str
    fps: float
    camera_key: str
    robot_type: str
    tasks: dict[int, str]
    schema: ActionSchema
    normalization_source: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _feature_size(feature: dict[str, Any]) -> int:
    shape = feature.get("shape", [1])
    return int(np.prod(shape))


def _select_feature_keys(features: dict[str, Any], prefix: str) -> tuple[str, ...]:
    keys = []
    for key in sorted(features):
        if not key.startswith(prefix):
            continue
        if key.startswith("master_actions"):
            continue
        if any(token in key for token in ("joint.position", "gripper.position", "gripper.openness")):
            keys.append(key)
    return tuple(keys)


def _feature_names(feature: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(feature.get("names", ()))
    return tuple(names)


def _is_named_joint_vector(feature: dict[str, Any]) -> bool:
    names = _feature_names(feature)
    if len(names) != _feature_size(feature):
        return False
    joint_tokens = (
        "joint",
        "motor",
        "waist",
        "shoulder",
        "elbow",
        "forearm",
        "wrist",
        "gripper",
    )
    return all(
        any(token in name.lower() for token in joint_tokens)
        for name in names
    )


_OXE_CARTESIAN_ACTION_NAMES = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
)
_OXE_STATE_NAME_VARIANTS = {
    tuple([f"motor_{index}" for index in range(7)] + ["gripper"]),
    ("x", "y", "z", "rx", "ry", "rz", "rw", "gripper"),
    ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"),
    tuple([f"motor_{index}" for index in range(7)] + ["pad"]),
    tuple([f"motor_{index}" for index in range(6)] + ["pad", "gripper"]),
    tuple(f"motor_{index}" for index in range(8)),
}


def _select_oxe_mixed_control(
    features: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    action = features.get("action", {})
    state = features.get("observation.state", {})
    if (
        not isinstance(action, dict)
        or not isinstance(state, dict)
        or action.get("dtype") != "float32"
        or state.get("dtype") != "float32"
        or _feature_size(state) != 8
        or _feature_names(state) not in _OXE_STATE_NAME_VARIANTS
    ):
        return (), (), None

    if (
        _feature_size(action) == 7
        and _feature_names(action) == _OXE_CARTESIAN_ACTION_NAMES
    ):
        return ("action",), ("observation.state",), "oxe_cartesian_euler"

    if _feature_size(action) == 8 and _is_named_joint_vector(action):
        adapter = (
            "oxe_joint_vector"
            if _is_named_joint_vector(state)
            else "oxe_joint_action_pose_state"
        )
        return ("action",), ("observation.state",), adapter

    return (), (), None


def _allow_stats_gr00t(adapter_override: str | None) -> bool:
    return adapter_override in {
        "oxe_mixed_control",
        "robomind_joint_vector",
    }


def _select_control_feature_keys(
    features: dict[str, Any],
    adapter_override: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    """Select only schemas whose control semantics are explicit in metadata."""
    action_keys = _select_feature_keys(features, "actions.")
    state_keys = _select_feature_keys(features, "states.")
    if action_keys and state_keys:
        return action_keys, state_keys, "joint_gripper"

    namespaced_action_keys = tuple(
        key
        for key in ("actions.effector.position", "actions.joint.position")
        if key in features
    )
    namespaced_state_keys = tuple(
        key
        for key in (
            "observation.states.effector.position",
            "observation.states.joint.position",
        )
        if key in features
    )
    effector_names = _feature_names(features.get("actions.effector.position", {}))
    state_effector_names = _feature_names(
        features.get("observation.states.effector.position", {})
    )
    if (
        len(namespaced_action_keys) == 2
        and len(namespaced_state_keys) == 2
        and effector_names
        and state_effector_names
        and all("gripper" in name.lower() for name in effector_names)
        and all("gripper" in name.lower() for name in state_effector_names)
    ):
        return (
            namespaced_action_keys,
            namespaced_state_keys,
            "namespaced_joint_gripper",
        )

    if adapter_override == "oxe_mixed_control":
        return _select_oxe_mixed_control(features)

    if (
        "action" in features
        and "observation.state" in features
        and _is_named_joint_vector(features["action"])
        and _is_named_joint_vector(features["observation.state"])
    ):
        return ("action",), ("observation.state",), "named_joint_vector"

    if adapter_override == "robomind_joint_vector":
        action = features.get("action", {})
        actions_alias = features.get("actions", {})
        state = features.get("observation.state", {})
        action_size = _feature_size(action) if isinstance(action, dict) else 0
        alias_size = (
            _feature_size(actions_alias) if isinstance(actions_alias, dict) else 0
        )
        state_size = _feature_size(state) if isinstance(state, dict) else 0
        if (
            isinstance(action, dict)
            and isinstance(actions_alias, dict)
            and isinstance(state, dict)
            and action.get("dtype") == "float32"
            and actions_alias.get("dtype") == "float32"
            and state.get("dtype") == "float32"
            and _feature_names(action) == ("action",)
            and _feature_names(actions_alias) == ("actions",)
            and _feature_names(state) == ("observation.state",)
            and action_size in {7, 8, 14, 16}
            and action_size == alias_size == state_size
        ):
            return ("action",), ("observation.state",), adapter_override

    return (), (), None


def _gripper_ranges(
    features: dict[str, Any],
    action_keys: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    cursor = 0
    for key in action_keys:
        size = _feature_size(features[key])
        names = _feature_names(features[key])
        if "gripper" in key or (
            names and all("gripper" in name.lower() for name in names)
        ):
            result.append((cursor, cursor + size))
        elif len(names) == size:
            result.extend(
                (cursor + index, cursor + index + 1)
                for index, name in enumerate(names)
                if "gripper" in name.lower()
            )
        cursor += size
    return tuple(result)


def _parse_norms(stats: dict[str, Any], keys: tuple[str, ...]) -> dict[str, FeatureNorm]:
    result: dict[str, FeatureNorm] = {}
    for key in keys:
        entry = stats.get(key, {})
        if not isinstance(entry, dict) or "mean" not in entry or "std" not in entry:
            continue
        mean = np.asarray(entry["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(entry["std"], dtype=np.float32).reshape(-1)
        if (
            mean.size == 0
            or mean.shape != std.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std < 0)
        ):
            raise ValueError(f"Invalid mean/std normalization statistics for {key}")
        result[key] = FeatureNorm(mean=mean, std=std)
    return result


def _aggregate_episode_norms(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[str, FeatureNorm]:
    """Aggregate LeRobot v2.1 per-episode statistics without reading Parquet.

    Each episode stores population mean/std and a scalar frame count. The
    parallel-variance merge below preserves both within-episode and
    between-episode variance while weighting every frame equally.
    """
    aggregates: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for row_index, row in enumerate(rows):
        stats = row.get("stats", {})
        if not isinstance(stats, dict):
            continue
        for key in keys:
            entry = stats.get(key)
            if not isinstance(entry, dict) or not all(
                name in entry for name in ("mean", "std", "count")
            ):
                continue
            mean = np.asarray(entry["mean"], dtype=np.float64).reshape(-1)
            std = np.asarray(entry["std"], dtype=np.float64).reshape(-1)
            counts = np.asarray(entry["count"], dtype=np.float64).reshape(-1)
            if mean.size == 0 or mean.shape != std.shape or counts.size == 0:
                raise ValueError(
                    f"Malformed episode statistics for {key} at row {row_index}"
                )
            if not np.allclose(counts, counts[0]):
                raise ValueError(
                    f"Feature {key} has non-scalar counts at row {row_index}: {counts}"
                )
            count = float(counts[0])
            if (
                count <= 0
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or np.any(std < 0)
            ):
                raise ValueError(
                    f"Invalid episode statistics for {key} at row {row_index}"
                )
            episode_m2 = np.square(std) * count
            if key not in aggregates:
                aggregates[key] = (count, mean.copy(), episode_m2)
                continue
            old_count, old_mean, old_m2 = aggregates[key]
            if old_mean.shape != mean.shape:
                raise ValueError(
                    f"Inconsistent episode-stat shape for {key}: "
                    f"{old_mean.shape} versus {mean.shape}"
                )
            combined_count = old_count + count
            delta = mean - old_mean
            combined_mean = old_mean + delta * (count / combined_count)
            combined_m2 = (
                old_m2
                + episode_m2
                + np.square(delta) * old_count * count / combined_count
            )
            aggregates[key] = (combined_count, combined_mean, combined_m2)

    return {
        key: FeatureNorm(
            mean=mean.astype(np.float32),
            std=np.sqrt(np.maximum(m2 / count, 0.0)).astype(np.float32),
        )
        for key, (count, mean, m2) in aggregates.items()
    }


def _load_norms(
    root: Path,
    keys: tuple[str, ...],
    *,
    allow_stats_gr00t: bool = False,
) -> tuple[dict[str, FeatureNorm], Path]:
    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        return _parse_norms(_read_json(stats_path), keys), stats_path

    episode_stats_path = root / "meta" / "episodes_stats.jsonl"
    if episode_stats_path.is_file():
        rows = _read_jsonl(episode_stats_path)
        return _aggregate_episode_norms(rows, keys), episode_stats_path

    stats_gr00t_path = root / "meta" / "stats_gr00t.json"
    if allow_stats_gr00t and stats_gr00t_path.is_file():
        return _parse_norms(_read_json(stats_gr00t_path), keys), stats_gr00t_path

    expected = f"{stats_path} or {episode_stats_path}"
    if allow_stats_gr00t:
        expected += f", or {stats_gr00t_path}"
    raise FileNotFoundError(
        f"Missing normalization metadata for {root}; expected {expected}"
    )


def _select_camera(features: dict[str, Any], requested: str) -> str:
    video_keys = [
        key for key, value in features.items() if value.get("dtype") == "video"
    ]
    if requested != "auto":
        if requested not in video_keys:
            raise KeyError(f"Requested camera {requested} is not in {video_keys}")
        return requested
    priorities = ("images.rgb.head", "observation.images.main", "observation.images.head")
    for key in priorities:
        if key in video_keys:
            return key
    head = [key for key in video_keys if "head" in key or "main" in key]
    if head:
        return sorted(head)[0]
    if not video_keys:
        raise ValueError("Subdataset has no video feature")
    return sorted(video_keys)[0]


def _resolve_robot_type(
    info: dict[str, Any],
    root: Path,
    adapter_name: str,
) -> str:
    explicit = info.get("robot_type")
    if explicit:
        return str(explicit)
    if adapter_name == "robomind_joint_vector":
        name = root.name.removeprefix("h5_")
        return re.sub(r"_\d+rgb$", "", name)
    if adapter_name.startswith("oxe_"):
        return root.name.removesuffix("_lerobot")
    return root.parent.name


def _episode_number(path: Path) -> int:
    match = re.search(r"episode_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _video_for_episode(root: Path, parquet: Path, camera_key: str) -> Path:
    chunk = parquet.parent.name
    direct = root / "videos" / chunk / camera_key / f"{parquet.stem}.mp4"
    if direct.is_file():
        return direct
    matches = list((root / "videos").glob(f"**/{camera_key}/{parquet.stem}.mp4"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return direct
    raise RuntimeError(f"Ambiguous video files for {parquet}: {matches}")


@lru_cache(maxsize=8)
def _read_episode_table(path: str):
    return pq.read_table(path)


def _table_rows(table, indices: list[int], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    selected = table.select(list(keys)).take(indices)
    columns = {key: selected[key].to_pylist() for key in keys}
    return [
        {key: columns[key][row] for key in keys}
        for row in range(len(indices))
    ]


class InternDataA1Dataset(Dataset[TrainingBatch]):
    """Direct reader for the validated joint-space LeRobot v2.1 schema."""

    def __init__(
        self,
        config: ExperimentConfig,
        split: str = "train",
        adapter_override: str | None = None,
    ):
        if config.data.backend != "lerobot_v21":
            raise ValueError("This MVP supports the official LeRobot v2.1 layout only")
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.config = config
        self.split = split
        self.adapter_override = adapter_override
        self.root = Path(config.data.root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"LeRobot v2.1 root does not exist: {self.root}")
        self.subdatasets: list[SubdatasetRecord] = []
        self.episodes: list[EpisodeRecord] = []
        self._cumulative: list[int] = []
        self._discover()
        if not self.episodes:
            raise RuntimeError(
                f"No usable joint-space LeRobot v2.1 episodes found under {self.root}. "
                "Run latent-wam-preflight to inspect its feature manifest."
            )
        self._raw_samples = self._cumulative[-1]
        self._fixed_sample_index = config.data.fixed_sample_index
        if (
            self._fixed_sample_index is not None
            and self._fixed_sample_index >= self._raw_samples
        ):
            raise IndexError(
                f"fixed_sample_index {self._fixed_sample_index} is outside "
                f"the discovered dataset of {self._raw_samples} samples"
            )

    def _discover(self):
        info_paths: set[Path] = set()
        for pattern in self.config.data.include_globs:
            info_paths.update(self.root.glob(pattern))
        paths = [
            path
            for path in sorted(info_paths)
            if not any(token in str(path) for token in self.config.data.exclude_contains)
        ]
        if self.config.data.max_subdatasets is not None:
            paths = paths[: self.config.data.max_subdatasets]
        total = 0
        for info_path in paths:
            info = _read_json(info_path)
            version = str(info.get("codebase_version", ""))
            if version and not version.startswith("v2"):
                if self.config.data.strict_manifest:
                    raise ValueError(
                        f"Unsupported LeRobot version {version!r} in {info_path}"
                    )
                continue
            root = info_path.parent.parent
            features = info.get("features", {})
            action_keys, state_keys, adapter_name = _select_control_feature_keys(
                features,
                self.adapter_override,
            )
            if not action_keys or not state_keys or adapter_name is None:
                if self.config.data.strict_manifest:
                    raise ValueError(
                        f"Unsupported control manifest in {info_path}: "
                        f"feature keys are {sorted(features)}"
                    )
                continue
            camera = _select_camera(features, self.config.video.camera_key)
            all_keys = (*action_keys, *state_keys)
            norms, normalization_path = _load_norms(
                root,
                all_keys,
                allow_stats_gr00t=_allow_stats_gr00t(self.adapter_override),
            )
            action_norms = {key: norms[key] for key in action_keys if key in norms}
            state_norms = {key: norms[key] for key in state_keys if key in norms}
            missing_norms = [
                key
                for key in all_keys
                if not ("gripper" in key and "openness" in key)
                and key not in norms
            ]
            if missing_norms:
                raise ValueError(
                    f"Missing mean/std statistics in {normalization_path} "
                    f"for {missing_norms}"
                )
            for key, norm in norms.items():
                expected_size = _feature_size(features[key])
                if norm.mean.size != expected_size or norm.std.size != expected_size:
                    raise ValueError(
                        f"Normalization statistics in {normalization_path} for {key} "
                        f"have shape ({norm.mean.size}, {norm.std.size}); "
                        f"expected {expected_size}"
                    )
            robot_type = _resolve_robot_type(info, root, adapter_name)
            schema_name = f"{robot_type}:{'|'.join(action_keys)}"
            if adapter_name != "joint_gripper":
                action_signature = "|".join(
                    f"{key}[{_feature_size(features[key])}]" for key in action_keys
                )
                state_signature = "|".join(
                    f"{key}[{_feature_size(features[key])}]" for key in state_keys
                )
                schema_name = (
                    f"{adapter_name}:{robot_type}:"
                    f"actions={action_signature}:states={state_signature}"
                )
            action_sizes = tuple(_feature_size(features[key]) for key in action_keys)
            schema = ActionSchema(
                name=schema_name,
                robot_type=robot_type,
                action_keys=action_keys,
                state_keys=state_keys,
                action_sizes=action_sizes,
                state_sizes=tuple(_feature_size(features[key]) for key in state_keys),
                gripper_ranges=_gripper_ranges(features, action_keys),
                action_norms=action_norms,
                state_norms=state_norms,
            )
            ActionSchemaAdapter(
                schema,
                self.config.action.max_action_dim,
                self.config.action.max_proprio_dim,
            )
            task_rows = _read_jsonl(root / "meta" / "tasks.jsonl")
            tasks = {
                int(row.get("task_index", index)): str(
                    row.get("task", row.get("name", root.name.replace("_", " ")))
                )
                for index, row in enumerate(task_rows)
            }
            record = SubdatasetRecord(
                root=root,
                name=str(root.relative_to(self.root)),
                fps=float(info.get("fps", 30)),
                camera_key=camera,
                robot_type=robot_type,
                tasks=tasks,
                schema=schema,
                normalization_source=normalization_path.name,
            )
            subdataset_index = len(self.subdatasets)
            self.subdatasets.append(record)
            parquets = sorted((root / "data").glob("**/episode_*.parquet"))
            if self.config.data.max_episodes_per_subdataset is not None:
                parquets = parquets[: self.config.data.max_episodes_per_subdataset]
            for parquet in parquets:
                episode_index = _episode_number(parquet)
                fraction_key = (episode_index * 2654435761) % 1000000 / 1000000.0
                in_train = fraction_key < self.config.data.train_fraction
                if (self.split == "train") != in_train:
                    continue
                rows = pq.ParquetFile(parquet).metadata.num_rows
                first, count = self._anchor_range(rows, record.fps)
                if count <= 0:
                    continue
                video = _video_for_episode(root, parquet, camera)
                if not video.is_file():
                    continue
                self.episodes.append(
                    EpisodeRecord(
                        subdataset_index,
                        parquet,
                        video,
                        episode_index,
                        rows,
                        first,
                        count,
                    )
                )
                total += count
                self._cumulative.append(total)

    def _anchor_range(self, rows: int, fps: float) -> tuple[int, int]:
        context_span = (self.config.video.context_frames - 1) / self.config.video.video_fps
        state_span = (self.config.action.proprio_history - 1) / self.config.action.action_hz
        past_span = self.config.action.past_action_history / self.config.action.action_hz
        before = int(np.ceil(max(context_span, state_span, past_span) * fps))
        after = int(np.ceil(max(
            self.config.video.future_frames / self.config.video.video_fps,
            self.config.action.chunk_size / self.config.action.action_hz,
        ) * fps))
        last = rows - 1 - after
        if last < before:
            return before, 0
        stride = self.config.data.sample_stride
        return before, (last - before) // stride + 1

    def __len__(self):
        return 1 if self._fixed_sample_index is not None else self._raw_samples

    def _locate(self, index: int) -> tuple[EpisodeRecord, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        raw_index = (
            self._fixed_sample_index
            if self._fixed_sample_index is not None
            else index
        )
        assert raw_index is not None
        episode_pos = bisect.bisect_right(self._cumulative, raw_index)
        previous = 0 if episode_pos == 0 else self._cumulative[episode_pos - 1]
        episode = self.episodes[episode_pos]
        anchor = episode.first_anchor + (
            raw_index - previous
        ) * self.config.data.sample_stride
        return episode, anchor

    def audit_summary(self) -> dict[str, Any]:
        """Return lightweight schema and sampling metadata for startup logs."""
        if self._fixed_sample_index is not None:
            episode, anchor = self._locate(0)
            selected_indices = {episode.subdataset_index}
            fixed_sample = {
                "raw_index": self._fixed_sample_index,
                "episode": episode.episode_index,
                "anchor": anchor,
            }
        else:
            selected_indices = {episode.subdataset_index for episode in self.episodes}
            fixed_sample = None
        schema_variants: dict[str, dict[str, Any]] = {}
        for index in sorted(selected_indices):
            record = self.subdatasets[index]
            schema = record.schema
            schema_variants.setdefault(
                schema.name,
                {
                    "schema": schema.name,
                    "robot_type": schema.robot_type,
                    "action_keys": list(schema.action_keys),
                    "state_keys": list(schema.state_keys),
                    "action_dim": schema.action_dim,
                    "state_dim": schema.state_dim,
                    "normalization_source": record.normalization_source,
                },
            )
        return {
            "raw_samples": self._raw_samples,
            "effective_samples": len(self),
            "fixed_sample": fixed_sample,
            "subdataset_count": len(selected_indices),
            "episode_count": (
                1 if self._fixed_sample_index is not None else len(self.episodes)
            ),
            "subdatasets": [
                self.subdatasets[index].name for index in sorted(selected_indices)
            ],
            "schema_variants": list(schema_variants.values()),
        }

    @staticmethod
    def _indices(anchor: int, offsets: np.ndarray, fps: float) -> list[int]:
        return [int(round(anchor + offset * fps)) for offset in offsets]

    def __getitem__(self, index: int) -> TrainingBatch:
        episode, anchor = self._locate(index)
        source = self.subdatasets[episode.subdataset_index]
        video = self.config.video
        action = self.config.action
        context_offsets = np.arange(-(video.context_frames - 1), 1) / video.video_fps
        future_offsets = np.arange(1, video.future_frames + 1) / video.video_fps
        visual_indices = self._indices(anchor, np.concatenate([context_offsets, future_offsets]), source.fps)
        frames = decode_selected_frames(
            episode.video_path,
            visual_indices,
            source.fps,
            self.config.data.decode_threads,
        )
        full_rgb = preprocess_vjepa_clip(frames, video.resolution)
        context_rgb = full_rgb[:, : video.context_frames]

        target_indices = self._indices(
            anchor,
            np.arange(1, action.chunk_size + 1) / action.action_hz,
            source.fps,
        )
        proprio_indices = self._indices(
            anchor,
            np.arange(-(action.proprio_history - 1), 1) / action.action_hz,
            source.fps,
        )
        past_action_indices = self._indices(
            anchor,
            -np.arange(action.past_action_history, 0, -1) / action.action_hz,
            source.fps,
        )
        table = _read_episode_table(str(episode.parquet_path))
        adapter = ActionSchemaAdapter(
            source.schema, action.max_action_dim, action.max_proprio_dim
        )
        action_rows = _table_rows(table, target_indices, source.schema.action_keys)
        target, target_valid, gripper = adapter.encode_actions(action_rows)
        proprio_rows = _table_rows(table, proprio_indices, source.schema.state_keys)
        proprio, proprio_valid = adapter.encode_states(proprio_rows)
        past_rows = _table_rows(table, past_action_indices, source.schema.action_keys)
        past_actions, past_valid_components, _ = adapter.encode_actions(past_rows)

        task_index = 0
        if "task_index" in table.column_names:
            value = table["task_index"][anchor].as_py()
            task_index = int(value[0] if isinstance(value, list) else value)
        instruction = source.tasks.get(task_index, source.root.name.replace("_", " "))
        schema_id = stable_bucket(source.schema.name, action.schema_buckets)
        embodiment_id = stable_bucket(source.robot_type, action.schema_buckets)
        return TrainingBatch(
            student=StudentInputs(
                context_rgb=context_rgb,
                instructions=[instruction],
                proprio=torch.from_numpy(proprio),
                proprio_valid=torch.from_numpy(proprio_valid),
                past_actions=torch.from_numpy(past_actions),
                past_action_valid=torch.from_numpy(past_valid_components.any(axis=1)),
                embodiment_ids=torch.tensor([embodiment_id], dtype=torch.long),
                schema_ids=torch.tensor([schema_id], dtype=torch.long),
            ),
            teacher=TeacherInputs(full_rgb=full_rgb),
            targets=TrainingTargets(
                actions=torch.from_numpy(target),
                action_valid=torch.from_numpy(target_valid),
                gripper_mask=torch.from_numpy(gripper),
                metadata=[
                    {
                        "subdataset": source.name,
                        "episode": episode.episode_index,
                        "anchor": anchor,
                        "camera": source.camera_key,
                        "schema": source.schema.name,
                        "normalization_source": source.normalization_source,
                    }
                ],
            ),
        )


def collate_training_batch(samples: list[TrainingBatch]) -> TrainingBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return TrainingBatch(
        student=StudentInputs(
            context_rgb=torch.stack([sample.student.context_rgb for sample in samples]),
            instructions=[sample.student.instructions[0] for sample in samples],
            proprio=torch.stack([sample.student.proprio for sample in samples]),
            proprio_valid=torch.stack([sample.student.proprio_valid for sample in samples]),
            past_actions=torch.stack([sample.student.past_actions for sample in samples]),
            past_action_valid=torch.stack([sample.student.past_action_valid for sample in samples]),
            embodiment_ids=torch.cat([sample.student.embodiment_ids for sample in samples]),
            schema_ids=torch.cat([sample.student.schema_ids for sample in samples]),
        ),
        teacher=TeacherInputs(
            full_rgb=torch.stack([sample.teacher.full_rgb for sample in samples])
        ),
        targets=TrainingTargets(
            actions=torch.stack([sample.targets.actions for sample in samples]),
            action_valid=torch.stack([sample.targets.action_valid for sample in samples]),
            gripper_mask=torch.stack([sample.targets.gripper_mask for sample in samples]),
            metadata=[sample.targets.metadata[0] for sample in samples],
        ),
    )
