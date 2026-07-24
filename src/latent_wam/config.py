from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VideoConfig:
    resolution: int = 384
    context_frames: int = 8
    future_frames: int = 4
    video_fps: float = 4.0
    patch_size: int = 16
    tubelet_size: int = 2
    camera_key: str = "auto"


@dataclass(frozen=True)
class ActionConfig:
    chunk_size: int = 10
    action_hz: float = 10.0
    proprio_history: int = 4
    past_action_history: int = 2
    max_action_dim: int = 32
    max_proprio_dim: int = 64
    schema_buckets: int = 1024


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str = "/mnt/sfs_turbo/fyy/checkpoints/vjepa2/vjepa2_1_vitG_384.pt"
    encoder_embed_dim: int = 1664
    encoder_depth: int = 48
    predictor_embed_dim: int = 384
    predictor_depth: int = 24
    predictor_heads: int = 12
    predictor_mask_tokens: int = 8
    joint_start_block: int = 12
    reciprocal_start_block: int = 18
    joint_attention: str = "time_aligned"
    conditioning_blocks: tuple[int, ...] = (4, 11, 17, 23)
    activation_checkpointing: bool = True
    text_backend: str = "t5"
    text_model: str = "google-t5/t5-large"
    text_local_files_only: bool = True
    max_text_tokens: int = 64
    hash_vocab_size: int = 32768


@dataclass(frozen=True)
class DataConfig:
    root: str = "/mnt/sfs_turbo/rl/InternData-A1/sim"
    roots: tuple[str, ...] = ()
    source_names: tuple[str, ...] = ()
    mixture_weights: tuple[float, ...] = ()
    control_adapter_overrides: dict[str, str] = field(default_factory=dict)
    mixture_epoch_samples: int | None = None
    strict_manifest: bool = False
    backend: str = "lerobot_v21"
    include_globs: tuple[str, ...] = ("**/meta/info.json",)
    exclude_contains: tuple[str, ...] = ("real_lerobotv30", "sim_updated_lerobotv30")
    sample_stride: int = 10
    max_subdatasets: int | None = None
    max_episodes_per_subdataset: int | None = None
    fixed_sample_index: int | None = None
    decode_threads: int = 1
    train_fraction: float = 0.995
    seed: int = 239


@dataclass(frozen=True)
class LossConfig:
    future_weight: float = 1.0
    action_weight: float = 1.0
    smoothness_weight: float = 0.01
    gripper_weight: float = 1.0
    huber_delta: float = 1.0
    normalize_future_levels: bool = True


@dataclass(frozen=True)
class TrainConfig:
    run_name: str = "interndata_a1_joint"
    output_root: str = "outputs"
    stage: str = "joint"
    max_steps: int = 120000
    batch_size_per_gpu: int = 1
    grad_accum_steps: int = 8
    num_workers: int = 4
    predictor_lr: float = 1.0e-5
    new_module_lr: float = 3.0e-4
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.01
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    log_every: int = 10
    save_every: int = 1000
    seed: int = 239
    bf16: bool = True
    deterministic: bool = False
    compile: bool = False
    resume: str | None = None
    init_student: str | None = None
    find_unused_parameters: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if self.video.context_frames % self.video.tubelet_size:
            raise ValueError("context_frames must be divisible by tubelet_size")
        if self.video.future_frames % self.video.tubelet_size:
            raise ValueError("future_frames must be divisible by tubelet_size")
        if self.video.resolution % self.video.patch_size:
            raise ValueError("resolution must be divisible by patch_size")
        if self.model.encoder_embed_dim != 1664 or self.model.encoder_depth != 48:
            raise ValueError("The supported 2B ViT-G checkpoint is 48 layers x 1664 dimensions")
        if (
            self.model.predictor_embed_dim != 384
            or self.model.predictor_depth != 24
            or self.model.predictor_heads != 12
            or self.model.predictor_mask_tokens != 8
        ):
            raise ValueError(
                "The paired V-JEPA 2.1 predictor must be 24 layers x 384 dimensions, "
                "with 12 heads and 8 mask tokens"
            )
        if not (0 <= self.model.joint_start_block < self.model.reciprocal_start_block < self.model.predictor_depth):
            raise ValueError("joint predictor stage boundaries are invalid")
        if (
            len(set(self.model.conditioning_blocks)) != len(self.model.conditioning_blocks)
            or any(
                index < 0 or index >= self.model.predictor_depth
                for index in self.model.conditioning_blocks
            )
        ):
            raise ValueError("conditioning_blocks must be unique valid predictor block indices")
        if self.model.joint_attention not in {"time_aligned", "one_way", "full"}:
            raise ValueError("joint_attention must be time_aligned, one_way, or full")
        if self.model.text_backend not in {"t5", "hash"}:
            raise ValueError("text_backend must be t5 or hash")
        if self.model.max_text_tokens <= 0 or self.model.hash_vocab_size <= 2:
            raise ValueError("text token count must be positive and hash_vocab_size must exceed 2")
        if self.train.stage not in {"future", "action_warmup", "joint"}:
            raise ValueError("stage must be future, action_warmup, or joint")
        if self.train.lr_schedule not in {"cosine", "constant"}:
            raise ValueError("lr_schedule must be cosine or constant")
        positive_train_values = (
            self.action.chunk_size,
            self.action.action_hz,
            self.action.max_action_dim,
            self.action.max_proprio_dim,
            self.train.max_steps,
            self.train.batch_size_per_gpu,
            self.train.grad_accum_steps,
            self.train.log_every,
            self.train.save_every,
        )
        if any(value <= 0 for value in positive_train_values):
            raise ValueError("action dimensions, rates, and training intervals must be positive")
        if self.train.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.data.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        if self.data.roots:
            source_count = len(self.data.roots)
            if any(not root for root in self.data.roots):
                raise ValueError("data roots cannot be empty")
            if len(set(self.data.roots)) != source_count:
                raise ValueError("data roots must be unique")
            if self.data.source_names and len(self.data.source_names) != source_count:
                raise ValueError("source_names must have one entry per data root")
            if self.data.mixture_weights and len(self.data.mixture_weights) != source_count:
                raise ValueError("mixture_weights must have one entry per data root")
            if len(set(self.data.source_names)) != len(self.data.source_names):
                raise ValueError("source_names must be unique")
            if any(
                not math.isfinite(weight) or weight <= 0
                for weight in self.data.mixture_weights
            ):
                raise ValueError("mixture_weights must be finite and positive")
            unknown_adapter_sources = (
                set(self.data.control_adapter_overrides) - set(self.data.source_names)
            )
            if unknown_adapter_sources:
                raise ValueError(
                    "control_adapter_overrides contains unknown source names: "
                    f"{sorted(unknown_adapter_sources)}"
                )
            supported_overrides = {"robomind_joint_vector"}
            unknown_adapters = (
                set(self.data.control_adapter_overrides.values())
                - supported_overrides
            )
            if unknown_adapters:
                raise ValueError(
                    "Unsupported control adapter overrides: "
                    f"{sorted(unknown_adapters)}"
                )
            if self.data.fixed_sample_index is not None:
                raise ValueError("fixed_sample_index is supported only with a single data root")
        elif (
            self.data.source_names
            or self.data.mixture_weights
            or self.data.control_adapter_overrides
        ):
            raise ValueError(
                "source_names, mixture_weights, and control_adapter_overrides "
                "require data.roots"
            )
        elif self.data.mixture_epoch_samples is not None:
            raise ValueError("mixture_epoch_samples requires data.roots")
        if (
            self.data.mixture_epoch_samples is not None
            and self.data.mixture_epoch_samples <= 0
        ):
            raise ValueError("mixture_epoch_samples must be positive")
        if self.data.fixed_sample_index is not None and self.data.fixed_sample_index < 0:
            raise ValueError("fixed_sample_index cannot be negative")
        if not 0.0 < self.data.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.loss.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if any(
            value < 0
            for value in (
                self.loss.future_weight,
                self.loss.action_weight,
                self.loss.smoothness_weight,
                self.loss.gripper_weight,
                self.train.weight_decay,
            )
        ):
            raise ValueError("loss weights and weight_decay cannot be negative")
        if not 0.0 <= self.train.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0.0 < self.train.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1]")

    @property
    def context_tokens(self) -> int:
        spatial = (self.video.resolution // self.video.patch_size) ** 2
        return (self.video.context_frames // self.video.tubelet_size) * spatial

    @property
    def future_tokens(self) -> int:
        spatial = (self.video.resolution // self.video.patch_size) ** 2
        return (self.video.future_frames // self.video.tubelet_size) * spatial


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    known = {f.name for f in dataclasses.fields(instance)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"Unknown config keys for {type(instance).__name__}: {sorted(unknown)}")
    payload: dict[str, Any] = {}
    for field_info in dataclasses.fields(instance):
        current = getattr(instance, field_info.name)
        if field_info.name not in values:
            payload[field_info.name] = current
        elif dataclasses.is_dataclass(current):
            payload[field_info.name] = _merge_dataclass(current, values[field_info.name])
        elif isinstance(current, tuple) and isinstance(values[field_info.name], list):
            payload[field_info.name] = tuple(values[field_info.name])
        else:
            payload[field_info.name] = values[field_info.name]
    return type(instance)(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    config = _merge_dataclass(ExperimentConfig(), values)
    config.validate()
    return config


def resolve_output_root(config: ExperimentConfig) -> Path:
    root = Path(config.train.output_root).expanduser()
    return root if root.is_absolute() else PROJECT_ROOT / root
