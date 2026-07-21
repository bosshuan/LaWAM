from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from latent_wam.config import ExperimentConfig
from latent_wam.vendor.vjepa21 import vit_gigantic_xformers, vit_predictor


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        cleaned[key] = value
    return cleaned


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dict checkpoint, got {type(checkpoint).__name__}")
    return checkpoint


@dataclass(frozen=True)
class LoadReport:
    checkpoint: str
    encoder_key: str
    encoder_parameters: int
    predictor_parameters: int
    encoder_layers: int
    predictor_layers: int
    encoder_embed_dim: int
    predictor_embed_dim: int


class VJEPA21ViTGAdapter(nn.Module):
    """Frozen 2B V-JEPA 2.1 ViT-G encoder.

    The paired predictor is constructed and loaded by :meth:`from_checkpoint`,
    then moved into the trainable JointStudent. Keeping this module separate
    prevents the 2B frozen encoder from being wrapped by DDP.
    """

    def __init__(self, encoder: nn.Module, config: ExperimentConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.level_dim = config.model.encoder_embed_dim
        self.context_tokens = config.context_tokens
        self.future_tokens = config.future_tokens
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @classmethod
    def from_checkpoint(
        cls, config: ExperimentConfig
    ) -> tuple["VJEPA21ViTGAdapter", nn.Module, LoadReport]:
        checkpoint_path = Path(config.model.checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"V-JEPA checkpoint does not exist: {checkpoint_path}. "
                "Checkpoint downloading is intentionally disabled."
            )
        total_frames = config.video.context_frames + config.video.future_frames
        encoder = vit_gigantic_xformers(
            img_size=(config.video.resolution, config.video.resolution),
            patch_size=config.video.patch_size,
            num_frames=total_frames,
            tubelet_size=config.video.tubelet_size,
            use_sdpa=True,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
            modality_embedding=True,
            n_output_distillation=4,
            use_activation_checkpointing=False,
        )
        predictor = vit_predictor(
            img_size=(config.video.resolution, config.video.resolution),
            patch_size=config.video.patch_size,
            use_mask_tokens=True,
            embed_dim=config.model.encoder_embed_dim,
            predictor_embed_dim=config.model.predictor_embed_dim,
            num_frames=total_frames,
            tubelet_size=config.video.tubelet_size,
            depth=config.model.predictor_depth,
            num_heads=config.model.predictor_heads,
            num_mask_tokens=config.model.predictor_mask_tokens,
            zero_init_mask_tokens=True,
            use_rope=True,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            n_output_distillation=4,
            return_all_tokens=True,
            img_temporal_dim_size=1,
            modality_embedding=True,
            interpolate_rope=False,
            use_activation_checkpointing=False,
        )

        checkpoint = _load_torch_checkpoint(checkpoint_path)
        encoder_key = "target_encoder"
        if encoder_key not in checkpoint:
            available = sorted(str(key) for key in checkpoint.keys())
            raise KeyError(
                f"Checkpoint must contain '{encoder_key}'. Available top-level keys: {available}"
            )
        if "predictor" not in checkpoint:
            raise KeyError("Checkpoint does not contain the paired V-JEPA 2.1 predictor")
        encoder_state = _clean_state_dict(checkpoint[encoder_key])
        predictor_state = _clean_state_dict(checkpoint["predictor"])
        cls._validate_architecture(encoder_state, predictor_state, config)
        encoder.load_state_dict(encoder_state, strict=True)
        predictor.load_state_dict(predictor_state, strict=True)
        del checkpoint, encoder_state, predictor_state

        report = LoadReport(
            checkpoint=str(checkpoint_path),
            encoder_key=encoder_key,
            encoder_parameters=sum(p.numel() for p in encoder.parameters()),
            predictor_parameters=sum(p.numel() for p in predictor.parameters()),
            encoder_layers=len(encoder.blocks),
            predictor_layers=len(predictor.predictor_blocks),
            encoder_embed_dim=encoder.embed_dim,
            predictor_embed_dim=predictor.predictor_norm.normalized_shape[0],
        )
        return cls(encoder, config), predictor, report

    @staticmethod
    def _validate_architecture(
        encoder_state: dict[str, torch.Tensor],
        predictor_state: dict[str, torch.Tensor],
        config: ExperimentConfig,
    ) -> None:
        patch_key = "patch_embed.proj.weight"
        if patch_key not in encoder_state:
            raise KeyError(f"Missing encoder tensor: {patch_key}")
        patch_shape = tuple(encoder_state[patch_key].shape)
        expected_patch = (
            config.model.encoder_embed_dim,
            3,
            config.video.tubelet_size,
            config.video.patch_size,
            config.video.patch_size,
        )
        if patch_shape != expected_patch:
            raise ValueError(
                f"Checkpoint is not the expected ViT-G/16 2B encoder: "
                f"{patch_key} has {patch_shape}, expected {expected_patch}"
            )
        last_block = f"blocks.{config.model.encoder_depth - 1}.attn.qkv.weight"
        if last_block not in encoder_state:
            raise ValueError(
                f"Checkpoint does not contain encoder block {config.model.encoder_depth - 1}; "
                "the lowercase ViT-g 1B checkpoint is not compatible"
            )
        last_predictor = f"predictor_blocks.{config.model.predictor_depth - 1}.attn.qkv.weight"
        if last_predictor not in predictor_state:
            raise ValueError("Checkpoint predictor depth does not match the configured 24 blocks")
        projection = predictor_state.get("predictor_proj.weight")
        expected_out = 4 * config.model.encoder_embed_dim
        if projection is None or projection.shape[0] != expected_out:
            actual = None if projection is None else tuple(projection.shape)
            raise ValueError(
                f"Predictor output projection must produce {expected_out} features, got {actual}"
            )

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode_context(self, context_rgb: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        features = self.encoder(context_rgb, training=True)
        expected = (context_rgb.shape[0], self.context_tokens, 4 * self.level_dim)
        if tuple(features.shape) != expected:
            raise RuntimeError(f"Unexpected context feature shape {tuple(features.shape)}; expected {expected}")
        return features

    @torch.no_grad()
    def encode_target(self, full_rgb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.encoder.eval()
        features = self.encoder(full_rgb, training=True)
        future = features[:, self.context_tokens : self.context_tokens + self.future_tokens]
        expected = (full_rgb.shape[0], self.future_tokens, 4 * self.level_dim)
        if tuple(future.shape) != expected:
            raise RuntimeError(f"Unexpected target feature shape {tuple(future.shape)}; expected {expected}")
        return tuple(future.split(self.level_dim, dim=-1))
