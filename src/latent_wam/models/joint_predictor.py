from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from latent_wam.config import ExperimentConfig
from latent_wam.models.conditioning import ConditioningEncoder, GatedConditioningAdapter
from latent_wam.models.joint_attention import (
    action_times,
    build_joint_attention_mask,
    forward_mixed_block,
    future_interval_ends,
)
from latent_wam.types import JointPrediction, StudentInputs


class JointStudent(nn.Module):
    """One V-JEPA predictor that jointly emits semantic future and actions."""

    def __init__(
        self,
        predictor: nn.Module,
        config: ExperimentConfig,
        text_input_dim: int,
    ):
        super().__init__()
        self.predictor = predictor
        self.config = config
        width = config.model.predictor_embed_dim
        if predictor.mask_tokens is None or len(predictor.mask_tokens) == 0:
            raise ValueError("The paired predictor must contain pretrained mask tokens")
        initial_mask = torch.stack([token.detach() for token in predictor.mask_tokens]).mean(dim=0)
        self.future_mask_token = nn.Parameter(initial_mask.clone())
        for token in self.predictor.mask_tokens:
            token.requires_grad_(False)

        self.conditioning = ConditioningEncoder(config, text_input_dim)
        self.conditioning_adapters = nn.ModuleDict(
            {
                str(index): GatedConditioningAdapter(width, config.model.predictor_heads)
                for index in config.model.conditioning_blocks
            }
        )
        self.action_queries = nn.Parameter(torch.empty(1, config.action.chunk_size, width))
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        self.action_time_embedding = nn.Sequential(
            nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.action_condition_projection = nn.Linear(width, width)
        self.action_norm = nn.LayerNorm(width)
        self.action_head = nn.Linear(width, config.action.max_action_dim)

        spatial = (config.video.resolution // config.video.patch_size) ** 2
        future_ends = future_interval_ends(
            config.video.future_frames,
            config.video.video_fps,
            config.video.tubelet_size,
            spatial,
        )
        times = action_times(config.action.chunk_size, config.action.action_hz)
        context_tubelets = config.video.context_frames // config.video.tubelet_size
        seconds_per_tubelet = config.video.tubelet_size / config.video.video_fps
        rope_times = (context_tubelets - 1) + times / seconds_per_tubelet
        self.register_buffer("future_ends", future_ends, persistent=False)
        self.register_buffer("action_time", times, persistent=False)
        self.register_buffer("action_rope_position", rope_times, persistent=False)

    @property
    def pretrained_parameter_prefix(self) -> str:
        return "predictor."

    def _standard_block(self, block, x, positions):
        if self.config.model.activation_checkpointing and self.training:
            return checkpoint(
                lambda hidden, ids: block(hidden, mask=ids)[0],
                x,
                positions,
                use_reentrant=False,
            )
        return block(x, mask=positions)[0]

    def _mixed_block(self, block, x, positions, visibility):
        spatial_size = self.config.video.resolution // self.config.video.patch_size
        if self.config.model.activation_checkpointing and self.training:
            return checkpoint(
                lambda hidden: forward_mixed_block(
                    block,
                    hidden,
                    positions,
                    self.action_rope_position,
                    visibility,
                    spatial_size,
                ),
                x,
                use_reentrant=False,
            )
        return forward_mixed_block(
            block,
            x,
            positions,
            self.action_rope_position,
            visibility,
            spatial_size,
        )

    def _condition(self, block_index, x, memory, memory_valid):
        adapter = self.conditioning_adapters[str(block_index)]
        n_context = self.config.context_tokens
        updated = adapter(x[:, n_context:], memory, memory_valid)
        return torch.cat([x[:, :n_context], updated], dim=1)

    def forward(
        self,
        context_features: torch.Tensor,
        inputs: StudentInputs,
        text_features: torch.Tensor,
        text_valid: torch.Tensor,
    ) -> JointPrediction:
        config = self.config
        batch = context_features.shape[0]
        expected = (batch, config.context_tokens, 4 * config.model.encoder_embed_dim)
        if tuple(context_features.shape) != expected:
            raise ValueError(f"context_features has {tuple(context_features.shape)}, expected {expected}")
        memory, memory_valid = self.conditioning(
            inputs, text_features.to(context_features.dtype), text_valid
        )
        context = self.predictor.predictor_embed(context_features)
        future = self.future_mask_token.to(context.dtype).expand(
            batch, config.future_tokens, -1
        )
        x = torch.cat([context, future], dim=1)
        if self.predictor.modality_embedding:
            x = x + self.predictor.video_mod_embed.to(x.dtype)
        n_visual = config.context_tokens + config.future_tokens
        positions = torch.arange(n_visual, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)

        for index in range(config.model.joint_start_block):
            x = self._standard_block(self.predictor.predictor_blocks[index], x, positions)
            if index in config.model.conditioning_blocks:
                x = self._condition(index, x, memory, memory_valid)

        if config.train.stage == "future":
            for index in range(config.model.joint_start_block, config.model.predictor_depth):
                x = self._standard_block(
                    self.predictor.predictor_blocks[index], x, positions
                )
                if index in config.model.conditioning_blocks:
                    x = self._condition(index, x, memory, memory_valid)
            x = self.predictor.predictor_norm(x)
            future_hidden = x[:, config.context_tokens:n_visual]
            future_concat = self.predictor.predictor_proj(future_hidden)
            levels = tuple(
                future_concat.split(config.model.encoder_embed_dim, dim=-1)
            )
            empty_actions = x.new_zeros(
                batch, config.action.chunk_size, config.action.max_action_dim
            )
            empty_hidden = x.new_zeros(
                batch, config.action.chunk_size, config.model.predictor_embed_dim
            )
            return JointPrediction(
                future_levels=levels,
                future_concat=future_concat,
                actions=empty_actions,
                action_logits=empty_actions,
                action_hidden=empty_hidden,
            )

        pooled_condition = (memory * memory_valid.unsqueeze(-1)).sum(dim=1)
        pooled_condition = pooled_condition / memory_valid.sum(dim=1, keepdim=True).clamp_min(1)
        time_embedding = self.action_time_embedding(self.action_time.view(1, -1, 1).to(x.dtype))
        actions = self.action_queries.to(x.dtype).expand(batch, -1, -1)
        actions = actions + time_embedding + self.action_condition_projection(pooled_condition).unsqueeze(1)
        x = torch.cat([x, actions], dim=1)

        one_way_mask = build_joint_attention_mask(
            config.context_tokens, self.future_ends, self.action_time, reciprocal=False
        )
        reciprocal_mask = None
        full_mask = None
        if config.model.joint_attention == "time_aligned":
            reciprocal_mask = build_joint_attention_mask(
                config.context_tokens,
                self.future_ends,
                self.action_time,
                reciprocal=True,
            )
        elif config.model.joint_attention == "full":
            full_mask = one_way_mask.clone()
            full_mask[config.context_tokens:n_visual, n_visual:] = True
        for index in range(config.model.joint_start_block, config.model.predictor_depth):
            if config.model.joint_attention == "one_way":
                visibility = one_way_mask
            elif config.model.joint_attention == "full":
                assert full_mask is not None
                visibility = full_mask
            else:
                assert reciprocal_mask is not None
                visibility = (
                    one_way_mask
                    if index < config.model.reciprocal_start_block
                    else reciprocal_mask
                )
            x = self._mixed_block(
                self.predictor.predictor_blocks[index], x, positions, visibility
            )
            if index in config.model.conditioning_blocks:
                x = self._condition(index, x, memory, memory_valid)

        x = self.predictor.predictor_norm(x)
        future_hidden = x[:, config.context_tokens:n_visual]
        action_hidden = self.action_norm(x[:, n_visual:])
        future_concat = self.predictor.predictor_proj(future_hidden)
        levels = tuple(future_concat.split(config.model.encoder_embed_dim, dim=-1))
        if len(levels) != 4:
            raise RuntimeError(f"Expected four future levels, got {len(levels)}")
        action_logits = self.action_head(action_hidden)
        return JointPrediction(
            future_levels=levels,
            future_concat=future_concat,
            actions=action_logits,
            action_logits=action_logits,
            action_hidden=action_hidden,
        )
