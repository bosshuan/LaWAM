from __future__ import annotations

import torch
import torch.nn.functional as F

from latent_wam.vendor.vjepa21.layers import Block, rotate_queries_or_keys


def action_times(chunk_size: int, action_hz: float, device=None) -> torch.Tensor:
    return torch.arange(1, chunk_size + 1, device=device, dtype=torch.float32) / action_hz


def future_interval_ends(
    future_frames: int,
    video_fps: float,
    tubelet_size: int,
    spatial_tokens: int,
    device=None,
) -> torch.Tensor:
    n_tubelets = future_frames // tubelet_size
    ends = torch.arange(1, n_tubelets + 1, device=device, dtype=torch.float32)
    ends = ends * tubelet_size / video_fps
    return ends.repeat_interleave(spatial_tokens)


def build_joint_attention_mask(
    context_tokens: int,
    future_ends: torch.Tensor,
    action_time: torch.Tensor,
    reciprocal: bool,
) -> torch.Tensor:
    """Return an SDPA boolean visibility mask [query, key].

    Context and future follow the native visual predictor. Action queries never
    read raw context tokens. Reciprocal blocks couple an action only with future
    intervals ending at or after that action, and a future interval only with
    actions occurring within that interval's horizon.
    """
    n_future, n_action = future_ends.numel(), action_time.numel()
    n_visual = context_tokens + n_future
    total = n_visual + n_action
    mask = torch.zeros(total, total, dtype=torch.bool, device=future_ends.device)
    mask[:n_visual, :n_visual] = True
    action_rows = slice(n_visual, total)
    action_cols = slice(n_visual, total)
    mask[action_rows, action_cols] = True
    if reciprocal:
        a_to_f = future_ends.unsqueeze(0) >= action_time.unsqueeze(1)
        f_to_a = action_time.unsqueeze(0) <= future_ends.unsqueeze(1)
        mask[n_visual:, context_tokens:n_visual] = a_to_f
        mask[context_tokens:n_visual, n_visual:] = f_to_a
    else:
        mask[n_visual:, context_tokens:n_visual] = True
    return mask


def _apply_mixed_rope(
    block: Block,
    q: torch.Tensor,
    k: torch.Tensor,
    visual_position_ids: torch.Tensor,
    action_position: torch.Tensor,
    spatial_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention = block.attn
    n_visual = visual_position_ids.shape[1]
    q_visual, k_visual = attention.apply_rope(
        q[..., :n_visual, :],
        k[..., :n_visual, :],
        visual_position_ids,
        spatial_size,
        spatial_size,
    )
    q_action, k_action = q[..., n_visual:, :], k[..., n_visual:, :]
    positions = action_position.to(dtype=q.dtype)
    positions = positions.view(1, 1, -1).expand(q.shape[0], q.shape[1], -1)
    temporal_dim = attention.d_dim
    q_temporal = rotate_queries_or_keys(q_action[..., :temporal_dim], positions)
    k_temporal = rotate_queries_or_keys(k_action[..., :temporal_dim], positions)
    q_action = torch.cat([q_temporal, q_action[..., temporal_dim:]], dim=-1)
    k_action = torch.cat([k_temporal, k_action[..., temporal_dim:]], dim=-1)
    return (
        torch.cat([q_visual, q_action], dim=-2),
        torch.cat([k_visual, k_action], dim=-2),
    )


def forward_mixed_block(
    block: Block,
    tokens: torch.Tensor,
    visual_position_ids: torch.Tensor,
    action_position: torch.Tensor,
    visibility: torch.Tensor,
    spatial_size: int,
) -> torch.Tensor:
    """Run an upstream V-JEPA block on visual and action tokens.

    No parameters are added or copied here. The function directly uses the
    strict-loaded block's qkv/projection, normalization, drop-path, and MLP.
    """
    residual = tokens
    x = block.norm1(tokens)
    batch, n_tokens, width = x.shape
    attention = block.attn
    qkv = attention.qkv(x).unflatten(-1, (3, attention.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = _apply_mixed_rope(
        block,
        qkv[0],
        qkv[1],
        visual_position_ids,
        action_position,
        spatial_size,
    )
    attn_mask = visibility.view(1, 1, n_tokens, n_tokens)
    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=attention.proj_drop_prob if block.training else 0.0,
        is_causal=False,
    )
    x = x.transpose(1, 2).reshape(batch, n_tokens, width)
    x = attention.proj_drop(attention.proj(x))
    x = residual + block.drop_path(x)
    return x + block.drop_path(block.mlp(block.norm2(x)))
