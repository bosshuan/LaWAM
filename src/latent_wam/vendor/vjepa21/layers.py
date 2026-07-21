# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in this directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path


def rotate_queries_or_keys(x, pos, n_registers=0, has_cls_first=False):
    _, _, n_tokens, dim = x.size()
    if dim % 2:
        raise AssertionError("Rotary embedding dimension must be even")
    n_cls = 1 if has_cls_first else 0
    end_ctx = n_tokens - n_registers
    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., n_cls:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers else None
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega = 1.0 / (10000 ** (omega / (dim / 2.0)))
    freq = torch.einsum("..., f -> ... f", pos, omega)
    sin = freq.sin().repeat_interleave(2, dim=-1)
    cos = freq.cos().repeat_interleave(2, dim=-1)
    y1, y2 = x_ctx.unflatten(-1, (-1, 2)).unbind(dim=-1)
    rotated = torch.stack((-y2, y1), dim=-1).flatten(-2)
    parts = []
    if x_cls is not None:
        parts.append(x_cls)
    parts.append(x_ctx * cos + rotated * sin)
    if x_reg is not None:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        width = int(2 * hidden_features / 3) if wide_silu else hidden_features
        width = (width + 7) // 8 * 8
        self.fc1 = nn.Linear(in_features, width)
        self.fc2 = nn.Linear(in_features, width)
        self.act = act_layer()
        self.fc3 = nn.Linear(width, out_features)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.pretrained_patch_size = patch_size
        self.pretrained_grid_size = int((252 if patch_size == 14 else 256) / patch_size)

    def _get_frame_pos(self, ids, height=None, width=None):
        tokens_per_frame = int((height or self.grid_size) * (width or self.grid_size))
        return ids // tokens_per_frame

    def separate_positions(self, ids, height=None, width=None):
        height = height or self.grid_size
        width = width or self.grid_size
        tokens_per_frame = int(height * width)
        frames = ids // tokens_per_frame
        within = ids - tokens_per_frame * frames
        rows = within // width
        cols = within - width * rows
        return frames.float(), rows.float(), cols.float()

    def apply_rope(self, q, k, ids, height=None, width=None):
        d_pos, h_pos, w_pos = self.separate_positions(ids, height, width)
        if d_pos.ndim == 2:
            d_pos = d_pos.unsqueeze(1).expand(-1, self.num_heads, -1)
            h_pos = h_pos.unsqueeze(1).expand(-1, self.num_heads, -1)
            w_pos = w_pos.unsqueeze(1).expand(-1, self.num_heads, -1)
        if self.interpolate_rope:
            height = height or self.grid_size
            width = width or self.grid_size
            h_pos = h_pos * (self.pretrained_grid_size - 1) / max(height - 1, 1)
            w_pos = w_pos * (self.pretrained_grid_size - 1) / max(width - 1, 1)
        s = 0
        q_parts, k_parts = [], []
        for size, pos in ((self.d_dim, d_pos), (self.h_dim, h_pos), (self.w_dim, w_pos)):
            q_parts.append(rotate_queries_or_keys(q[..., s : s + size], pos))
            k_parts.append(rotate_queries_or_keys(k[..., s : s + size], pos))
            s += size
        if s < self.head_dim:
            q_parts.append(q[..., s:])
            k_parts.append(k[..., s:])
        return torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1)

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
        batch, n_tokens, dim = x.shape
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if mask is None:
            mask = torch.arange(n_tokens - self.n_registers, device=x.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch, -1)
        q, k = self.apply_rope(q, k, mask, H_patches, W_patches)
        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.proj_drop_prob if self.training else 0.0,
                is_causal=self.is_causal,
            )
            attn = None
        else:
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(batch, n_tokens, dim)
        x = self.proj_drop(self.proj(x))
        return x, attn if return_attn else None


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x):
        batch, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, n_tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob if self.training else 0.0, is_causal=self.is_causal)
        else:
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            x = self.attn_drop(attn) @ v
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, n_tokens, dim)))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_rope = use_rope
        attention_cls = RoPEAttention if use_rope else Attention
        attention_kwargs = dict(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            proj_drop=drop,
        )
        if use_rope:
            attention_kwargs.update(
                grid_size=grid_size,
                n_registers=n_registers,
                has_cls_first=has_cls_first,
                interpolate_rope=interpolate_rope,
                patch_size=patch_size,
            )
        self.attn = attention_cls(**attention_kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = (
            SwiGLUFFN(dim, hidden, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
            if act_layer is nn.SiLU
            else MLP(dim, hidden, act_layer=act_layer, drop=drop)
        )

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False, mode="video"):
        if self.use_rope:
            y, attn = self.attn(self.norm1(x), mask, T, H_patches, W_patches, return_attn)
        else:
            y, attn = self.attn(self.norm1(x)), None
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn if return_attn else None
