# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in this directory.

import math
from functools import partial

import torch
import torch.nn as nn

from .layers import Block
from .masks import apply_masks
from .tensor_utils import repeat_interleave_batch, trunc_normal_


class VisionTransformerPredictor(nn.Module):
    """V-JEPA 2.1 predictor with upstream-compatible parameter names."""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        is_causal=False,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        img_temporal_dim_size=None,
        teacher_embed_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.chop_last_n_tokens = chop_last_n_tokens
        self.has_cls_first = has_cls_first
        hierarchical = {
            4: [0, 1, 2, 3],
            8: [1, 3, 5, 7],
            12: [2, 5, 8, 11],
            20: [4, 9, 14, 19],
            24: [4, 11, 17, 23],
            40: [9, 19, 29, 39],
        }
        if depth not in hierarchical:
            raise ValueError(f"Unsupported predictor depth: {depth}")
        all_layers = hierarchical[depth]
        n_output = kwargs.get("n_output_distillation", len(all_layers))
        self.hierarchical_layers = all_layers[-n_output:]
        act = nn.SiLU if use_silu else nn.GELU
        if len(self.hierarchical_layers) == 1:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        else:
            self.predictor_embed = nn.Sequential(
                nn.Linear(embed_dim * len(self.hierarchical_layers), embed_dim, bias=True),
                act(),
                nn.Linear(embed_dim, predictor_embed_dim, bias=True),
            )
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for _ in range(num_mask_tokens)]
            )
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.grid_depth = num_frames // tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing
        self.num_patches = self.grid_depth * self.grid_height * self.grid_width
        self.modality_embedding = False
        if img_temporal_dim_size is not None and modality_embedding:
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            self.modality_embedding = True
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    is_causal=is_causal,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    n_registers=n_registers,
                    has_cls_first=has_cls_first,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )
        if out_embed_dim is None:
            out_embed_dim = (
                teacher_embed_dim // len(self.hierarchical_layers)
                if teacher_embed_dim is not None
                else embed_dim
            )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim,
            len(self.hierarchical_layers) * out_embed_dim,
            bias=True,
        )
        if self.return_all_tokens:
            self.predictor_proj_context = nn.Linear(
                predictor_embed_dim,
                out_embed_dim * len(self.hierarchical_layers),
                bias=True,
            )
        self.init_std = init_std
        if not zero_init_mask_tokens and self.mask_tokens is not None:
            for token in self.mask_tokens:
                trunc_normal_(token, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, block in enumerate(self.predictor_blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def forward(self, x, masks_x, masks_y, mod="video", mask_index=1):
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]
        batch = len(x) // len(masks_x)
        x = self.predictor_embed(x)
        n_context = x.shape[1]
        if self.mask_tokens is None:
            raise RuntimeError("Predictor was constructed without mask tokens")
        token = self.mask_tokens[mask_index % self.num_mask_tokens]
        pred_tokens = apply_masks(token.repeat(batch, self.num_patches, 1), masks_y)
        x = torch.cat([x.repeat(len(masks_x), 1, 1), pred_tokens], dim=1)
        masks_x_cat, masks_y_cat = torch.cat(masks_x), torch.cat(masks_y)
        positions = torch.cat([masks_x_cat, masks_y_cat], dim=1)
        order = torch.argsort(positions, dim=1)
        positions = torch.stack([positions[i, row] for i, row in enumerate(order)])
        x = torch.stack([x[i, row] for i, row in enumerate(order)])
        if self.modality_embedding:
            x = x + (self.img_mod_embed if mod == "image" else self.video_mod_embed)
        for block in self.predictor_blocks:
            x, _ = block(x, mask=positions)
        x = self.predictor_norm(x)
        reverse = torch.argsort(order, dim=1)
        x = torch.stack([x[i, row] for i, row in enumerate(reverse)])
        x_pred, x_context = x[:, n_context:], x[:, :n_context]
        return self.predictor_proj(x_pred), (
            self.predictor_proj_context(x_context) if self.return_all_tokens else None
        )


def vit_predictor(**kwargs):
    return VisionTransformerPredictor(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
