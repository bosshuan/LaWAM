# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in this directory.

import math
from functools import partial

import torch
import torch.nn as nn

from .layers import Block
from .masks import apply_masks
from .patch_embed import PatchEmbed, PatchEmbed3D
from .tensor_utils import trunc_normal_


class VisionTransformer(nn.Module):
    """V-JEPA 2.1 vision encoder with upstream-compatible parameter names."""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        is_causal=False,
        use_rope=False,
        init_type="default",
        handle_nonsquare_inputs=True,
        img_temporal_dim_size=None,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        n_output_distillation=4,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.init_type = init_type
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        self.use_activation_checkpointing = use_activation_checkpointing

        if self.is_video:
            self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, in_chans, embed_dim)
        else:
            self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.num_patches = (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_embed_img = (
            PatchEmbed3D(patch_size, 1, in_chans, embed_dim)
            if img_temporal_dim_size is not None
            else None
        )
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
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
        self.attn_out = False
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        hierarchical = {
            12: [2, 5, 8, 11],
            24: [5, 11, 17, 23],
            40: [9, 19, 29, 39],
            48: [11, 23, 37, 47],
        }
        if depth not in hierarchical:
            raise ValueError(f"Unsupported hierarchical encoder depth: {depth}")
        self.hierarchical_layers = hierarchical[depth]
        self.out_layers_distillation = (
            self.hierarchical_layers
            if n_output_distillation == 4
            else [self.hierarchical_layers[-1]]
        )
        self.norms_block = nn.ModuleList(
            [norm_layer(embed_dim) for _ in self.hierarchical_layers]
        )
        self.cls_token = None
        self.return_hierarchical = False
        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            if self.init_type == "default":
                trunc_normal_(module.weight, std=self.init_std)
            elif self.init_type == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif self.init_type == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            else:
                raise ValueError(f"Unknown init_type: {self.init_type}")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _rescale_blocks(self):
        for layer_id, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def check_temporal_dim(self, shape):
        return self.img_temporal_dim_size is not None and shape[2] == self.img_temporal_dim_size

    def forward(self, x, masks=None, training=False):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]
        if x.ndim == 4:
            _, _, height, width = x.shape
            time = 1
        else:
            _, _, raw_time, height, width = x.shape
            time = raw_time if self.check_temporal_dim(x.shape) else raw_time // self.tubelet_size
        h_patches, w_patches = height // self.patch_size, width // self.patch_size

        if self.check_temporal_dim(x.shape):
            if self.patch_embed_img is None:
                raise RuntimeError("Image patch embed is not initialized")
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x = x + self.img_mod_embed
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x = x + self.video_mod_embed

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        outs, hierarchical = [], []
        for index, block in enumerate(self.blocks):
            args = (x, masks)
            kwargs = dict(
                T=time,
                H_patches=h_patches,
                W_patches=w_patches,
                return_attn=self.attn_out,
                mode=mode,
            )
            if self.use_activation_checkpointing and self.training:
                x, attn = torch.utils.checkpoint.checkpoint(
                    block, *args, **kwargs, use_reentrant=False
                )
            else:
                x, attn = block(*args, **kwargs)
            if self.out_layers is not None and index in self.out_layers:
                out_index = self.hierarchical_layers.index(index)
                outs.append(self.norms_block[out_index](x))
            if index in self.out_layers_distillation:
                out_index = self.hierarchical_layers.index(index)
                hierarchical.append(self.norms_block[out_index](x))
        if self.out_layers is not None:
            return outs
        if training or self.return_hierarchical:
            return torch.cat(hierarchical, dim=2)
        return self.norms_block[-1](x)


def vit_gigantic_xformers(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1664,
        depth=48,
        num_heads=26,
        mlp_ratio=64 / 13,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
