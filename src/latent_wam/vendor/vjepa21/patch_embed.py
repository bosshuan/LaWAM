# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in this directory.

from einops import rearrange
import torch.nn as nn


class AudioPatchEmbed(nn.Module):
    def __init__(self, freq_bands=128, tubelet_size=2, embed_dim=768):
        super().__init__()
        self.freq_bands = freq_bands
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(freq_bands, tubelet_size),
            stride=(freq_bands, tubelet_size),
        )

    def forward(self, x):
        return self.proj(rearrange(x, "b t c f -> b c f t")).flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        return self.proj(x).flatten(2).transpose(1, 2)
