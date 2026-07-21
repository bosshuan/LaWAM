# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in this directory.

import torch


def apply_masks(x, masks, concat=True):
    all_x = []
    for mask in masks:
        keep = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)
