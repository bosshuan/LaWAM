"""Minimal V-JEPA 2.1 model code vendored from facebookresearch/vjepa2.

The files in this directory retain the upstream MIT license. Training, data,
evaluation, and download utilities are intentionally not vendored.
"""

from .predictor import VisionTransformerPredictor, vit_predictor
from .vision_transformer import VisionTransformer, vit_gigantic_xformers

__all__ = [
    "VisionTransformer",
    "VisionTransformerPredictor",
    "vit_gigantic_xformers",
    "vit_predictor",
]
