"""
AFN-DeSeg Model Architectures.

This module provides the main model components:
- AFNDeSeg: Main dual-encoder model for joint denoising and segmentation
- DINOv3Encoder: Vision Transformer encoder with LoRA adaptation
- UNetEncoder: CNN encoder for local feature extraction
"""

from .afn_deseg import (
    AFNDeSeg,
    create_afn_deseg,
    DINOv3Encoder,
    UNetEncoder,
    FusionBlock,
    DenoisingDecoder,
    SegmentationDecoder,
    LoRALinear
)

__all__ = [
    'AFNDeSeg',
    'create_afn_deseg',
    'DINOv3Encoder',
    'UNetEncoder',
    'FusionBlock',
    'DenoisingDecoder',
    'SegmentationDecoder',
    'LoRALinear'
]
