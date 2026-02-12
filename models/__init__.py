"""
AFN-DeSeg Model Architectures.

This module provides the main model components:
- AFNDeSeg: Main dual-encoder model for joint denoising and segmentation
- DINOv3Encoder: Vision Transformer encoder with LoRA adaptation
- UNetEncoder: CNN encoder for local feature extraction
- AttentionUNet: Attention U-Net for KDA prediction
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

from .attention_gate import (
    AttentionGate,
    MultiScaleAttentionGate
)

from .attention_unet import (
    AttentionUNet,
    create_attention_unet,
    ConvBlock,
    EncoderBlock,
    DecoderBlock,
    KDAAttentionUNet
)

__all__ = [
    # AFN-DeSeg
    'AFNDeSeg',
    'create_afn_deseg',
    'DINOv3Encoder',
    'UNetEncoder',
    'FusionBlock',
    'DenoisingDecoder',
    'SegmentationDecoder',
    'LoRALinear',
    # Attention U-Net
    'AttentionUNet',
    'create_attention_unet',
    'AttentionGate',
    'MultiScaleAttentionGate',
    'ConvBlock',
    'EncoderBlock',
    'DecoderBlock',
    'KDAAttentionUNet'
]
