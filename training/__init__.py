"""
AFN-DeSeg Training Scripts.

This module provides training pipelines:
- Stage 1: DINO domain adaptation (train_stage1_dino.py)
- Stage 2: Joint denoising and segmentation training (train_stage2_joint.py)
- KDA Training: Attention U-Net for Key Diagnostic Area prediction (kda_trainer.py)
"""

from .kda_trainer import (
    KDATrainer,
    KDADataset,
    CombinedLoss,
    DiceLoss
)

__all__ = [
    'train_stage1_dino',
    'train_stage2_joint',
    # KDA training
    'KDATrainer',
    'KDADataset',
    'CombinedLoss',
    'DiceLoss'
]
