"""
AFN-DeSeg Training Scripts.

This module provides training pipelines:
- Stage 1: DINO domain adaptation (train_stage1_dino.py)
- Stage 2: Joint denoising and segmentation training (train_stage2_joint.py)
"""

__all__ = [
    'train_stage1_dino',
    'train_stage2_joint'
]
