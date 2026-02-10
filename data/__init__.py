"""
AFN-DeSeg Data Processing.

This module provides data loading and preprocessing:
- TPAFDataset: Dataset for supervised training
- TPAFUnlabeledDataset: Dataset for self-supervised learning
- SyntheticNoiseDataset: Dataset with on-the-fly noise synthesis
- MPGNoiseModel: Physics-informed noise model
- Augmentation utilities
"""

from .dataset import (
    TPAFDataset,
    TPAFUnlabeledDataset,
    SyntheticNoiseDataset,
    create_dataloaders,
    load_image,
    normalize_image
)

from .mpg_noise_synthesis import (
    MPGNoiseModel,
    MPGNoiseSynthesizer,
    add_mpg_noise,
    add_mpg_noise_numpy,
    add_mpg_noise_torch
)

from .augmentations import (
    get_train_transform,
    get_val_transform,
    get_dino_transform,
    SimpleAugmentation
)

__all__ = [
    # Dataset classes
    'TPAFDataset',
    'TPAFUnlabeledDataset',
    'SyntheticNoiseDataset',
    'create_dataloaders',
    'load_image',
    'normalize_image',
    # Noise synthesis
    'MPGNoiseModel',
    'MPGNoiseSynthesizer',
    'add_mpg_noise',
    'add_mpg_noise_numpy',
    'add_mpg_noise_torch',
    # Augmentations
    'get_train_transform',
    'get_val_transform',
    'get_dino_transform',
    'SimpleAugmentation'
]
