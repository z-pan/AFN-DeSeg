"""
AFN-DeSeg Inference.

This module provides inference utilities:
- AFNDeSegPredictor: Easy-to-use predictor class for single-channel images
- MultichannelTPAFPredictor: Predictor for two-channel (red/green) TPAF images
- KDAPredictor: Predictor for Key Diagnostic Areas
- predict_directory: Batch prediction on directories
"""

from .predict import (
    AFNDeSegPredictor,
    predict_directory,
    load_image,
    save_image,
    preprocess,
    postprocess
)

from .kda_predictor import (
    KDAPredictor,
    predict_directory as predict_kda_directory,
    load_mask,
    save_prediction
)

from .multichannel_predict import (
    MultichannelTPAFPredictor,
    predict_multichannel_directory,
    load_multichannel_image,
    save_multichannel_image,
    save_instance_mask,
    binary_to_instance_segmentation,
    merge_instance_masks
)

__all__ = [
    # AFN-DeSeg inference (single-channel)
    'AFNDeSegPredictor',
    'predict_directory',
    'load_image',
    'save_image',
    'preprocess',
    'postprocess',
    # Multichannel TPAF inference (two-channel)
    'MultichannelTPAFPredictor',
    'predict_multichannel_directory',
    'load_multichannel_image',
    'save_multichannel_image',
    'save_instance_mask',
    'binary_to_instance_segmentation',
    'merge_instance_masks',
    # KDA inference
    'KDAPredictor',
    'predict_kda_directory',
    'load_mask',
    'save_prediction'
]
