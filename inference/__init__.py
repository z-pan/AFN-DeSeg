"""
AFN-DeSeg Inference.

This module provides inference utilities:
- AFNDeSegPredictor: Easy-to-use predictor class
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

__all__ = [
    # AFN-DeSeg inference
    'AFNDeSegPredictor',
    'predict_directory',
    'load_image',
    'save_image',
    'preprocess',
    'postprocess',
    # KDA inference
    'KDAPredictor',
    'predict_kda_directory',
    'load_mask',
    'save_prediction'
]
