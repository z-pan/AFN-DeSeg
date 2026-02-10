"""
AFN-DeSeg Inference.

This module provides inference utilities:
- AFNDeSegPredictor: Easy-to-use predictor class
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

__all__ = [
    'AFNDeSegPredictor',
    'predict_directory',
    'load_image',
    'save_image',
    'preprocess',
    'postprocess'
]
