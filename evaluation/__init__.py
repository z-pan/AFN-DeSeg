"""
AFN-DeSeg Evaluation.

This module provides evaluation utilities:
- AFNDeSegEvaluator: Comprehensive model evaluation
- Metric computation and reporting
"""

from .evaluate import (
    AFNDeSegEvaluator,
    print_results
)

__all__ = [
    'AFNDeSegEvaluator',
    'print_results'
]
