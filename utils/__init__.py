"""
AFN-DeSeg Utilities.

This module provides utility functions for:
- Evaluation metrics (PSNR, SSIM, Dice, mAP, HD95, etc.)
- Visualization (training curves, result comparison)
"""

from .metrics import (
    compute_psnr,
    compute_ssim,
    compute_dice,
    compute_iou,
    compute_map_at_iou,
    compute_hausdorff_distance,
    compute_morphology_stats,
    compute_nuclear_area,
    compute_nuclear_circularity,
    compute_nuclear_density,
    compute_bland_altman,
    compute_key_area_fraction,
    MetricTracker,
    evaluate_batch
)

from .visualization import (
    plot_image,
    plot_comparison,
    plot_denoising_result,
    plot_segmentation_result,
    plot_full_result,
    plot_training_history,
    plot_loss_components,
    save_prediction_samples
)

__all__ = [
    # Metrics
    'compute_psnr',
    'compute_ssim',
    'compute_dice',
    'compute_iou',
    'compute_map_at_iou',
    'compute_hausdorff_distance',
    'compute_morphology_stats',
    'compute_nuclear_area',
    'compute_nuclear_circularity',
    'compute_nuclear_density',
    'compute_bland_altman',
    'compute_key_area_fraction',
    'MetricTracker',
    'evaluate_batch',
    # Visualization
    'plot_image',
    'plot_comparison',
    'plot_denoising_result',
    'plot_segmentation_result',
    'plot_full_result',
    'plot_training_history',
    'plot_loss_components',
    'save_prediction_samples'
]
