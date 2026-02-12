"""
AFN-DeSeg Utilities.

This module provides utility functions for:
- Evaluation metrics (PSNR, SSIM, Dice, mAP, HD95, etc.)
- KDA metrics (Key Area Fraction, nuclear density, correlation)
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

from .kda_metrics import (
    calculate_kaf,
    calculate_kaf_per_region,
    calculate_nuclear_density as calculate_kda_nuclear_density,
    calculate_nuclear_density_map,
    calculate_pearson_correlation,
    calculate_bland_altman as calculate_kda_bland_altman,
    correlation_analysis,
    hausdorff_distance_95,
    average_surface_distance,
    calculate_dice as calculate_kda_dice,
    calculate_iou as calculate_kda_iou,
    calculate_sensitivity,
    calculate_specificity,
    evaluate_kda_prediction
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
    # KDA Metrics
    'calculate_kaf',
    'calculate_kaf_per_region',
    'calculate_kda_nuclear_density',
    'calculate_nuclear_density_map',
    'calculate_pearson_correlation',
    'calculate_kda_bland_altman',
    'correlation_analysis',
    'hausdorff_distance_95',
    'average_surface_distance',
    'calculate_kda_dice',
    'calculate_kda_iou',
    'calculate_sensitivity',
    'calculate_specificity',
    'evaluate_kda_prediction',
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
