"""
KDA-Specific Evaluation Metrics.

Implements metrics for evaluating Key Diagnostic Area predictions:
- Key Area Fraction (KAF)
- Nuclear density analysis
- Correlation analysis (Pearson, Bland-Altman)
- Boundary metrics (HD95)
- Pixel-level metrics (Dice, IoU)
"""

import numpy as np
from scipy import stats
from scipy.ndimage import distance_transform_edt, label
from typing import Dict, List, Optional, Tuple, Union


# =============================================================================
# Key Area Fraction (KAF)
# =============================================================================

def calculate_kaf(
    kda_mask: np.ndarray,
    tissue_mask: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Key Area Fraction (KAF).

    KAF represents the proportion of tissue area classified as
    Key Diagnostic Area (high tumor burden region).

    KAF = Area(KDA) / Area(Tissue)

    Args:
        kda_mask: Binary KDA prediction, shape (H, W).
                  Values in {0, 1} where 1 = KDA.
        tissue_mask: Optional tissue mask. If None, entire image
                    is considered tissue.

    Returns:
        KAF value in range [0, 1].

    Example:
        >>> kda = np.zeros((512, 512))
        >>> kda[100:200, 100:200] = 1  # 100x100 KDA region
        >>> kaf = calculate_kaf(kda)
        >>> print(f"KAF: {kaf:.3f}")  # ~0.038
    """
    # Binarize
    kda_binary = (kda_mask > 0.5).astype(float)

    if tissue_mask is not None:
        tissue_binary = (tissue_mask > 0.5).astype(float)
        tissue_area = tissue_binary.sum()
        kda_area = (kda_binary * tissue_binary).sum()
    else:
        tissue_area = kda_mask.size
        kda_area = kda_binary.sum()

    if tissue_area == 0:
        return 0.0

    kaf = kda_area / tissue_area

    return float(kaf)


def calculate_kaf_per_region(
    kda_mask: np.ndarray,
    region_labels: np.ndarray
) -> Dict[int, float]:
    """
    Calculate KAF for each labeled region.

    Args:
        kda_mask: Binary KDA prediction.
        region_labels: Integer labels for different regions.

    Returns:
        Dictionary mapping region ID to KAF value.
    """
    unique_labels = np.unique(region_labels)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    kaf_per_region = {}
    for label_id in unique_labels:
        region_mask = (region_labels == label_id)
        region_kaf = calculate_kaf(kda_mask, region_mask)
        kaf_per_region[int(label_id)] = region_kaf

    return kaf_per_region


# =============================================================================
# Nuclear Density
# =============================================================================

def calculate_nuclear_density(
    nuclear_mask: np.ndarray,
    roi_size: int = 150,
    n_samples: int = 100,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate nuclear density statistics in random ROIs.

    Samples random ROIs and counts nuclei per ROI to estimate
    nuclear density distribution.

    Args:
        nuclear_mask: Binary nuclear segmentation, shape (H, W).
        roi_size: Size of square ROI in pixels (default: 150x150).
        n_samples: Number of ROIs to sample.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with density statistics:
        - mean: Mean nuclei per ROI
        - std: Standard deviation
        - median: Median nuclei per ROI
        - min, max: Range

    Example:
        >>> density = calculate_nuclear_density(nuclear_mask, roi_size=150)
        >>> print(f"Nuclear density: {density['mean']:.1f} nuclei per FOV")
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = nuclear_mask.shape
    densities = []

    # Label connected components (nuclei)
    labeled, num_nuclei = label(nuclear_mask > 0.5)

    for _ in range(n_samples):
        # Random ROI position
        y = np.random.randint(0, max(1, H - roi_size))
        x = np.random.randint(0, max(1, W - roi_size))

        # Extract ROI
        roi = labeled[y:y+roi_size, x:x+roi_size]

        # Count unique nuclei in ROI (exclude background 0)
        unique_labels = np.unique(roi)
        nuclei_count = len(unique_labels[unique_labels > 0])

        densities.append(nuclei_count)

    densities = np.array(densities)

    return {
        'mean': float(np.mean(densities)),
        'std': float(np.std(densities)),
        'median': float(np.median(densities)),
        'min': float(np.min(densities)),
        'max': float(np.max(densities)),
        'total_nuclei': num_nuclei
    }


def calculate_nuclear_density_map(
    nuclear_mask: np.ndarray,
    kernel_size: int = 64,
    stride: int = 32
) -> np.ndarray:
    """
    Generate a nuclear density heatmap.

    Args:
        nuclear_mask: Binary nuclear segmentation.
        kernel_size: Size of counting kernel.
        stride: Stride between kernel positions.

    Returns:
        Density map, shape approximately (H/stride, W/stride).
    """
    H, W = nuclear_mask.shape
    labeled, _ = label(nuclear_mask > 0.5)

    # Output dimensions
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    density_map = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            y, x = i * stride, j * stride
            roi = labeled[y:y+kernel_size, x:x+kernel_size]
            unique = np.unique(roi)
            density_map[i, j] = len(unique[unique > 0])

    return density_map


# =============================================================================
# Correlation Analysis
# =============================================================================

def calculate_pearson_correlation(
    pred_values: np.ndarray,
    gt_values: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Pearson correlation coefficient.

    Args:
        pred_values: Predicted values (1D array).
        gt_values: Ground truth values (1D array).

    Returns:
        Dictionary with correlation coefficient and p-value.
    """
    if len(pred_values) < 2:
        return {'r': 0.0, 'p_value': 1.0}

    r, p = stats.pearsonr(pred_values, gt_values)

    return {
        'r': float(r),
        'p_value': float(p),
        'r_squared': float(r ** 2)
    }


def calculate_bland_altman(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Perform Bland-Altman analysis for method comparison.

    Calculates mean bias and limits of agreement between
    predicted and ground truth values.

    Args:
        pred_values: Predicted values (1D array).
        gt_values: Ground truth values (1D array).
        confidence: Confidence level for limits of agreement.

    Returns:
        Dictionary with:
        - bias: Mean difference (pred - gt)
        - loa_lower: Lower limit of agreement
        - loa_upper: Upper limit of agreement
        - std_diff: Standard deviation of differences

    Reference:
        Bland & Altman, "Statistical methods for assessing agreement
        between two methods of clinical measurement", Lancet 1986
    """
    differences = pred_values - gt_values
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # Z-score for confidence level
    z = stats.norm.ppf((1 + confidence) / 2)

    loa_lower = mean_diff - z * std_diff
    loa_upper = mean_diff + z * std_diff

    return {
        'bias': float(mean_diff),
        'std_diff': float(std_diff),
        'loa_lower': float(loa_lower),
        'loa_upper': float(loa_upper),
        'confidence': confidence
    }


def correlation_analysis(
    pred_kda_list: List[np.ndarray],
    gt_kda_list: List[np.ndarray],
    metric: str = 'kaf'
) -> Dict[str, float]:
    """
    Comprehensive correlation analysis between predicted and GT KDA.

    Args:
        pred_kda_list: List of predicted KDA masks.
        gt_kda_list: List of ground truth KDA masks.
        metric: Metric to correlate ('kaf', 'area', 'density').

    Returns:
        Dictionary with correlation metrics.
    """
    pred_values = []
    gt_values = []

    for pred, gt in zip(pred_kda_list, gt_kda_list):
        if metric == 'kaf':
            pred_values.append(calculate_kaf(pred))
            gt_values.append(calculate_kaf(gt))
        elif metric == 'area':
            pred_values.append((pred > 0.5).sum())
            gt_values.append((gt > 0.5).sum())
        else:
            raise ValueError(f"Unknown metric: {metric}")

    pred_values = np.array(pred_values)
    gt_values = np.array(gt_values)

    # Pearson correlation
    pearson = calculate_pearson_correlation(pred_values, gt_values)

    # Bland-Altman
    bland_altman = calculate_bland_altman(pred_values, gt_values)

    return {
        'pearson_r': pearson['r'],
        'pearson_p': pearson['p_value'],
        'r_squared': pearson['r_squared'],
        'bias': bland_altman['bias'],
        'loa_lower': bland_altman['loa_lower'],
        'loa_upper': bland_altman['loa_upper'],
        'n_samples': len(pred_values)
    }


# =============================================================================
# Boundary Metrics
# =============================================================================

def hausdorff_distance_95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Calculate 95th percentile Hausdorff Distance (HD95).

    Measures the maximum boundary distance at the 95th percentile,
    providing a robust estimate of boundary precision.

    Args:
        pred_mask: Predicted binary mask.
        gt_mask: Ground truth binary mask.

    Returns:
        HD95 distance in pixels. Returns inf if either mask is empty.

    Note:
        HD95 is more robust to outliers than standard Hausdorff distance.
    """
    pred_binary = (pred_mask > 0.5).astype(bool)
    gt_binary = (gt_mask > 0.5).astype(bool)

    # Handle empty masks
    if not pred_binary.any() or not gt_binary.any():
        return float('inf')

    # Distance transforms
    pred_dist = distance_transform_edt(~pred_binary)
    gt_dist = distance_transform_edt(~gt_binary)

    # Distances from pred boundary to gt
    pred_boundary = pred_binary ^ distance_transform_edt(pred_binary) > 1
    dist_pred_to_gt = gt_dist[pred_boundary]

    # Distances from gt boundary to pred
    gt_boundary = gt_binary ^ distance_transform_edt(gt_binary) > 1
    dist_gt_to_pred = pred_dist[gt_boundary]

    # Handle edge cases
    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return float('inf')

    # 95th percentile of combined distances
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    hd95 = np.percentile(all_distances, 95)

    return float(hd95)


def average_surface_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Calculate Average Surface Distance (ASD).

    Args:
        pred_mask: Predicted binary mask.
        gt_mask: Ground truth binary mask.

    Returns:
        Mean surface distance in pixels.
    """
    pred_binary = (pred_mask > 0.5).astype(bool)
    gt_binary = (gt_mask > 0.5).astype(bool)

    if not pred_binary.any() or not gt_binary.any():
        return float('inf')

    pred_dist = distance_transform_edt(~pred_binary)
    gt_dist = distance_transform_edt(~gt_binary)

    # Surface distances
    pred_boundary = pred_binary ^ (distance_transform_edt(pred_binary) > 1)
    gt_boundary = gt_binary ^ (distance_transform_edt(gt_binary) > 1)

    dist_pred_to_gt = gt_dist[pred_boundary]
    dist_gt_to_pred = pred_dist[gt_boundary]

    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return float('inf')

    asd = (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2

    return float(asd)


# =============================================================================
# Pixel-Level Metrics
# =============================================================================

def calculate_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Dice Similarity Coefficient.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)

    Args:
        pred_mask: Predicted binary mask.
        gt_mask: Ground truth binary mask.
        smooth: Smoothing factor for numerical stability.

    Returns:
        Dice coefficient in [0, 1].
    """
    pred_binary = (pred_mask > 0.5).astype(float)
    gt_binary = (gt_mask > 0.5).astype(float)

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return float(dice)


def calculate_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Intersection over Union (Jaccard Index).

    IoU = |X ∩ Y| / |X ∪ Y|

    Args:
        pred_mask: Predicted binary mask.
        gt_mask: Ground truth binary mask.
        smooth: Smoothing factor.

    Returns:
        IoU in [0, 1].
    """
    pred_binary = (pred_mask > 0.5).astype(float)
    gt_binary = (gt_mask > 0.5).astype(float)

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return float(iou)


def calculate_sensitivity(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Calculate sensitivity (true positive rate / recall).

    Sensitivity = TP / (TP + FN)

    Args:
        pred_mask: Predicted binary mask.
        gt_mask: Ground truth binary mask.

    Returns:
        Sensitivity in [0, 1].
    """
    pred_binary = (pred_mask > 0.5)
    gt_binary = (gt_mask > 0.5)

    tp = (pred_binary & gt_binary).sum()
    fn = (~pred_binary & gt_binary).sum()

    if tp + fn == 0:
        return 0.0

    return float(tp / (tp + fn))


def calculate_specificity(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Calculate specificity (true negative rate).

    Specificity = TN / (TN + FP)

    Args:
        pred_mask: Predicted binary mask.
        gt_mask: Ground truth binary mask.

    Returns:
        Specificity in [0, 1].
    """
    pred_binary = (pred_mask > 0.5)
    gt_binary = (gt_mask > 0.5)

    tn = (~pred_binary & ~gt_binary).sum()
    fp = (pred_binary & ~gt_binary).sum()

    if tn + fp == 0:
        return 0.0

    return float(tn / (tn + fp))


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

def evaluate_kda_prediction(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    nuclear_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of KDA prediction.

    Args:
        pred_mask: Predicted KDA mask.
        gt_mask: Ground truth KDA mask.
        nuclear_mask: Optional nuclear segmentation for density analysis.

    Returns:
        Dictionary with all evaluation metrics.
    """
    results = {}

    # Pixel-level metrics
    results['dice'] = calculate_dice(pred_mask, gt_mask)
    results['iou'] = calculate_iou(pred_mask, gt_mask)
    results['sensitivity'] = calculate_sensitivity(pred_mask, gt_mask)
    results['specificity'] = calculate_specificity(pred_mask, gt_mask)

    # Boundary metrics
    results['hd95'] = hausdorff_distance_95(pred_mask, gt_mask)
    results['asd'] = average_surface_distance(pred_mask, gt_mask)

    # KAF
    results['kaf_pred'] = calculate_kaf(pred_mask)
    results['kaf_gt'] = calculate_kaf(gt_mask)
    results['kaf_diff'] = abs(results['kaf_pred'] - results['kaf_gt'])

    # Nuclear density (if mask provided)
    if nuclear_mask is not None:
        density = calculate_nuclear_density(nuclear_mask)
        results['nuclear_density_mean'] = density['mean']
        results['total_nuclei'] = density['total_nuclei']

    return results
