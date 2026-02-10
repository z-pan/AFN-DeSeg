"""
Evaluation Metrics for AFN-DeSeg.

Based on Note S6 from the supplementary information.

Metrics implemented:
- Image Restoration: PSNR, SSIM
- Segmentation: Dice Coefficient, mAP@IoU
- Nuclei Morphology: Nuclear Area, Circularity, Density
- Clinical Validation: KAF, Bland-Altman, HD95
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

try:
    from scipy import ndimage
    from scipy.spatial.distance import directed_hausdorff
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# =============================================================================
# Image Restoration Metrics
# =============================================================================

def compute_psnr(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    max_val: float = 1.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(MAX_I^2 / MSE)

    Args:
        pred: Predicted image.
        target: Ground truth image.
        max_val: Maximum pixel value.

    Returns:
        PSNR value in dB.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)

    if mse < 1e-10:
        return 100.0

    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)


def compute_ssim(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    max_val: float = 1.0
) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Uses local means, variances, and covariances.

    Args:
        pred: Predicted image (H, W) or (B, C, H, W).
        target: Ground truth image.
        window_size: Size of the Gaussian window.
        k1, k2: Stability constants.
        max_val: Maximum pixel value.

    Returns:
        SSIM value.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Flatten to 2D if needed
    pred = pred.squeeze()
    target = target.squeeze()

    if pred.ndim > 2:
        # Average over batch/channel dimensions
        ssim_vals = []
        for i in range(pred.shape[0]):
            ssim_vals.append(compute_ssim(pred[i], target[i], window_size, k1, k2, max_val))
        return float(np.mean(ssim_vals))

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Compute local statistics using uniform filter
    if HAS_SCIPY:
        mu_pred = ndimage.uniform_filter(pred, size=window_size)
        mu_target = ndimage.uniform_filter(target, size=window_size)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = ndimage.uniform_filter(pred ** 2, size=window_size) - mu_pred_sq
        sigma_target_sq = ndimage.uniform_filter(target ** 2, size=window_size) - mu_target_sq
        sigma_pred_target = ndimage.uniform_filter(pred * target, size=window_size) - mu_pred_target

        # SSIM formula
        numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
        denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

        ssim_map = numerator / (denominator + 1e-10)
        return float(np.mean(ssim_map))
    else:
        # Simple global SSIM without scipy
        mu_pred = np.mean(pred)
        mu_target = np.mean(target)
        sigma_pred = np.std(pred)
        sigma_target = np.std(target)
        sigma_pred_target = np.mean((pred - mu_pred) * (target - mu_target))

        numerator = (2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)
        denominator = (mu_pred**2 + mu_target**2 + c1) * (sigma_pred**2 + sigma_target**2 + c2)

        return float(numerator / (denominator + 1e-10))


# =============================================================================
# Segmentation Metrics
# =============================================================================

def compute_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Compute Dice Coefficient.

    DC = (2 * TP) / (2 * TP + FP + FN)
       = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        pred: Predicted probability map.
        target: Ground truth binary mask.
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor.

    Returns:
        Dice coefficient value.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred_binary = (pred > threshold).astype(np.float32).flatten()
    target_binary = target.astype(np.float32).flatten()

    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def compute_iou(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).

    IoU = |X ∩ Y| / |X ∪ Y|

    Args:
        pred: Predicted probability map.
        target: Ground truth binary mask.
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor.

    Returns:
        IoU value.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred_binary = (pred > threshold).astype(np.float32).flatten()
    target_binary = target.astype(np.float32).flatten()

    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def compute_instance_map(
    mask: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, int]:
    """
    Convert binary mask to instance segmentation map.

    Args:
        mask: Binary mask (H, W).
        threshold: Threshold for binarization.

    Returns:
        Tuple of (instance_map, num_instances).
    """
    binary = (mask > threshold).astype(np.uint8)

    if HAS_SCIPY:
        labeled, num_instances = ndimage.label(binary)
    elif HAS_SKIMAGE:
        labeled = measure.label(binary)
        num_instances = labeled.max()
    else:
        # Simple connected component (4-connectivity)
        labeled = np.zeros_like(binary, dtype=np.int32)
        num_instances = 0
        # Placeholder - proper implementation would need flood fill
        labeled = binary.astype(np.int32)
        num_instances = 1 if binary.any() else 0

    return labeled, num_instances


def compute_map_at_iou(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> float:
    """
    Compute mean Average Precision at IoU threshold.

    Instance-level detection metric. A detection is TP if IoU > threshold.

    Args:
        pred_mask: Predicted binary mask.
        target_mask: Ground truth binary mask.
        iou_threshold: IoU threshold for matching.

    Returns:
        mAP value.
    """
    # Get instance maps
    pred_instances, n_pred = compute_instance_map(pred_mask)
    target_instances, n_target = compute_instance_map(target_mask)

    if n_target == 0:
        return 1.0 if n_pred == 0 else 0.0

    if n_pred == 0:
        return 0.0

    # Match predictions to targets
    matched_targets = set()
    tp = 0

    for pred_id in range(1, n_pred + 1):
        pred_region = (pred_instances == pred_id)
        best_iou = 0.0
        best_target = None

        for target_id in range(1, n_target + 1):
            if target_id in matched_targets:
                continue

            target_region = (target_instances == target_id)

            intersection = np.sum(pred_region & target_region)
            union = np.sum(pred_region | target_region)
            iou = intersection / (union + 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_target = target_id

        if best_iou >= iou_threshold and best_target is not None:
            tp += 1
            matched_targets.add(best_target)

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_target if n_target > 0 else 0.0

    # For single threshold, AP ≈ precision (simplified)
    return float(precision)


# =============================================================================
# Nuclei Morphology Metrics
# =============================================================================

def compute_nuclear_area(instance_map: np.ndarray) -> List[float]:
    """
    Compute area of each nucleus instance.

    Args:
        instance_map: Instance segmentation map.

    Returns:
        List of nuclear areas (in pixels).
    """
    areas = []
    for instance_id in range(1, instance_map.max() + 1):
        area = np.sum(instance_map == instance_id)
        if area > 0:
            areas.append(float(area))
    return areas


def compute_nuclear_circularity(instance_map: np.ndarray) -> List[float]:
    """
    Compute circularity of each nucleus.

    Circularity = (4 * π * Area) / Perimeter²
    1.0 = perfect circle

    Args:
        instance_map: Instance segmentation map.

    Returns:
        List of circularity values.
    """
    if not HAS_SKIMAGE:
        return []

    circularities = []

    for instance_id in range(1, instance_map.max() + 1):
        region = (instance_map == instance_id).astype(np.uint8)

        # Find contours
        contours = measure.find_contours(region, 0.5)
        if len(contours) == 0:
            continue

        # Use largest contour
        contour = max(contours, key=len)

        # Compute perimeter
        perimeter = 0.0
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            perimeter += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # Compute area
        area = np.sum(region)

        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            circularities.append(float(min(circularity, 1.0)))

    return circularities


def compute_nuclear_density(instance_map: np.ndarray, fov_area: Optional[float] = None) -> float:
    """
    Compute nuclear density (count per unit area).

    Args:
        instance_map: Instance segmentation map.
        fov_area: Field of view area in pixels. If None, uses image area.

    Returns:
        Nuclear density.
    """
    n_nuclei = instance_map.max()
    if fov_area is None:
        fov_area = instance_map.size
    return float(n_nuclei / fov_area) if fov_area > 0 else 0.0


def compute_morphology_stats(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute all morphological statistics from a mask.

    Args:
        mask: Binary or instance segmentation mask.

    Returns:
        Dictionary with morphology statistics.
    """
    instance_map, n_instances = compute_instance_map(mask)

    areas = compute_nuclear_area(instance_map)
    circularities = compute_nuclear_circularity(instance_map)

    stats = {
        'count': n_instances,
        'density': compute_nuclear_density(instance_map),
    }

    if areas:
        stats['area_mean'] = float(np.mean(areas))
        stats['area_std'] = float(np.std(areas))
        stats['area_median'] = float(np.median(areas))

    if circularities:
        stats['circularity_mean'] = float(np.mean(circularities))
        stats['circularity_std'] = float(np.std(circularities))
        stats['circularity_median'] = float(np.median(circularities))

    return stats


# =============================================================================
# Clinical Validation Metrics
# =============================================================================

def compute_key_area_fraction(
    key_area_mask: np.ndarray,
    tissue_mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Key Area Fraction (KAF).

    KAF = (Area of Key Diagnostic Region) / (Total Tissue Area)

    Args:
        key_area_mask: Binary mask of key diagnostic areas.
        tissue_mask: Binary mask of total tissue area. If None, uses full image.

    Returns:
        KAF value.
    """
    key_area = np.sum(key_area_mask > 0.5)

    if tissue_mask is not None:
        total_area = np.sum(tissue_mask > 0.5)
    else:
        total_area = key_area_mask.size

    return float(key_area / total_area) if total_area > 0 else 0.0


def compute_bland_altman(
    measurements1: np.ndarray,
    measurements2: np.ndarray
) -> Dict[str, float]:
    """
    Perform Bland-Altman analysis.

    Bias = Mean(measurements1 - measurements2)
    LoA = Bias ± 1.96 * SD_diff

    Args:
        measurements1: First set of measurements (e.g., TPAF predictions).
        measurements2: Second set of measurements (e.g., H&E ground truth).

    Returns:
        Dictionary with bias, std, and limits of agreement.
    """
    diff = measurements1 - measurements2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    return {
        'bias': float(mean_diff),
        'std': float(std_diff),
        'loa_lower': float(mean_diff - 1.96 * std_diff),
        'loa_upper': float(mean_diff + 1.96 * std_diff),
    }


def compute_hausdorff_distance(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    percentile: float = 95
) -> float:
    """
    Compute Hausdorff Distance (HD95).

    95th percentile of Hausdorff Distance for boundary precision.

    Args:
        pred_mask: Predicted binary mask.
        target_mask: Ground truth binary mask.
        percentile: Percentile for HD (default 95).

    Returns:
        HD95 value in pixels.
    """
    if not HAS_SCIPY or not HAS_SKIMAGE:
        return float('nan')

    # Get boundaries
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    target_binary = (target_mask > 0.5).astype(np.uint8)

    # Find boundary points
    pred_contours = measure.find_contours(pred_binary, 0.5)
    target_contours = measure.find_contours(target_binary, 0.5)

    if len(pred_contours) == 0 or len(target_contours) == 0:
        return float('nan')

    # Concatenate all contour points
    pred_points = np.vstack(pred_contours)
    target_points = np.vstack(target_contours)

    # Compute directed distances
    distances_pred_to_target = []
    for p in pred_points:
        min_dist = np.min(np.sqrt(np.sum((target_points - p) ** 2, axis=1)))
        distances_pred_to_target.append(min_dist)

    distances_target_to_pred = []
    for t in target_points:
        min_dist = np.min(np.sqrt(np.sum((pred_points - t) ** 2, axis=1)))
        distances_target_to_pred.append(min_dist)

    all_distances = distances_pred_to_target + distances_target_to_pred
    hd = np.percentile(all_distances, percentile)

    return float(hd)


# =============================================================================
# Metric Aggregation
# =============================================================================

class MetricTracker:
    """
    Track and aggregate metrics during training/evaluation.
    """

    def __init__(self):
        self.metrics = {}

    def update(self, metrics_dict: Dict[str, float]):
        """Add metrics from a batch."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def compute(self) -> Dict[str, float]:
        """Compute mean of all tracked metrics."""
        return {key: float(np.mean(values)) for key, values in self.metrics.items()}

    def reset(self):
        """Clear all tracked metrics."""
        self.metrics = {}


def evaluate_batch(
    denoised_pred: torch.Tensor,
    seg_pred: torch.Tensor,
    clean_gt: torch.Tensor,
    seg_gt: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all metrics for a batch.

    Args:
        denoised_pred: Predicted denoised image.
        seg_pred: Predicted segmentation mask.
        clean_gt: Ground truth clean image.
        seg_gt: Ground truth segmentation mask.

    Returns:
        Dictionary of metric values.
    """
    metrics = {}

    # Image restoration metrics
    metrics['psnr'] = compute_psnr(denoised_pred, clean_gt)
    metrics['ssim'] = compute_ssim(denoised_pred, clean_gt)

    # Segmentation metrics
    metrics['dice'] = compute_dice(seg_pred, seg_gt)
    metrics['iou'] = compute_iou(seg_pred, seg_gt)

    return metrics
