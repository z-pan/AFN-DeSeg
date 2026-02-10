"""
Visualization Utilities for AFN-DeSeg.

Provides functions for visualizing:
- Training progress and metrics
- Denoising results
- Segmentation predictions
- Side-by-side comparisons
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import torch


# =============================================================================
# Tensor/Array Conversion
# =============================================================================

def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def prepare_image(
    img: Union[np.ndarray, torch.Tensor],
    squeeze: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Prepare image for visualization.

    Args:
        img: Input image tensor or array.
        squeeze: Whether to squeeze singleton dimensions.
        normalize: Whether to normalize to [0, 1].

    Returns:
        2D numpy array ready for display.
    """
    img = to_numpy(img)

    if squeeze:
        img = img.squeeze()

    # Handle batch dimension
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3:
        if img.shape[0] <= 4:  # Channels first
            img = img[0]
        else:  # Channels last
            img = img[..., 0]

    if normalize:
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)

    return img


# =============================================================================
# Image Visualization
# =============================================================================

def plot_image(
    img: Union[np.ndarray, torch.Tensor],
    title: str = '',
    cmap: str = 'gray',
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None
):
    """
    Plot a single image.

    Args:
        img: Image to display.
        title: Plot title.
        cmap: Colormap.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for visualization")
        return

    img = prepare_image(img)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(
    images: List[Union[np.ndarray, torch.Tensor]],
    titles: List[str],
    cmap: str = 'gray',
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
):
    """
    Plot multiple images side by side for comparison.

    Args:
        images: List of images to display.
        titles: List of titles for each image.
        cmap: Colormap.
        figsize: Figure size (auto if None).
        save_path: Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for visualization")
        return

    n = len(images)
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        img = prepare_image(img)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_denoising_result(
    noisy: Union[np.ndarray, torch.Tensor],
    denoised: Union[np.ndarray, torch.Tensor],
    clean: Optional[Union[np.ndarray, torch.Tensor]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Plot denoising result with comparison.

    Args:
        noisy: Noisy input image.
        denoised: Denoised output image.
        clean: Ground truth clean image (optional).
        figsize: Figure size.
        save_path: Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return

    images = [noisy, denoised]
    titles = ['Noisy Input', 'Denoised']

    if clean is not None:
        images.append(clean)
        titles.append('Ground Truth')

        # Add residual
        denoised_np = prepare_image(denoised)
        clean_np = prepare_image(clean)
        residual = np.abs(denoised_np - clean_np)
        images.append(residual)
        titles.append('Residual')

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    for ax, img, title in zip(axes, images, titles):
        img = prepare_image(img)
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_segmentation_result(
    image: Union[np.ndarray, torch.Tensor],
    pred_mask: Union[np.ndarray, torch.Tensor],
    gt_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Plot segmentation result with overlay.

    Args:
        image: Input image.
        pred_mask: Predicted segmentation mask.
        gt_mask: Ground truth mask (optional).
        alpha: Overlay transparency.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return

    image = prepare_image(image)
    pred_mask = prepare_image(pred_mask, normalize=False)
    pred_binary = (pred_mask > 0.5).astype(np.float32)

    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Prediction overlay
    axes[1].imshow(image, cmap='gray')
    mask_overlay = np.zeros((*image.shape, 4))
    mask_overlay[pred_binary > 0.5] = [1, 0, 0, alpha]  # Red overlay
    axes[1].imshow(mask_overlay)
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Ground truth comparison
    if gt_mask is not None:
        gt_mask = prepare_image(gt_mask, normalize=False)
        gt_binary = (gt_mask > 0.5).astype(np.float32)

        axes[2].imshow(image, cmap='gray')
        # Green for GT, Red for prediction
        overlay = np.zeros((*image.shape, 4))
        overlay[gt_binary > 0.5] = [0, 1, 0, alpha]  # Green for GT
        overlay[pred_binary > 0.5] = [1, 0, 0, alpha]  # Red for pred
        # Yellow for overlap
        overlap = (gt_binary > 0.5) & (pred_binary > 0.5)
        overlay[overlap] = [1, 1, 0, alpha]
        axes[2].imshow(overlay)
        axes[2].set_title('GT (green) vs Pred (red)')
        axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_full_result(
    noisy: Union[np.ndarray, torch.Tensor],
    denoised: Union[np.ndarray, torch.Tensor],
    seg_pred: Union[np.ndarray, torch.Tensor],
    clean: Optional[Union[np.ndarray, torch.Tensor]] = None,
    seg_gt: Optional[Union[np.ndarray, torch.Tensor]] = None,
    figsize: Tuple[int, int] = (20, 8),
    save_path: Optional[str] = None
):
    """
    Plot full AFN-DeSeg result (denoising + segmentation).

    Args:
        noisy: Noisy input image.
        denoised: Denoised output.
        seg_pred: Predicted segmentation.
        clean: Ground truth clean image (optional).
        seg_gt: Ground truth segmentation (optional).
        figsize: Figure size.
        save_path: Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return

    has_gt = clean is not None and seg_gt is not None
    n_cols = 5 if has_gt else 3

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Noisy input
    axes[0].imshow(prepare_image(noisy), cmap='gray')
    axes[0].set_title('Noisy Input')
    axes[0].axis('off')

    # Denoised
    axes[1].imshow(prepare_image(denoised), cmap='gray')
    axes[1].set_title('Denoised')
    axes[1].axis('off')

    # Segmentation prediction
    denoised_np = prepare_image(denoised)
    seg_pred_np = prepare_image(seg_pred, normalize=False)
    axes[2].imshow(denoised_np, cmap='gray')
    mask_overlay = np.zeros((*denoised_np.shape, 4))
    mask_overlay[seg_pred_np > 0.5] = [1, 0, 0, 0.5]
    axes[2].imshow(mask_overlay)
    axes[2].set_title('Segmentation')
    axes[2].axis('off')

    if has_gt:
        # Ground truth clean
        axes[3].imshow(prepare_image(clean), cmap='gray')
        axes[3].set_title('Clean GT')
        axes[3].axis('off')

        # Ground truth segmentation
        clean_np = prepare_image(clean)
        seg_gt_np = prepare_image(seg_gt, normalize=False)
        axes[4].imshow(clean_np, cmap='gray')
        mask_overlay = np.zeros((*clean_np.shape, 4))
        mask_overlay[seg_gt_np > 0.5] = [0, 1, 0, 0.5]
        axes[4].imshow(mask_overlay)
        axes[4].set_title('GT Segmentation')
        axes[4].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Training Visualization
# =============================================================================

def plot_training_history(
    history: List[Dict[str, float]],
    metrics: List[str] = ['train_loss', 'val_loss', 'val_dice', 'val_psnr'],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Plot training history curves.

    Args:
        history: List of metric dictionaries per epoch.
        metrics: List of metric names to plot.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return

    # Filter metrics that exist in history
    available_metrics = [m for m in metrics if m in history[0]]
    n_metrics = len(available_metrics)

    if n_metrics == 0:
        print("No metrics to plot")
        return

    rows = (n_metrics + 1) // 2
    cols = min(2, n_metrics)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    epochs = range(1, len(history) + 1)

    for ax, metric in zip(axes, available_metrics):
        values = [h.get(metric, 0) for h in history]
        ax.plot(epochs, values, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_components(
    history: List[Dict[str, float]],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot individual loss components over training.

    Args:
        history: List of metric dictionaries per epoch.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history) + 1)
    loss_keys = ['train_loss_rec', 'train_loss_seg', 'train_loss_percep']
    colors = ['blue', 'green', 'red']
    labels = ['Reconstruction', 'Segmentation', 'Perceptual']

    for key, color, label in zip(loss_keys, colors, labels):
        if key in history[0]:
            values = [h.get(key, 0) for h in history]
            ax.plot(epochs, values, color=color, label=label, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Utility Functions
# =============================================================================

def create_grid(
    images: List[Union[np.ndarray, torch.Tensor]],
    nrow: int = 4,
    padding: int = 2,
    normalize: bool = True
) -> np.ndarray:
    """
    Create a grid of images.

    Args:
        images: List of images.
        nrow: Number of images per row.
        padding: Padding between images.
        normalize: Whether to normalize images.

    Returns:
        Grid image as numpy array.
    """
    images = [prepare_image(img, normalize=normalize) for img in images]

    n = len(images)
    ncol = (n + nrow - 1) // nrow

    h, w = images[0].shape
    grid_h = h * ncol + padding * (ncol + 1)
    grid_w = w * nrow + padding * (nrow + 1)
    grid = np.ones((grid_h, grid_w)) * 0.5  # Gray background

    for idx, img in enumerate(images):
        i = idx // nrow
        j = idx % nrow
        y = padding + i * (h + padding)
        x = padding + j * (w + padding)
        grid[y:y+h, x:x+w] = img

    return grid


def save_prediction_samples(
    model,
    dataloader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 5
):
    """
    Save sample predictions from model.

    Args:
        model: AFN-DeSeg model.
        dataloader: DataLoader with test samples.
        device: Device for inference.
        output_dir: Directory to save visualizations.
        num_samples: Number of samples to save.
    """
    from models.afn_deseg import AFNDeSeg

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break

            noisy, clean, mask = batch[:3]
            noisy = noisy.to(device)

            # Preprocess and forward
            noisy_rgb = AFNDeSeg.preprocess_tpaf(noisy)
            denoised, seg_pred = model(noisy_rgb)

            # Save each sample in batch
            for i in range(noisy.shape[0]):
                if count >= num_samples:
                    break

                plot_full_result(
                    noisy[i], denoised[i], seg_pred[i],
                    clean[i], mask[i],
                    save_path=str(output_dir / f'sample_{count:03d}.png')
                )
                count += 1
