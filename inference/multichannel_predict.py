"""
Multichannel TPAF Inference Pipeline for AFN-DeSeg.

Handles two-channel (red and green) TPAF images:
1. Each channel is processed separately through AFN-DeSeg
2. Outputs combined denoised images and instance segmentation masks
3. Instance segmentation using connected components with watershed refinement
4. Integration with KDA predictor for clinical analysis

Usage:
    python inference/multichannel_predict.py \
        --checkpoint /path/to/best_model.pth \
        --input /path/to/images \
        --output /path/to/output \
        --run_kda --kda_checkpoint /path/to/kda_model.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.afn_deseg import AFNDeSeg

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    from skimage import io as skimage_io
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Image I/O for Multichannel Images
# =============================================================================

def load_multichannel_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load multichannel TPAF image from file.

    Supports:
    - Two-channel images: (H, W, 2) or (2, H, W)
    - RGB images: Extracts red and green channels
    - Single-channel: Duplicates to two channels

    Args:
        path: Path to image file (.tif, .tiff, .png, .npy).

    Returns:
        Two-channel image as (H, W, 2) array, float32.
    """
    path = Path(path)

    if path.suffix == '.npy':
        img = np.load(path).astype(np.float32)
    elif path.suffix in ['.tif', '.tiff'] and HAS_TIFFFILE:
        img = tifffile.imread(str(path)).astype(np.float32)
    elif HAS_PIL:
        img = np.array(Image.open(path)).astype(np.float32)
    elif HAS_SKIMAGE:
        img = skimage_io.imread(str(path)).astype(np.float32)
    else:
        raise ImportError("Install PIL, scikit-image, or tifffile to load images")

    # Handle different input formats
    if img.ndim == 2:
        # Single channel: duplicate to two channels
        img = np.stack([img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[0] in [2, 3, 4] and img.shape[0] < img.shape[1]:
            # Channel-first format (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))

        if img.shape[-1] >= 3:
            # RGB or RGBA: extract red and green channels
            img = img[..., :2]
        elif img.shape[-1] == 1:
            # Single channel: duplicate
            img = np.concatenate([img, img], axis=-1)

    # Ensure (H, W, 2) format
    if img.shape[-1] != 2:
        raise ValueError(f"Expected 2 channels, got {img.shape[-1]}")

    return img


def save_multichannel_image(
    img: np.ndarray,
    path: Union[str, Path],
    normalize: bool = True
):
    """
    Save multichannel image to file.

    Args:
        img: Two-channel image, shape (H, W, 2).
        path: Output path.
        normalize: Whether to normalize to [0, 1] range.
    """
    path = Path(path)

    if normalize:
        for c in range(img.shape[-1]):
            c_min, c_max = img[..., c].min(), img[..., c].max()
            if c_max - c_min > 1e-8:
                img[..., c] = (img[..., c] - c_min) / (c_max - c_min)

    if path.suffix == '.npy':
        np.save(path, img)
    elif path.suffix in ['.tif', '.tiff'] and HAS_TIFFFILE:
        # Save as multichannel TIFF
        tifffile.imwrite(str(path), img.astype(np.float32), photometric='minisblack')
    elif HAS_PIL:
        # Convert to RGB for PNG (red, green, black)
        h, w, _ = img.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = (img[..., 0] * 255).clip(0, 255).astype(np.uint8)  # Red
        rgb[..., 1] = (img[..., 1] * 255).clip(0, 255).astype(np.uint8)  # Green
        Image.fromarray(rgb).save(path)
    else:
        np.save(path.with_suffix('.npy'), img)


def save_instance_mask(
    mask: np.ndarray,
    path: Union[str, Path]
):
    """
    Save instance segmentation mask.

    Args:
        mask: Instance labels, shape (H, W), dtype int.
        path: Output path.
    """
    path = Path(path)

    if path.suffix == '.npy':
        np.save(path, mask.astype(np.int32))
    elif path.suffix in ['.tif', '.tiff'] and HAS_TIFFFILE:
        tifffile.imwrite(str(path), mask.astype(np.int32))
    elif HAS_PIL:
        # For PNG, save as 16-bit if many instances
        if mask.max() > 255:
            np.save(path.with_suffix('.npy'), mask.astype(np.int32))
        else:
            Image.fromarray(mask.astype(np.uint8)).save(path)
    else:
        np.save(path.with_suffix('.npy'), mask.astype(np.int32))


# =============================================================================
# Instance Segmentation
# =============================================================================

def binary_to_instance_segmentation(
    binary_mask: np.ndarray,
    min_size: int = 50,
    use_watershed: bool = True
) -> np.ndarray:
    """
    Convert binary segmentation mask to instance segmentation.

    Uses connected components labeling with optional watershed
    refinement to separate touching nuclei.

    Args:
        binary_mask: Binary mask, shape (H, W), values in {0, 1}.
        min_size: Minimum nucleus area in pixels.
        use_watershed: Whether to use watershed for separating touching nuclei.

    Returns:
        Instance segmentation mask, shape (H, W), where each
        unique positive integer represents a different nucleus.
    """
    if not HAS_SKIMAGE:
        # Fallback: simple connected components
        from scipy.ndimage import label as scipy_label
        labeled, num_features = scipy_label(binary_mask > 0.5)
        return labeled.astype(np.int32)

    # Ensure binary
    binary = (binary_mask > 0.5).astype(np.uint8)

    if use_watershed:
        # Distance transform for watershed
        distance = ndi.distance_transform_edt(binary)

        # Find local maxima (nuclei centers)
        coords = peak_local_max(
            distance,
            min_distance=10,
            labels=binary,
            exclude_border=False
        )

        # Create markers for watershed
        markers = np.zeros_like(binary, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        markers = ndi.label(ndi.binary_dilation(markers > 0, iterations=2))[0]

        # Apply watershed
        if markers.max() > 0:
            labeled = watershed(-distance, markers, mask=binary)
        else:
            labeled = label(binary)
    else:
        # Simple connected components
        labeled = label(binary)

    # Remove small objects
    if min_size > 0:
        for region in regionprops(labeled):
            if region.area < min_size:
                labeled[labeled == region.label] = 0

        # Relabel to ensure consecutive integers
        labeled = label(labeled > 0)

    return labeled.astype(np.int32)


def merge_instance_masks(
    mask_red: np.ndarray,
    mask_green: np.ndarray,
    overlap_threshold: float = 0.5
) -> np.ndarray:
    """
    Merge instance segmentation masks from two channels.

    Takes the union of nuclei from both channels, handling overlaps
    by keeping the larger instance.

    Args:
        mask_red: Instance mask from red channel.
        mask_green: Instance mask from green channel.
        overlap_threshold: IoU threshold for merging overlapping instances.

    Returns:
        Merged instance segmentation mask.
    """
    # Start with red channel instances
    merged = mask_red.copy()
    next_label = mask_red.max() + 1

    # Add green channel instances
    for region in regionprops(mask_green):
        region_mask = mask_green == region.label

        # Check overlap with existing instances
        overlap_labels = np.unique(merged[region_mask])
        overlap_labels = overlap_labels[overlap_labels > 0]

        if len(overlap_labels) == 0:
            # No overlap: add as new instance
            merged[region_mask] = next_label
            next_label += 1
        else:
            # Check if significant overlap
            max_iou = 0
            for ol in overlap_labels:
                existing_mask = merged == ol
                intersection = np.logical_and(region_mask, existing_mask).sum()
                union = np.logical_or(region_mask, existing_mask).sum()
                iou = intersection / union if union > 0 else 0
                max_iou = max(max_iou, iou)

            if max_iou < overlap_threshold:
                # Low overlap: add as new instance
                merged[region_mask & (merged == 0)] = next_label
                next_label += 1
            # High overlap: keep existing instance

    return merged


# =============================================================================
# Multichannel TPAF Predictor
# =============================================================================

class MultichannelTPAFPredictor:
    """
    Multichannel predictor for two-channel TPAF images.

    Processes red and green channels separately through AFN-DeSeg,
    then combines outputs into:
    - Two-channel denoised image
    - Instance segmentation mask (merged from both channels)

    Args:
        checkpoint_path: Path to AFN-DeSeg model checkpoint.
        device: Device for inference ('cuda' or 'cpu').
        img_size: Input image size.
        min_nucleus_size: Minimum nucleus area in pixels.
        use_watershed: Use watershed for instance segmentation.

    Example:
        >>> predictor = MultichannelTPAFPredictor('checkpoint/best_model.pth')
        >>> denoised, instances = predictor.predict('input.tif')
        >>> print(f"Found {instances.max()} nuclei")
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        img_size: int = 512,
        min_nucleus_size: int = 50,
        use_watershed: bool = True
    ):
        self.img_size = img_size
        self.min_nucleus_size = min_nucleus_size
        self.use_watershed = use_watershed

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        print(f"MultichannelTPAFPredictor initialized on {self.device}")

    def _load_model(self, checkpoint_path: str) -> AFNDeSeg:
        """Load AFN-DeSeg model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")

        model = AFNDeSeg(
            img_size=self.img_size,
            in_channels=3,
            base_channels=64,
            vit_embed_dim=768,
            vit_depth=12,
            vit_num_heads=12,
            lora_r=16,
            lora_alpha=16
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        return model

    def _preprocess_channel(
        self,
        channel: np.ndarray
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess single channel for model input.

        Args:
            channel: Single channel image, shape (H, W).

        Returns:
            Tuple of (preprocessed tensor, original size).
        """
        original_size = channel.shape

        # Normalize to [0, 1]
        c_min, c_max = channel.min(), channel.max()
        if c_max - c_min > 1e-8:
            channel = (channel - c_min) / (c_max - c_min)

        # Pad to target size if needed
        h, w = channel.shape
        pad_h = max(0, self.img_size - h)
        pad_w = max(0, self.img_size - w)

        if pad_h > 0 or pad_w > 0:
            channel = np.pad(
                channel,
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )

        # Convert to tensor: (1, 1, H, W)
        tensor = torch.from_numpy(channel).unsqueeze(0).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        # Convert to pseudo-RGB for DINOv3: (1, 3, H, W)
        tensor = AFNDeSeg.preprocess_tpaf(tensor, normalize=True)

        return tensor, original_size

    def _postprocess(
        self,
        tensor: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Postprocess model output to original size."""
        output = tensor.detach().cpu().numpy().squeeze()
        h, w = original_size
        return output[:h, :w]

    @torch.no_grad()
    def predict_channel(
        self,
        channel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process single channel through AFN-DeSeg.

        Args:
            channel: Single channel image, shape (H, W).

        Returns:
            Tuple of (denoised_channel, binary_mask).
        """
        input_tensor, original_size = self._preprocess_channel(channel)

        # Forward pass
        denoised, segmentation = self.model(input_tensor)

        # Postprocess
        denoised_out = self._postprocess(denoised, original_size)
        seg_out = self._postprocess(segmentation, original_size)

        # Binarize segmentation
        binary_mask = (seg_out > 0.5).astype(np.float32)

        return denoised_out, binary_mask

    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        return_per_channel: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run prediction on two-channel TPAF image.

        Processes each channel separately through AFN-DeSeg and
        combines results into denoised image and instance segmentation.

        Args:
            image: Two-channel image, shape (H, W, 2), or path to image.
            return_per_channel: If True, also return per-channel results.

        Returns:
            Tuple of (denoised_image, instance_mask).
            - denoised_image: shape (H, W, 2)
            - instance_mask: shape (H, W), integer labels

        Example:
            >>> predictor = MultichannelTPAFPredictor('model.pth')
            >>> denoised, instances = predictor.predict('input.tif')
            >>> print(f"Denoised shape: {denoised.shape}")
            >>> print(f"Number of nuclei: {instances.max()}")
        """
        # Load if path
        if isinstance(image, (str, Path)):
            image = load_multichannel_image(image)

        # Validate input
        if image.ndim != 3 or image.shape[-1] != 2:
            raise ValueError(f"Expected (H, W, 2) image, got {image.shape}")

        H, W, _ = image.shape

        # Process red channel (channel 0)
        print("  Processing red channel...")
        red_channel = image[..., 0]
        denoised_red, binary_red = self.predict_channel(red_channel)

        # Process green channel (channel 1)
        print("  Processing green channel...")
        green_channel = image[..., 1]
        denoised_green, binary_green = self.predict_channel(green_channel)

        # Combine denoised channels
        denoised = np.stack([denoised_red, denoised_green], axis=-1)

        # Generate instance segmentation for each channel
        print("  Generating instance segmentation...")
        instances_red = binary_to_instance_segmentation(
            binary_red,
            min_size=self.min_nucleus_size,
            use_watershed=self.use_watershed
        )
        instances_green = binary_to_instance_segmentation(
            binary_green,
            min_size=self.min_nucleus_size,
            use_watershed=self.use_watershed
        )

        # Merge instances from both channels
        instance_mask = merge_instance_masks(instances_red, instances_green)

        print(f"  Found {instance_mask.max()} nuclei instances")

        if return_per_channel:
            return denoised, instance_mask, {
                'red': {'denoised': denoised_red, 'binary': binary_red, 'instances': instances_red},
                'green': {'denoised': denoised_green, 'binary': binary_green, 'instances': instances_green}
            }

        return denoised, instance_mask

    def predict_with_kda(
        self,
        image: Union[np.ndarray, str, Path],
        kda_predictor: 'KDAPredictor'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full pipeline: denoising, segmentation, and KDA prediction.

        Args:
            image: Two-channel TPAF image or path.
            kda_predictor: Initialized KDAPredictor instance.

        Returns:
            Tuple of (denoised_image, instance_mask, kda_mask).
        """
        # Get denoised image and instance segmentation
        denoised, instance_mask = self.predict(image)

        # Convert instance mask to binary for KDA
        binary_mask = (instance_mask > 0).astype(np.float32)

        # Run KDA prediction
        print("  Predicting Key Diagnostic Areas...")
        kda_mask = kda_predictor.predict(binary_mask)

        return denoised, instance_mask, kda_mask


# =============================================================================
# Directory Processing
# =============================================================================

def predict_multichannel_directory(
    predictor: MultichannelTPAFPredictor,
    input_dir: str,
    output_dir: str,
    kda_predictor: Optional['KDAPredictor'] = None,
    save_format: str = 'tif'
):
    """
    Run multichannel prediction on all images in a directory.

    Args:
        predictor: MultichannelTPAFPredictor instance.
        input_dir: Input directory with two-channel TPAF images.
        output_dir: Output directory for results.
        kda_predictor: Optional KDAPredictor for KDA analysis.
        save_format: Output format ('tif', 'png', 'npy').
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    denoised_dir = output_dir / 'denoised'
    instances_dir = output_dir / 'instance_masks'
    denoised_dir.mkdir(parents=True, exist_ok=True)
    instances_dir.mkdir(parents=True, exist_ok=True)

    if kda_predictor is not None:
        kda_dir = output_dir / 'kda_masks'
        kda_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_paths = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.npy']:
        image_paths.extend(input_dir.glob(ext))
    image_paths = sorted(image_paths)

    print(f"Found {len(image_paths)} images")

    # Process each image
    results_summary = []
    for path in tqdm(image_paths, desc="Processing"):
        stem = path.stem
        print(f"\nProcessing: {stem}")

        try:
            if kda_predictor is not None:
                denoised, instances, kda = predictor.predict_with_kda(path, kda_predictor)

                # Save KDA mask
                save_instance_mask(
                    (kda > 0.5).astype(np.int32),
                    kda_dir / f"{stem}_kda.{save_format}"
                )
            else:
                denoised, instances = predictor.predict(path)

            # Save denoised image
            save_multichannel_image(denoised, denoised_dir / f"{stem}_denoised.{save_format}")

            # Save instance mask
            save_instance_mask(instances, instances_dir / f"{stem}_instances.{save_format}")

            results_summary.append({
                'file': stem,
                'num_nuclei': int(instances.max()),
                'status': 'success'
            })

        except Exception as e:
            print(f"  Error: {e}")
            results_summary.append({
                'file': stem,
                'error': str(e),
                'status': 'failed'
            })

    # Save summary
    import json
    with open(output_dir / 'processing_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"Successfully processed: {sum(1 for r in results_summary if r['status'] == 'success')}/{len(results_summary)}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AFN-DeSeg Multichannel Inference: Process two-channel TPAF images'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to AFN-DeSeg model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory (two-channel TPAF)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--format', type=str, default='tif',
                        choices=['tif', 'png', 'npy'],
                        help='Output format')
    parser.add_argument('--min_nucleus_size', type=int, default=50,
                        help='Minimum nucleus area in pixels')
    parser.add_argument('--no_watershed', action='store_true',
                        help='Disable watershed for instance segmentation')

    # KDA options
    parser.add_argument('--run_kda', action='store_true',
                        help='Run KDA prediction')
    parser.add_argument('--kda_checkpoint', type=str, default=None,
                        help='Path to KDA model checkpoint')

    return parser.parse_args()


def main():
    """Entry point for multichannel TPAF inference."""
    args = parse_args()

    # Initialize AFN-DeSeg predictor
    predictor = MultichannelTPAFPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        img_size=args.img_size,
        min_nucleus_size=args.min_nucleus_size,
        use_watershed=not args.no_watershed
    )

    # Initialize KDA predictor if requested
    kda_predictor = None
    if args.run_kda:
        if args.kda_checkpoint is None:
            raise ValueError("--kda_checkpoint required when --run_kda is set")

        from inference.kda_predictor import KDAPredictor
        kda_predictor = KDAPredictor(
            model_path=args.kda_checkpoint,
            device=args.device
        )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        # Process directory
        predict_multichannel_directory(
            predictor,
            args.input,
            args.output,
            kda_predictor,
            args.format
        )
    else:
        # Process single image
        print(f"Processing {input_path}")

        if kda_predictor is not None:
            denoised, instances, kda = predictor.predict_with_kda(input_path, kda_predictor)
            save_instance_mask(
                (kda > 0.5).astype(np.int32),
                output_path / f"{input_path.stem}_kda.{args.format}"
            )
        else:
            denoised, instances = predictor.predict(input_path)

        # Save outputs
        save_multichannel_image(denoised, output_path / f"{input_path.stem}_denoised.{args.format}")
        save_instance_mask(instances, output_path / f"{input_path.stem}_instances.{args.format}")

        print(f"Results saved to {output_path}")
        print(f"Found {instances.max()} nuclei instances")


if __name__ == '__main__':
    main()
