"""
Inference Script for AFN-DeSeg.

Runs trained model on new images to produce denoised images and segmentation masks.

Usage:
    python inference/predict.py \
        --checkpoint /path/to/best_model.pth \
        --input /path/to/images \
        --output /path/to/output
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.afn_deseg import AFNDeSeg

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    from skimage import io as skimage_io
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# =============================================================================
# Image I/O
# =============================================================================

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from file."""
    path = Path(path)

    if path.suffix == '.npy':
        return np.load(path).astype(np.float32)
    elif path.suffix in ['.tif', '.tiff'] and HAS_TIFFFILE:
        return tifffile.imread(str(path)).astype(np.float32)
    elif HAS_SKIMAGE:
        return skimage_io.imread(str(path)).astype(np.float32)
    else:
        raise ImportError("Install scikit-image or tifffile to load images")


def save_image(
    img: np.ndarray,
    path: Union[str, Path],
    normalize: bool = True
):
    """Save image to file."""
    path = Path(path)

    if normalize:
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)

    if path.suffix == '.npy':
        np.save(path, img)
    elif path.suffix in ['.tif', '.tiff'] and HAS_TIFFFILE:
        tifffile.imwrite(str(path), img.astype(np.float32))
    elif HAS_SKIMAGE:
        # Convert to uint8 for PNG
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        skimage_io.imsave(str(path), img_uint8)
    else:
        np.save(path.with_suffix('.npy'), img)


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess(
    image: np.ndarray,
    img_size: int = 512,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess image for model input.

    Args:
        image: Input image (H, W) or (H, W, C).
        img_size: Target size (will pad if needed).
        device: Target device.

    Returns:
        Tuple of (preprocessed tensor, original size).
    """
    original_size = image.shape[:2]

    # Convert to 2D
    if image.ndim == 3:
        if image.shape[-1] <= 4:
            image = image.mean(axis=-1)
        else:
            image = image[..., 0]

    # Normalize to [0, 1]
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 1e-8:
        image = (image - img_min) / (img_max - img_min)

    # Pad to target size if needed
    h, w = image.shape
    pad_h = max(0, img_size - h)
    pad_w = max(0, img_size - w)

    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )

    # Convert to tensor: (1, 1, H, W)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    tensor = tensor.to(device)

    # Convert to pseudo-RGB and normalize for DINOv3
    tensor = AFNDeSeg.preprocess_tpaf(tensor, normalize=True)

    return tensor, original_size


def postprocess(
    tensor: torch.Tensor,
    original_size: Tuple[int, int]
) -> np.ndarray:
    """
    Postprocess model output.

    Args:
        tensor: Model output tensor.
        original_size: Original image size (H, W).

    Returns:
        Output image cropped to original size.
    """
    # Convert to numpy
    output = tensor.detach().cpu().numpy().squeeze()

    # Crop to original size
    h, w = original_size
    output = output[:h, :w]

    return output


# =============================================================================
# Inference
# =============================================================================

class AFNDeSegPredictor:
    """
    Predictor class for AFN-DeSeg inference.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device for inference ('cuda' or 'cpu').
        img_size: Input image size.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        img_size: int = 512
    ):
        self.img_size = img_size

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

    def _load_model(self, checkpoint_path: str) -> AFNDeSeg:
        """Load model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")

        # Initialize model
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

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        print(f"Model loaded successfully on {self.device}")

        return model

    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run prediction on a single image.

        Args:
            image: Input image as array or path.

        Returns:
            Tuple of (denoised_image, segmentation_mask).
        """
        # Load if path
        if isinstance(image, (str, Path)):
            image = load_image(image)

        # Preprocess
        input_tensor, original_size = preprocess(image, self.img_size, self.device)

        # Forward pass
        denoised, segmentation = self.model(input_tensor)

        # Postprocess
        denoised_output = postprocess(denoised, original_size)
        segmentation_output = postprocess(segmentation, original_size)

        # Binarize segmentation
        segmentation_binary = (segmentation_output > 0.5).astype(np.float32)

        return denoised_output, segmentation_binary

    def predict_batch(
        self,
        images: list
    ) -> Tuple[list, list]:
        """
        Run prediction on a batch of images.

        Args:
            images: List of input images.

        Returns:
            Tuple of (denoised_images, segmentation_masks).
        """
        denoised_list = []
        segmentation_list = []

        for img in images:
            denoised, segmentation = self.predict(img)
            denoised_list.append(denoised)
            segmentation_list.append(segmentation)

        return denoised_list, segmentation_list


def predict_directory(
    predictor: AFNDeSegPredictor,
    input_dir: str,
    output_dir: str,
    save_format: str = 'tif'
):
    """
    Run prediction on all images in a directory.

    Args:
        predictor: AFNDeSegPredictor instance.
        input_dir: Input directory with images.
        output_dir: Output directory for results.
        save_format: Output format ('tif', 'png', 'npy').
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    denoised_dir = output_dir / 'denoised'
    masks_dir = output_dir / 'masks'
    denoised_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_paths = []
    for ext in ['*.npy', '*.png', '*.tif', '*.tiff']:
        image_paths.extend(input_dir.glob(ext))
    image_paths = sorted(image_paths)

    print(f"Found {len(image_paths)} images")

    # Process each image
    for path in tqdm(image_paths, desc="Processing"):
        # Predict
        denoised, segmentation = predictor.predict(path)

        # Save outputs
        stem = path.stem
        save_image(denoised, denoised_dir / f"{stem}.{save_format}")
        save_image(segmentation, masks_dir / f"{stem}.{save_format}", normalize=False)

    print(f"Results saved to {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='AFN-DeSeg Inference: Denoise and segment TPAF images'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--format', type=str, default='tif',
                        choices=['tif', 'png', 'npy'],
                        help='Output format')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images')

    return parser.parse_args()


def main():
    """Entry point for AFN-DeSeg inference."""
    args = parse_args()

    # Initialize predictor
    predictor = AFNDeSegPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        img_size=args.img_size
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        # Process directory
        predict_directory(predictor, args.input, args.output, args.format)
    else:
        # Process single image
        print(f"Processing {input_path}")
        denoised, segmentation = predictor.predict(input_path)

        # Save outputs
        stem = input_path.stem
        save_image(denoised, output_path / f"{stem}_denoised.{args.format}")
        save_image(segmentation, output_path / f"{stem}_mask.{args.format}", normalize=False)

        print(f"Results saved to {output_path}")

    # Visualize if requested
    if args.visualize:
        try:
            from utils.visualization import plot_full_result

            if input_path.is_file():
                image = load_image(input_path)
                denoised, segmentation = predictor.predict(image)
                plot_full_result(
                    image, denoised, segmentation,
                    save_path=str(output_path / f"{input_path.stem}_visualization.png")
                )
        except ImportError:
            print("Visualization requires matplotlib")


if __name__ == '__main__':
    main()
