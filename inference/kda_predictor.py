"""
Inference Script for Key Diagnostic Area (KDA) Prediction.

Runs trained Attention U-Net model on nuclear segmentation masks
to predict Key Diagnostic Areas for clinical analysis.

Usage:
    python inference/kda_predictor.py \
        --checkpoint checkpoints/kda_best.pth \
        --input /path/to/nuclear_masks \
        --output /path/to/kda_predictions
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

from models.attention_unet import AttentionUNet, create_attention_unet

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Image I/O
# =============================================================================

def load_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Load nuclear segmentation mask from file.

    Args:
        path: Path to mask file (.npy, .png, .tif).

    Returns:
        Binary mask as float32 array, shape (H, W).
    """
    path = Path(path)

    if path.suffix == '.npy':
        mask = np.load(path)
    elif path.suffix in ['.tif', '.tiff'] and HAS_TIFFFILE:
        mask = tifffile.imread(str(path))
    elif HAS_PIL:
        mask = np.array(Image.open(path))
    else:
        raise ImportError("Install PIL or tifffile to load image files")

    # Ensure 2D
    if mask.ndim == 3:
        mask = mask.mean(axis=-1)

    # Ensure binary float
    mask = (mask > 0.5).astype(np.float32)

    return mask


def save_prediction(
    prediction: np.ndarray,
    path: Union[str, Path],
    as_probability: bool = False
):
    """
    Save KDA prediction to file.

    Args:
        prediction: Predicted KDA mask or probability map.
        path: Output path.
        as_probability: If True, save as probability map; else binary.
    """
    path = Path(path)

    if not as_probability:
        prediction = (prediction > 0.5).astype(np.uint8) * 255

    if path.suffix == '.npy':
        np.save(path, prediction)
    elif path.suffix in ['.tif', '.tiff'] and HAS_TIFFFILE:
        tifffile.imwrite(str(path), prediction.astype(np.float32 if as_probability else np.uint8))
    elif HAS_PIL:
        if as_probability:
            img = (prediction * 255).clip(0, 255).astype(np.uint8)
        else:
            img = prediction.astype(np.uint8)
        Image.fromarray(img).save(path)
    else:
        np.save(path.with_suffix('.npy'), prediction)


# =============================================================================
# KDA Predictor
# =============================================================================

class KDAPredictor:
    """
    Inference engine for Key Diagnostic Area prediction.

    Uses trained Attention U-Net to predict KDA from nuclear
    segmentation masks.

    Args:
        model_path: Path to trained Attention U-Net checkpoint.
        device: Device for inference ('cuda' or 'cpu').
        threshold: Probability threshold for binarization (default: 0.5).
        base_filters: Base filter count (must match training).
        attention_filters: Attention filter count (must match training).

    Example:
        >>> predictor = KDAPredictor(model_path='checkpoints/kda_best.pth')
        >>> kda_mask = predictor.predict(nuclear_mask)
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
        base_filters: int = 64,
        attention_filters: int = 256
    ):
        self.threshold = threshold

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(
            model_path,
            base_filters,
            attention_filters
        )
        self.model.eval()

        print(f"KDA Predictor initialized on {self.device}")

    def _load_model(
        self,
        model_path: str,
        base_filters: int,
        attention_filters: int
    ) -> AttentionUNet:
        """Load model from checkpoint."""
        print(f"Loading model from {model_path}")

        # Initialize model
        model = AttentionUNet(
            in_channels=1,
            base_filters=base_filters,
            attention_filters=attention_filters
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'config' in checkpoint:
                print(f"  Checkpoint config: {checkpoint['config']}")
            if 'best_val_dice' in checkpoint:
                print(f"  Best validation Dice: {checkpoint['best_val_dice']:.4f}")
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)

        return model

    def preprocess(
        self,
        nuclear_mask: np.ndarray
    ) -> torch.Tensor:
        """
        Preprocess nuclear mask for model input.

        Args:
            nuclear_mask: Binary nuclear mask, shape (H, W) or (1, H, W).

        Returns:
            Preprocessed tensor, shape (1, 1, H, W).
        """
        # Ensure numpy
        if isinstance(nuclear_mask, torch.Tensor):
            nuclear_mask = nuclear_mask.cpu().numpy()

        # Ensure 2D
        if nuclear_mask.ndim == 3:
            nuclear_mask = nuclear_mask.squeeze()

        # Ensure binary float
        mask = (nuclear_mask > 0.5).astype(np.float32)

        # Convert to tensor: (1, 1, H, W)
        tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        return tensor

    def postprocess(
        self,
        output: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None,
        return_prob: bool = False
    ) -> np.ndarray:
        """
        Postprocess model output.

        Args:
            output: Model output tensor.
            original_size: Original (H, W) to resize to.
            return_prob: If True, return probability map; else binary.

        Returns:
            KDA prediction as numpy array.
        """
        # Convert to numpy
        pred = output.detach().cpu().numpy().squeeze()

        # Resize if needed
        if original_size is not None and pred.shape != original_size:
            from scipy.ndimage import zoom
            zoom_factors = (
                original_size[0] / pred.shape[0],
                original_size[1] / pred.shape[1]
            )
            pred = zoom(pred, zoom_factors, order=1)

        # Binarize if requested
        if not return_prob:
            pred = (pred > self.threshold).astype(np.float32)

        return pred

    @torch.no_grad()
    def predict(
        self,
        nuclear_mask: Union[np.ndarray, torch.Tensor, str, Path],
        return_prob: bool = False
    ) -> np.ndarray:
        """
        Predict Key Diagnostic Areas from nuclear segmentation mask.

        Args:
            nuclear_mask: Binary nuclear mask, shape (H, W) or (1, H, W).
                         Can also be a path to a mask file.
            return_prob: If True, return probability map [0, 1];
                        else binary mask {0, 1}.

        Returns:
            KDA prediction: Binary mask (H, W) or probability map.

        Example:
            >>> predictor = KDAPredictor('kda_best.pth')
            >>> kda = predictor.predict(nuclear_mask)
            >>> kda.shape
            (512, 512)
        """
        # Load if path
        if isinstance(nuclear_mask, (str, Path)):
            nuclear_mask = load_mask(nuclear_mask)

        # Store original size
        if isinstance(nuclear_mask, torch.Tensor):
            original_size = nuclear_mask.shape[-2:]
        else:
            original_size = nuclear_mask.shape[-2:] if nuclear_mask.ndim > 2 else nuclear_mask.shape

        # Preprocess
        input_tensor = self.preprocess(nuclear_mask)

        # Forward pass
        output = self.model(input_tensor)

        # Postprocess
        prediction = self.postprocess(output, original_size, return_prob)

        return prediction

    @torch.no_grad()
    def predict_with_attention(
        self,
        nuclear_mask: Union[np.ndarray, torch.Tensor, str, Path]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict KDA with attention map extraction for visualization.

        Args:
            nuclear_mask: Binary nuclear mask.

        Returns:
            Tuple of (kda_prediction, attention_maps).
            attention_maps contains attention coefficients from each decoder level.
        """
        # Load if path
        if isinstance(nuclear_mask, (str, Path)):
            nuclear_mask = load_mask(nuclear_mask)

        original_size = nuclear_mask.shape[-2:] if nuclear_mask.ndim > 2 else nuclear_mask.shape

        # Preprocess
        input_tensor = self.preprocess(nuclear_mask)

        # Forward with attention
        output, attention_maps_torch = self.model.forward_with_attention(input_tensor)

        # Convert attention maps to numpy
        attention_maps = {}
        for level, attn in attention_maps_torch.items():
            attention_maps[level] = attn.detach().cpu().numpy().squeeze()

        # Postprocess prediction
        prediction = self.postprocess(output, original_size, return_prob=False)

        return prediction, attention_maps

    def batch_predict(
        self,
        nuclear_masks: List[Union[np.ndarray, str, Path]],
        return_prob: bool = False,
        batch_size: int = 8
    ) -> List[np.ndarray]:
        """
        Batch prediction for multiple masks.

        Args:
            nuclear_masks: List of nuclear masks or paths.
            return_prob: If True, return probability maps.
            batch_size: Batch size for inference.

        Returns:
            List of KDA predictions.
        """
        predictions = []

        for mask in tqdm(nuclear_masks, desc="Predicting KDA"):
            pred = self.predict(mask, return_prob=return_prob)
            predictions.append(pred)

        return predictions


# =============================================================================
# Directory Processing
# =============================================================================

def predict_directory(
    predictor: KDAPredictor,
    input_dir: str,
    output_dir: str,
    save_format: str = 'png',
    save_attention: bool = False
):
    """
    Run KDA prediction on all masks in a directory.

    Args:
        predictor: KDAPredictor instance.
        input_dir: Directory with nuclear masks.
        output_dir: Output directory for predictions.
        save_format: Output format ('png', 'tif', 'npy').
        save_attention: Whether to save attention maps.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    kda_dir = output_dir / 'kda_masks'
    kda_dir.mkdir(parents=True, exist_ok=True)

    if save_attention:
        attn_dir = output_dir / 'attention_maps'
        attn_dir.mkdir(parents=True, exist_ok=True)

    # Find all masks
    mask_paths = []
    for ext in ['*.npy', '*.png', '*.tif', '*.tiff']:
        mask_paths.extend(input_dir.glob(ext))
    mask_paths = sorted(mask_paths)

    print(f"Found {len(mask_paths)} masks")

    # Process each mask
    for path in tqdm(mask_paths, desc="Processing"):
        stem = path.stem

        if save_attention:
            kda_pred, attention_maps = predictor.predict_with_attention(path)

            # Save attention maps
            for level, attn in attention_maps.items():
                attn_path = attn_dir / f"{stem}_{level}.npy"
                np.save(attn_path, attn)
        else:
            kda_pred = predictor.predict(path)

        # Save KDA prediction
        save_prediction(kda_pred, kda_dir / f"{stem}_kda.{save_format}")

    print(f"Results saved to {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='KDA Prediction: Predict Key Diagnostic Areas from nuclear masks'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained Attention U-Net checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input nuclear mask or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for binarization')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'tif', 'npy'],
                        help='Output format')
    parser.add_argument('--save_attention', action='store_true',
                        help='Save attention maps')
    parser.add_argument('--save_prob', action='store_true',
                        help='Save probability maps instead of binary')

    return parser.parse_args()


def main():
    """Entry point for KDA prediction."""
    args = parse_args()

    # Initialize predictor
    predictor = KDAPredictor(
        model_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        # Process directory
        predict_directory(
            predictor,
            args.input,
            args.output,
            args.format,
            args.save_attention
        )
    else:
        # Process single mask
        print(f"Processing {input_path}")

        if args.save_attention:
            kda_pred, attention_maps = predictor.predict_with_attention(input_path)

            # Save attention
            for level, attn in attention_maps.items():
                np.save(output_path / f"{input_path.stem}_{level}.npy", attn)
        else:
            kda_pred = predictor.predict(input_path, return_prob=args.save_prob)

        # Save prediction
        save_prediction(
            kda_pred,
            output_path / f"{input_path.stem}_kda.{args.format}",
            as_probability=args.save_prob
        )

        print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
