"""
Dataset classes for AFN-DeSeg.

Provides reusable PyTorch Dataset implementations for:
- TPAFDataset: Supervised training with noisy/clean/mask triplets
- TPAFUnlabeledDataset: Self-supervised training with unlabeled TPAF images
- SyntheticNoiseDataset: Training with synthetic noise from clean images
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
# Utility Functions
# =============================================================================

def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load image from various formats.

    Args:
        path: Path to image file (.npy, .png, .tif, .tiff).

    Returns:
        Image as numpy array (float32).
    """
    path = Path(path)

    if path.suffix == '.npy':
        return np.load(path).astype(np.float32)
    elif path.suffix in ['.tif', '.tiff'] and HAS_TIFFFILE:
        return tifffile.imread(str(path)).astype(np.float32)
    elif HAS_SKIMAGE:
        return skimage_io.imread(str(path)).astype(np.float32)
    else:
        raise ImportError("Please install scikit-image or tifffile to load images")


def find_file_with_extensions(
    directory: Path,
    name: str,
    extensions: List[str] = ['.npy', '.png', '.tif', '.tiff']
) -> Optional[Path]:
    """
    Find file with given name in directory (trying multiple extensions).

    Args:
        directory: Directory to search in.
        name: Filename without extension.
        extensions: List of extensions to try.

    Returns:
        Path to file if found, None otherwise.
    """
    for ext in extensions:
        path = directory / f"{name}{ext}"
        if path.exists():
            return path
    return None


def normalize_image(img: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        img: Input image.
        method: Normalization method ('minmax', 'percentile', 'fixed').

    Returns:
        Normalized image.
    """
    if method == 'minmax':
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            return (img - img_min) / (img_max - img_min)
        return img
    elif method == 'percentile':
        p_low, p_high = np.percentile(img, [1, 99])
        img = np.clip(img, p_low, p_high)
        if p_high - p_low > 1e-8:
            return (img - p_low) / (p_high - p_low)
        return img
    elif method == 'fixed':
        # Assume 8-bit or 16-bit images
        if img.max() > 255:
            return img / 65535.0
        elif img.max() > 1:
            return img / 255.0
        return img
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def ensure_2d(img: np.ndarray) -> np.ndarray:
    """Convert image to 2D if needed."""
    if img.ndim == 3:
        if img.shape[-1] <= 4:  # Channels last
            return img.mean(axis=-1)
        elif img.shape[0] <= 4:  # Channels first
            return img.mean(axis=0)
    return img


# =============================================================================
# TPAF Dataset (Supervised)
# =============================================================================

class TPAFDataset(Dataset):
    """
    TPAF Dataset for supervised joint denoising and segmentation.

    Expected directory structure:
        data_dir/
            noisy/      # Noisy TPAF images
            clean/      # Clean reference images (ground truth for denoising)
            masks/      # Segmentation masks (ground truth for segmentation)

    All images should have matching filenames across folders.

    Args:
        data_dir: Root directory containing noisy/, clean/, masks/ folders.
        split: Dataset split ('train', 'val', 'test').
        img_size: Expected image size (for validation).
        transform: Optional transform/augmentation function.
        normalize: Normalization method ('minmax', 'percentile', 'fixed').
        return_filename: Whether to return filename with each sample.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        img_size: int = 512,
        transform: Optional[Callable] = None,
        normalize: str = 'minmax',
        return_filename: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.normalize = normalize
        self.return_filename = return_filename

        # Setup directories
        self.noisy_dir = self.data_dir / 'noisy'
        self.clean_dir = self.data_dir / 'clean'
        self.mask_dir = self.data_dir / 'masks'

        # Validate directories exist
        if not self.noisy_dir.exists():
            raise ValueError(f"Noisy directory not found: {self.noisy_dir}")

        # Get sample list
        self.samples = self._get_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.noisy_dir}")

    def _get_samples(self) -> List[str]:
        """Get list of sample names from noisy directory."""
        samples = []
        for f in self.noisy_dir.iterdir():
            if f.suffix in ['.npy', '.png', '.tif', '.tiff']:
                samples.append(f.stem)
        return sorted(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
        sample_name = self.samples[idx]

        # Load images
        noisy_path = find_file_with_extensions(self.noisy_dir, sample_name)
        clean_path = find_file_with_extensions(self.clean_dir, sample_name)
        mask_path = find_file_with_extensions(self.mask_dir, sample_name)

        # Load noisy (required)
        noisy = load_image(noisy_path)
        noisy = ensure_2d(noisy)

        # Load clean (use noisy as fallback)
        if clean_path and clean_path.exists():
            clean = load_image(clean_path)
            clean = ensure_2d(clean)
        else:
            clean = noisy.copy()

        # Load mask (use zeros as fallback)
        if mask_path and mask_path.exists():
            mask = load_image(mask_path)
            mask = ensure_2d(mask)
        else:
            mask = np.zeros_like(noisy)

        # Normalize images
        noisy = normalize_image(noisy, self.normalize)
        clean = normalize_image(clean, self.normalize)
        mask = (mask > 0.5).astype(np.float32)  # Binary mask

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=noisy, clean=clean, mask=mask)
            noisy = transformed['image']
            clean = transformed['clean']
            mask = transformed['mask']

        # Convert to tensors with channel dimension
        noisy = torch.from_numpy(noisy).unsqueeze(0).float()
        clean = torch.from_numpy(clean).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.return_filename:
            return noisy, clean, mask, sample_name
        return noisy, clean, mask


# =============================================================================
# Unlabeled TPAF Dataset (Self-Supervised)
# =============================================================================

class TPAFUnlabeledDataset(Dataset):
    """
    Unlabeled TPAF Dataset for self-supervised learning (Stage 1 DINO adaptation).

    Args:
        data_dir: Directory containing unlabeled TPAF images.
        img_size: Image size for cropping.
        transform: Optional transform/augmentation function.
        normalize: Normalization method.
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = 512,
        transform: Optional[Callable] = None,
        normalize: str = 'minmax'
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform
        self.normalize = normalize

        # Get all image files
        self.image_paths = self._get_image_paths()

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")

    def _get_image_paths(self) -> List[Path]:
        """Get all image file paths."""
        paths = []
        for ext in ['*.npy', '*.png', '*.tif', '*.tiff']:
            paths.extend(self.data_dir.glob(ext))
            paths.extend(self.data_dir.glob(f'**/{ext}'))  # Recursive
        return sorted(set(paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load image
        img = load_image(self.image_paths[idx])
        img = ensure_2d(img)
        img = normalize_image(img, self.normalize)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']

        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0).float()

        return img


# =============================================================================
# Synthetic Noise Dataset
# =============================================================================

class SyntheticNoiseDataset(Dataset):
    """
    Dataset that generates synthetic noisy images from clean images.

    Uses MPG noise model to create training pairs on-the-fly.

    Args:
        clean_dir: Directory containing clean images.
        noise_synthesizer: MPGNoiseSynthesizer instance.
        mask_dir: Optional directory containing segmentation masks.
        img_size: Image size.
        transform: Optional transform/augmentation function.
        normalize: Normalization method.
    """

    def __init__(
        self,
        clean_dir: str,
        noise_synthesizer,
        mask_dir: Optional[str] = None,
        img_size: int = 512,
        transform: Optional[Callable] = None,
        normalize: str = 'minmax'
    ):
        self.clean_dir = Path(clean_dir)
        self.noise_synthesizer = noise_synthesizer
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.img_size = img_size
        self.transform = transform
        self.normalize = normalize

        # Get clean image files
        self.samples = self._get_samples()

    def _get_samples(self) -> List[str]:
        """Get list of sample names."""
        samples = []
        for f in self.clean_dir.iterdir():
            if f.suffix in ['.npy', '.png', '.tif', '.tiff']:
                samples.append(f.stem)
        return sorted(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_name = self.samples[idx]

        # Load clean image
        clean_path = find_file_with_extensions(self.clean_dir, sample_name)
        clean = load_image(clean_path)
        clean = ensure_2d(clean)
        clean = normalize_image(clean, self.normalize)

        # Load mask if available
        if self.mask_dir:
            mask_path = find_file_with_extensions(self.mask_dir, sample_name)
            if mask_path:
                mask = load_image(mask_path)
                mask = ensure_2d(mask)
                mask = (mask > 0.5).astype(np.float32)
            else:
                mask = np.zeros_like(clean)
        else:
            mask = np.zeros_like(clean)

        # Apply transforms before noise (to augment both clean and mask consistently)
        if self.transform is not None:
            transformed = self.transform(image=clean, mask=mask)
            clean = transformed['image']
            mask = transformed['mask']

        # Synthesize noisy image
        noisy, _ = self.noise_synthesizer.synthesize(clean)

        # Convert to tensors
        noisy = torch.from_numpy(noisy).unsqueeze(0).float()
        clean = torch.from_numpy(clean).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return noisy, clean, mask


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    img_size: int = 512,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    train_split: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Root data directory.
        batch_size: Batch size.
        img_size: Image size.
        num_workers: Number of dataloader workers.
        train_transform: Transform for training data.
        val_transform: Transform for validation data.
        train_split: Fraction of data for training.
        seed: Random seed for splitting.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Create full dataset
    full_dataset = TPAFDataset(
        data_dir,
        img_size=img_size,
        transform=None,  # Will set per-split
        normalize='minmax'
    )

    # Split dataset
    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
