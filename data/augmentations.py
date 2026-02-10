"""
Data Augmentation Pipeline for AFN-DeSeg.

Provides augmentation transforms for training using albumentations library.
Based on section 4.2.3: "Input images underwent geometric augmentations
including random flips and orthogonal rotations."
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# =============================================================================
# Custom Transforms
# =============================================================================

if HAS_ALBUMENTATIONS:

    class IntensityPerturbation(ImageOnlyTransform):
        """
        Random intensity perturbation for self-supervised learning.

        Applies random brightness and contrast adjustments.
        """

        def __init__(
            self,
            brightness_limit: float = 0.2,
            contrast_limit: float = 0.2,
            always_apply: bool = False,
            p: float = 0.5
        ):
            super().__init__(always_apply, p)
            self.brightness_limit = brightness_limit
            self.contrast_limit = contrast_limit

        def apply(self, img: np.ndarray, **params) -> np.ndarray:
            # Random brightness
            brightness = 1.0 + np.random.uniform(-self.brightness_limit, self.brightness_limit)
            # Random contrast
            contrast = 1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit)

            # Apply: img = contrast * (img - mean) + mean + brightness_offset
            mean = img.mean()
            img = contrast * (img - mean) + mean
            img = img + (brightness - 1.0) * mean

            return np.clip(img, 0, 1).astype(np.float32)

        def get_transform_init_args_names(self) -> Tuple[str, ...]:
            return ('brightness_limit', 'contrast_limit')


    class GaussianNoisePerturbation(ImageOnlyTransform):
        """
        Add small Gaussian noise for data augmentation.

        Different from MPG noise synthesis - this is for augmentation only.
        """

        def __init__(
            self,
            var_limit: Tuple[float, float] = (0.001, 0.01),
            always_apply: bool = False,
            p: float = 0.5
        ):
            super().__init__(always_apply, p)
            self.var_limit = var_limit

        def apply(self, img: np.ndarray, **params) -> np.ndarray:
            var = np.random.uniform(self.var_limit[0], self.var_limit[1])
            noise = np.random.normal(0, np.sqrt(var), img.shape)
            return np.clip(img + noise, 0, 1).astype(np.float32)

        def get_transform_init_args_names(self) -> Tuple[str, ...]:
            return ('var_limit',)


    class MultiCropTransform:
        """
        Multi-crop augmentation for DINO-style self-supervised learning.

        Generates multiple views of the same image:
        - Global crops (large, covering most of the image)
        - Local crops (small, random patches)
        """

        def __init__(
            self,
            global_crop_size: int = 224,
            local_crop_size: int = 96,
            n_global_crops: int = 2,
            n_local_crops: int = 6,
            global_crop_scale: Tuple[float, float] = (0.4, 1.0),
            local_crop_scale: Tuple[float, float] = (0.05, 0.4)
        ):
            self.global_crop_size = global_crop_size
            self.local_crop_size = local_crop_size
            self.n_global_crops = n_global_crops
            self.n_local_crops = n_local_crops

            # Global crop transform
            self.global_transform = A.Compose([
                A.RandomResizedCrop(
                    height=global_crop_size,
                    width=global_crop_size,
                    scale=global_crop_scale,
                    ratio=(0.75, 1.33)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                IntensityPerturbation(p=0.8),
            ])

            # Local crop transform
            self.local_transform = A.Compose([
                A.RandomResizedCrop(
                    height=local_crop_size,
                    width=local_crop_size,
                    scale=local_crop_scale,
                    ratio=(0.75, 1.33)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                IntensityPerturbation(p=0.8),
            ])

        def __call__(self, image: np.ndarray) -> Dict[str, List[np.ndarray]]:
            """
            Generate multiple crops from input image.

            Args:
                image: Input image (H, W) or (H, W, C).

            Returns:
                Dictionary with 'global_crops' and 'local_crops' lists.
            """
            global_crops = []
            local_crops = []

            for _ in range(self.n_global_crops):
                crop = self.global_transform(image=image)['image']
                global_crops.append(crop)

            for _ in range(self.n_local_crops):
                crop = self.local_transform(image=image)['image']
                local_crops.append(crop)

            return {
                'global_crops': global_crops,
                'local_crops': local_crops
            }


# =============================================================================
# Standard Augmentation Pipelines
# =============================================================================

def get_train_transform(img_size: int = 512) -> Any:
    """
    Get training augmentation pipeline.

    Based on paper section 4.2.3: geometric augmentations including
    random flips and orthogonal rotations.

    Args:
        img_size: Target image size.

    Returns:
        Albumentations Compose object.
    """
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose(
        [
            # Geometric augmentations (as per paper)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # Optional: slight elastic deformation for robustness
            A.ElasticTransform(
                alpha=50,
                sigma=10,
                p=0.2
            ),

            # Ensure correct size
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,  # cv2.BORDER_CONSTANT
                value=0
            ),
            A.CenterCrop(height=img_size, width=img_size),
        ],
        additional_targets={'clean': 'image', 'mask': 'mask'}
    )


def get_val_transform(img_size: int = 512) -> Any:
    """
    Get validation/test augmentation pipeline (no augmentation, just resize).

    Args:
        img_size: Target image size.

    Returns:
        Albumentations Compose object.
    """
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                value=0
            ),
            A.CenterCrop(height=img_size, width=img_size),
        ],
        additional_targets={'clean': 'image', 'mask': 'mask'}
    )


def get_dino_transform(img_size: int = 512) -> Any:
    """
    Get DINO self-supervised augmentation pipeline.

    For Stage 1 domain adaptation with intensity perturbations.

    Args:
        img_size: Target image size.

    Returns:
        Albumentations Compose object.
    """
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Intensity perturbations (as mentioned in paper)
        IntensityPerturbation(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.8
        ),

        # Random crop for multi-scale learning
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
            p=0.5
        ),

        # Ensure correct size
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            value=0
        ),
        A.CenterCrop(height=img_size, width=img_size),

        # Optional Gaussian noise
        GaussianNoisePerturbation(var_limit=(0.001, 0.005), p=0.3),
    ])


# =============================================================================
# Simple Augmentation (No albumentations dependency)
# =============================================================================

class SimpleAugmentation:
    """
    Simple augmentation without albumentations dependency.

    Implements basic geometric augmentations using numpy.
    """

    def __init__(self, flip: bool = True, rotate: bool = True):
        self.flip = flip
        self.rotate = rotate

    def __call__(
        self,
        image: np.ndarray,
        clean: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply augmentations.

        Args:
            image: Input image (H, W).
            clean: Optional clean image (H, W).
            mask: Optional mask (H, W).

        Returns:
            Dictionary with augmented arrays.
        """
        result = {'image': image.copy()}
        if clean is not None:
            result['clean'] = clean.copy()
        if mask is not None:
            result['mask'] = mask.copy()

        # Random horizontal flip
        if self.flip and np.random.rand() > 0.5:
            for key in result:
                result[key] = np.flip(result[key], axis=-1).copy()

        # Random vertical flip
        if self.flip and np.random.rand() > 0.5:
            for key in result:
                result[key] = np.flip(result[key], axis=-2).copy()

        # Random 90-degree rotation
        if self.rotate:
            k = np.random.randint(0, 4)
            if k > 0:
                for key in result:
                    result[key] = np.rot90(result[key], k, axes=(-2, -1)).copy()

        return result


def get_simple_train_transform() -> SimpleAugmentation:
    """Get simple training augmentation (no dependencies)."""
    return SimpleAugmentation(flip=True, rotate=True)


def get_simple_val_transform() -> SimpleAugmentation:
    """Get simple validation transform (no augmentation)."""
    return SimpleAugmentation(flip=False, rotate=False)
