"""
Physics-Informed Mixed Poisson-Gaussian (MPG) Noise Synthesis Module.

Based on Note S3 from the AFN-DeSeg supplementary information.
This module implements the noise model for TPAF microscopy images:

    I_noisy = (1/g) * Poisson(g * I_clean) + N(0, sigma_read^2)

Where:
    - g: Detector gain factor (converts photons to digital units)
    - sigma_read: Standard deviation of electronic readout noise
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import torch


class MPGNoiseModel:
    """
    Mixed Poisson-Gaussian Noise Model for TPAF microscopy.

    This class handles parameter estimation from calibration images and
    noise synthesis for training data generation.

    Attributes:
        gain (float): Detector gain factor (g).
        sigma_read (float): Standard deviation of readout noise.
    """

    def __init__(self, gain: float = 1.0, sigma_read: float = 0.0):
        """
        Initialize the MPG noise model.

        Args:
            gain: Detector gain factor. Default is 1.0.
            sigma_read: Standard deviation of readout noise. Default is 0.0.
        """
        self.gain = gain
        self.sigma_read = sigma_read

    def estimate_parameters(
        self,
        noisy_images: np.ndarray,
        clean_images: np.ndarray,
        patch_size: int = 32
    ) -> Tuple[float, float]:
        """
        Estimate gain and readout noise from calibration image pairs.

        Uses linear regression on the mean-variance relationship:
            variance = (1/g) * mean + sigma_read^2

        The estimation process:
            1. Compute pixel-wise difference: D = I_noisy - I_clean
            2. Calculate mean (mu) of I_clean and variance (v) of D for local patches
            3. Perform linear regression: v = (1/g) * mu + sigma_read^2
               (Slope = 1/g, Intercept = sigma_read^2)

        Args:
            noisy_images: Noisy calibration images, shape (N, H, W) or (H, W).
            clean_images: Clean reference images (e.g., 16-frame average),
                         shape (N, H, W) or (H, W).
            patch_size: Size of local patches for statistics computation.

        Returns:
            Tuple of (gain, sigma_read).
        """
        # Ensure 3D arrays
        if noisy_images.ndim == 2:
            noisy_images = noisy_images[np.newaxis, ...]
        if clean_images.ndim == 2:
            clean_images = clean_images[np.newaxis, ...]

        means = []
        variances = []

        for noisy, clean in zip(noisy_images, clean_images):
            # Compute pixel-wise difference
            diff = noisy.astype(np.float64) - clean.astype(np.float64)

            h, w = clean.shape

            # Compute statistics for local patches
            for i in range(0, h - patch_size + 1, patch_size):
                for j in range(0, w - patch_size + 1, patch_size):
                    clean_patch = clean[i:i+patch_size, j:j+patch_size]
                    diff_patch = diff[i:i+patch_size, j:j+patch_size]

                    # Skip patches with very low signal (avoid division issues)
                    patch_mean = np.mean(clean_patch)
                    if patch_mean < 1e-6:
                        continue

                    patch_var = np.var(diff_patch)

                    means.append(patch_mean)
                    variances.append(patch_var)

        means = np.array(means)
        variances = np.array(variances)

        # Linear regression: v = (1/g) * mu + sigma_read^2
        # Using least squares: [mu, 1] @ [1/g, sigma_read^2]^T = v
        A = np.column_stack([means, np.ones_like(means)])
        coeffs, _, _, _ = np.linalg.lstsq(A, variances, rcond=None)

        slope = coeffs[0]  # 1/g
        intercept = coeffs[1]  # sigma_read^2

        # Extract parameters
        self.gain = 1.0 / max(slope, 1e-10)  # Avoid division by zero
        self.sigma_read = np.sqrt(max(intercept, 0.0))  # Ensure non-negative

        return self.gain, self.sigma_read

    def get_parameters(self) -> Tuple[float, float]:
        """
        Get current noise model parameters.

        Returns:
            Tuple of (gain, sigma_read).
        """
        return self.gain, self.sigma_read

    def set_parameters(self, gain: float, sigma_read: float) -> None:
        """
        Set noise model parameters manually.

        Args:
            gain: Detector gain factor.
            sigma_read: Standard deviation of readout noise.
        """
        self.gain = gain
        self.sigma_read = sigma_read


def add_mpg_noise_numpy(
    clean_image: np.ndarray,
    gain: float,
    sigma_read: float,
    clip_range: Optional[Tuple[float, float]] = None,
    random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Add Mixed Poisson-Gaussian noise to a clean image (NumPy implementation).

    Applies the noise model:
        I_noisy = (1/g) * Poisson(g * I_clean) + N(0, sigma_read^2)

    Args:
        clean_image: Clean input image as NumPy array, shape (H, W) or (C, H, W).
        gain: Detector gain factor (g).
        sigma_read: Standard deviation of readout noise.
        clip_range: Optional tuple (min, max) to clip output values.
        random_state: Optional numpy RandomState for reproducibility.

    Returns:
        Noisy image as NumPy array with same shape as input.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    clean_image = clean_image.astype(np.float64)

    # Ensure non-negative values for Poisson distribution
    clean_clipped = np.clip(clean_image, 0, None)

    # Scale by gain and apply Poisson noise
    # Poisson parameter must be non-negative
    poisson_param = gain * clean_clipped
    poisson_noise = random_state.poisson(poisson_param).astype(np.float64)

    # Scale back by 1/gain
    shot_noise_component = poisson_noise / gain

    # Add Gaussian readout noise
    gaussian_noise = random_state.normal(0, sigma_read, clean_image.shape)

    # Combine: I_noisy = shot_noise_component + gaussian_noise
    noisy_image = shot_noise_component + gaussian_noise

    # Optionally clip to valid range
    if clip_range is not None:
        noisy_image = np.clip(noisy_image, clip_range[0], clip_range[1])

    return noisy_image.astype(clean_image.dtype)


def add_mpg_noise_torch(
    clean_image: torch.Tensor,
    gain: float,
    sigma_read: float,
    clip_range: Optional[Tuple[float, float]] = None,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Add Mixed Poisson-Gaussian noise to a clean image (PyTorch implementation).

    Applies the noise model:
        I_noisy = (1/g) * Poisson(g * I_clean) + N(0, sigma_read^2)

    Args:
        clean_image: Clean input image as PyTorch tensor, shape (H, W),
                    (C, H, W), or (B, C, H, W).
        gain: Detector gain factor (g).
        sigma_read: Standard deviation of readout noise.
        clip_range: Optional tuple (min, max) to clip output values.
        generator: Optional torch Generator for reproducibility.

    Returns:
        Noisy image as PyTorch tensor with same shape and device as input.
    """
    device = clean_image.device
    dtype = clean_image.dtype

    # Work in float64 for numerical stability
    clean_float = clean_image.double()

    # Ensure non-negative values for Poisson distribution
    clean_clipped = torch.clamp(clean_float, min=0)

    # Scale by gain and apply Poisson noise
    poisson_param = gain * clean_clipped

    # PyTorch Poisson distribution
    poisson_dist = torch.distributions.Poisson(poisson_param)
    if generator is not None:
        # Set the seed from generator for reproducibility
        torch.manual_seed(generator.initial_seed())
    poisson_noise = poisson_dist.sample()

    # Scale back by 1/gain
    shot_noise_component = poisson_noise / gain

    # Add Gaussian readout noise
    gaussian_noise = torch.randn(
        clean_image.shape,
        device=device,
        dtype=torch.float64,
        generator=generator
    ) * sigma_read

    # Combine: I_noisy = shot_noise_component + gaussian_noise
    noisy_image = shot_noise_component + gaussian_noise

    # Optionally clip to valid range
    if clip_range is not None:
        noisy_image = torch.clamp(noisy_image, clip_range[0], clip_range[1])

    return noisy_image.to(dtype)


def add_mpg_noise(
    clean_image: Union[np.ndarray, torch.Tensor],
    gain: float,
    sigma_read: float,
    clip_range: Optional[Tuple[float, float]] = None,
    random_state: Optional[Union[np.random.RandomState, torch.Generator]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Add Mixed Poisson-Gaussian noise to a clean image.

    Automatically dispatches to NumPy or PyTorch implementation based on input type.

    Applies the noise model from Note S3:
        I_noisy = (1/g) * Poisson(g * I_clean) + N(0, sigma_read^2)

    Args:
        clean_image: Clean input image as NumPy array or PyTorch tensor.
        gain: Detector gain factor (g).
        sigma_read: Standard deviation of readout noise.
        clip_range: Optional tuple (min, max) to clip output values.
        random_state: Optional random state (np.random.RandomState for NumPy,
                     torch.Generator for PyTorch).

    Returns:
        Noisy image with same type and shape as input.

    Example:
        >>> import numpy as np
        >>> clean = np.random.rand(512, 512) * 255
        >>> noisy = add_mpg_noise(clean, gain=2.0, sigma_read=5.0)

        >>> import torch
        >>> clean_t = torch.rand(1, 1, 512, 512) * 255
        >>> noisy_t = add_mpg_noise(clean_t, gain=2.0, sigma_read=5.0)
    """
    if isinstance(clean_image, torch.Tensor):
        return add_mpg_noise_torch(
            clean_image, gain, sigma_read, clip_range,
            generator=random_state if isinstance(random_state, torch.Generator) else None
        )
    else:
        return add_mpg_noise_numpy(
            clean_image, gain, sigma_read, clip_range,
            random_state=random_state if isinstance(random_state, np.random.RandomState) else None
        )


class MPGNoiseSynthesizer:
    """
    Synthesizer for generating training pairs with multiple noise levels.

    Based on Note S3, this class supports multiple empirical noise levels
    (e.g., 1-frame, 4-frame, 8-frame equivalent noise).

    Attributes:
        noise_levels: List of (gain, sigma_read) tuples for different noise levels.
    """

    def __init__(self, noise_levels: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize the noise synthesizer.

        Args:
            noise_levels: List of (gain, sigma_read) tuples. If None, uses
                         default levels based on typical TPAF imaging.
        """
        if noise_levels is None:
            # Default noise levels (to be calibrated with real data)
            # These are placeholder values - should be estimated from calibration
            self.noise_levels = [
                (1.0, 10.0),   # Level 1: High noise (1-frame equivalent)
                (2.0, 7.0),    # Level 2: Medium noise (4-frame equivalent)
                (4.0, 5.0),    # Level 3: Low noise (8-frame equivalent)
            ]
        else:
            self.noise_levels = noise_levels

    def add_noise_levels_from_calibration(
        self,
        model: MPGNoiseModel,
        noisy_images_list: List[np.ndarray],
        clean_images: np.ndarray,
        patch_size: int = 32
    ) -> None:
        """
        Estimate and add noise levels from calibration data.

        Args:
            model: MPGNoiseModel instance for parameter estimation.
            noisy_images_list: List of noisy image arrays at different noise levels.
            clean_images: Clean reference images.
            patch_size: Patch size for statistics computation.
        """
        self.noise_levels = []
        for noisy_images in noisy_images_list:
            gain, sigma_read = model.estimate_parameters(
                noisy_images, clean_images, patch_size
            )
            self.noise_levels.append((gain, sigma_read))

    def synthesize(
        self,
        clean_image: Union[np.ndarray, torch.Tensor],
        noise_level: Optional[int] = None,
        clip_range: Optional[Tuple[float, float]] = None,
        random_state: Optional[Union[np.random.RandomState, torch.Generator]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
        """
        Synthesize a noisy image from a clean image.

        Args:
            clean_image: Clean input image.
            noise_level: Index of noise level to use. If None, randomly selects.
            clip_range: Optional tuple (min, max) to clip output values.
            random_state: Optional random state for reproducibility.

        Returns:
            Tuple of (noisy_image, noise_level_index).
        """
        if noise_level is None:
            noise_level = np.random.randint(0, len(self.noise_levels))

        gain, sigma_read = self.noise_levels[noise_level]

        noisy_image = add_mpg_noise(
            clean_image, gain, sigma_read, clip_range, random_state
        )

        return noisy_image, noise_level

    def __len__(self) -> int:
        """Return number of noise levels."""
        return len(self.noise_levels)

    def get_noise_level_params(self, level: int) -> Tuple[float, float]:
        """
        Get parameters for a specific noise level.

        Args:
            level: Noise level index.

        Returns:
            Tuple of (gain, sigma_read).
        """
        return self.noise_levels[level]
