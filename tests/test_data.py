"""
Unit tests for AFN-DeSeg data processing.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import (
    MPGNoiseModel,
    MPGNoiseSynthesizer,
    add_mpg_noise,
    add_mpg_noise_numpy,
    add_mpg_noise_torch
)


class TestMPGNoiseModel:
    """Tests for Mixed Poisson-Gaussian noise model."""

    def test_model_instantiation(self):
        """Test noise model creation."""
        model = MPGNoiseModel(gain=2.0, sigma_read=5.0)

        assert model.gain == 2.0
        assert model.sigma_read == 5.0

    def test_get_parameters(self):
        """Test getting parameters."""
        model = MPGNoiseModel(gain=1.5, sigma_read=3.0)
        gain, sigma = model.get_parameters()

        assert gain == 1.5
        assert sigma == 3.0

    def test_set_parameters(self):
        """Test setting parameters."""
        model = MPGNoiseModel()
        model.set_parameters(gain=2.5, sigma_read=4.0)

        assert model.gain == 2.5
        assert model.sigma_read == 4.0

    def test_parameter_estimation(self):
        """Test parameter estimation from calibration images."""
        model = MPGNoiseModel()

        # Create synthetic clean and noisy images
        clean = np.random.rand(5, 64, 64).astype(np.float32) * 100
        # Add known noise
        gain = 2.0
        sigma = 5.0
        noisy = clean + np.random.randn(*clean.shape).astype(np.float32) * sigma

        estimated_gain, estimated_sigma = model.estimate_parameters(
            noisy, clean, patch_size=16
        )

        # Parameters should be estimated (not exact due to noise)
        assert estimated_gain > 0
        assert estimated_sigma >= 0


class TestMPGNoiseSynthesis:
    """Tests for MPG noise synthesis functions."""

    def test_add_noise_numpy(self):
        """Test adding noise to numpy array."""
        clean = np.random.rand(64, 64).astype(np.float32) * 100

        noisy = add_mpg_noise_numpy(clean, gain=2.0, sigma_read=5.0)

        assert noisy.shape == clean.shape
        assert noisy.dtype == clean.dtype
        # Noisy should be different from clean
        assert not np.allclose(noisy, clean)

    def test_add_noise_torch(self):
        """Test adding noise to PyTorch tensor."""
        clean = torch.rand(1, 1, 64, 64) * 100

        noisy = add_mpg_noise_torch(clean, gain=2.0, sigma_read=5.0)

        assert noisy.shape == clean.shape
        assert noisy.device == clean.device
        # Noisy should be different from clean
        assert not torch.allclose(noisy, clean)

    def test_add_noise_dispatch(self):
        """Test automatic dispatch based on input type."""
        # NumPy input
        clean_np = np.random.rand(64, 64).astype(np.float32)
        noisy_np = add_mpg_noise(clean_np, gain=2.0, sigma_read=5.0)
        assert isinstance(noisy_np, np.ndarray)

        # PyTorch input
        clean_torch = torch.rand(64, 64)
        noisy_torch = add_mpg_noise(clean_torch, gain=2.0, sigma_read=5.0)
        assert isinstance(noisy_torch, torch.Tensor)

    def test_clip_range(self):
        """Test clipping output to valid range."""
        clean = np.random.rand(64, 64).astype(np.float32) * 100

        noisy = add_mpg_noise_numpy(
            clean, gain=2.0, sigma_read=5.0,
            clip_range=(0, 100)
        )

        assert noisy.min() >= 0
        assert noisy.max() <= 100


class TestMPGNoiseSynthesizer:
    """Tests for noise synthesizer class."""

    def test_synthesizer_creation(self):
        """Test synthesizer with default noise levels."""
        synthesizer = MPGNoiseSynthesizer()

        assert len(synthesizer) == 3  # Default 3 noise levels

    def test_synthesizer_custom_levels(self):
        """Test synthesizer with custom noise levels."""
        levels = [(1.0, 10.0), (2.0, 5.0)]
        synthesizer = MPGNoiseSynthesizer(noise_levels=levels)

        assert len(synthesizer) == 2

    def test_synthesize(self):
        """Test synthesizing noisy image."""
        synthesizer = MPGNoiseSynthesizer()
        clean = np.random.rand(64, 64).astype(np.float32)

        noisy, level = synthesizer.synthesize(clean)

        assert noisy.shape == clean.shape
        assert 0 <= level < len(synthesizer)

    def test_get_noise_level_params(self):
        """Test getting parameters for specific noise level."""
        levels = [(1.0, 10.0), (2.0, 5.0)]
        synthesizer = MPGNoiseSynthesizer(noise_levels=levels)

        gain, sigma = synthesizer.get_noise_level_params(0)
        assert gain == 1.0
        assert sigma == 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
