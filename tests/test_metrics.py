"""
Unit tests for AFN-DeSeg evaluation metrics.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import (
    compute_psnr,
    compute_ssim,
    compute_dice,
    compute_iou,
    compute_morphology_stats,
    compute_bland_altman,
    MetricTracker
)


class TestImageRestorationMetrics:
    """Tests for image restoration metrics."""

    def test_psnr_identical(self):
        """Test PSNR for identical images."""
        img = np.random.rand(64, 64)
        psnr = compute_psnr(img, img)

        # Should be very high (close to 100 dB)
        assert psnr >= 90.0

    def test_psnr_different(self):
        """Test PSNR for different images."""
        img1 = np.random.rand(64, 64)
        img2 = np.random.rand(64, 64)
        psnr = compute_psnr(img1, img2)

        # Should be finite and positive
        assert 0 < psnr < 100

    def test_psnr_tensor(self):
        """Test PSNR with PyTorch tensors."""
        img1 = torch.rand(1, 1, 64, 64)
        img2 = torch.rand(1, 1, 64, 64)
        psnr = compute_psnr(img1, img2)

        assert isinstance(psnr, float)
        assert psnr > 0

    def test_ssim_identical(self):
        """Test SSIM for identical images."""
        img = np.random.rand(64, 64)
        ssim = compute_ssim(img, img)

        # Should be 1.0 for identical images
        assert ssim == pytest.approx(1.0, abs=0.01)

    def test_ssim_different(self):
        """Test SSIM for different images."""
        img1 = np.random.rand(64, 64)
        img2 = np.random.rand(64, 64)
        ssim = compute_ssim(img1, img2)

        # Should be between -1 and 1
        assert -1 <= ssim <= 1

    def test_ssim_tensor(self):
        """Test SSIM with PyTorch tensors."""
        img1 = torch.rand(1, 1, 64, 64)
        img2 = torch.rand(1, 1, 64, 64)
        ssim = compute_ssim(img1, img2)

        assert isinstance(ssim, float)


class TestSegmentationMetrics:
    """Tests for segmentation metrics."""

    def test_dice_perfect(self):
        """Test Dice for perfect match."""
        mask = np.ones((64, 64))
        dice = compute_dice(mask, mask)

        assert dice == pytest.approx(1.0, abs=1e-6)

    def test_dice_no_overlap(self):
        """Test Dice for no overlap."""
        pred = np.zeros((64, 64))
        target = np.ones((64, 64))
        dice = compute_dice(pred, target)

        assert dice == pytest.approx(0.0, abs=1e-6)

    def test_dice_partial(self):
        """Test Dice for partial overlap."""
        pred = np.zeros((64, 64))
        pred[:32, :] = 1
        target = np.zeros((64, 64))
        target[:, :32] = 1

        dice = compute_dice(pred, target)

        # Should be between 0 and 1
        assert 0 < dice < 1

    def test_iou_perfect(self):
        """Test IoU for perfect match."""
        mask = np.ones((64, 64))
        iou = compute_iou(mask, mask)

        assert iou == pytest.approx(1.0, abs=1e-6)

    def test_iou_range(self):
        """Test IoU is in valid range."""
        pred = np.random.rand(64, 64)
        target = (np.random.rand(64, 64) > 0.5).astype(float)

        iou = compute_iou(pred, target)

        assert 0 <= iou <= 1


class TestMorphologyMetrics:
    """Tests for morphology metrics."""

    def test_morphology_stats_empty(self):
        """Test morphology stats for empty mask."""
        mask = np.zeros((64, 64))
        stats = compute_morphology_stats(mask)

        assert stats['count'] == 0
        assert stats['density'] == 0.0

    def test_morphology_stats_single_object(self):
        """Test morphology stats for single object."""
        mask = np.zeros((64, 64))
        mask[20:40, 20:40] = 1  # 20x20 square

        stats = compute_morphology_stats(mask)

        assert stats['count'] == 1
        assert stats['density'] > 0


class TestBlandAltman:
    """Tests for Bland-Altman analysis."""

    def test_bland_altman_identical(self):
        """Test Bland-Altman for identical measurements."""
        x = np.array([1, 2, 3, 4, 5])
        result = compute_bland_altman(x, x)

        assert result['bias'] == pytest.approx(0.0, abs=1e-6)

    def test_bland_altman_offset(self):
        """Test Bland-Altman with constant offset."""
        x = np.array([1, 2, 3, 4, 5])
        y = x + 1  # Constant offset of 1

        result = compute_bland_altman(x, y)

        assert result['bias'] == pytest.approx(-1.0, abs=1e-6)

    def test_bland_altman_loa(self):
        """Test limits of agreement."""
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.1

        result = compute_bland_altman(x, y)

        # LOA should bracket the bias
        assert result['loa_lower'] < result['bias'] < result['loa_upper']


class TestMetricTracker:
    """Tests for metric tracker."""

    def test_tracker_update(self):
        """Test updating tracker."""
        tracker = MetricTracker()
        tracker.update({'loss': 1.0, 'dice': 0.8})
        tracker.update({'loss': 0.5, 'dice': 0.9})

        metrics = tracker.compute()

        assert metrics['loss'] == pytest.approx(0.75, abs=1e-6)
        assert metrics['dice'] == pytest.approx(0.85, abs=1e-6)

    def test_tracker_reset(self):
        """Test resetting tracker."""
        tracker = MetricTracker()
        tracker.update({'loss': 1.0})
        tracker.reset()

        assert tracker.metrics == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
