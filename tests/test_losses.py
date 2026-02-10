"""
Unit tests for AFN-DeSeg loss functions.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses import AFNJointLoss, ReconstructionLoss, SegmentationLoss, DiceLoss


class TestReconstructionLoss:
    """Tests for reconstruction loss."""

    def test_reconstruction_loss_zero(self):
        """Test that loss is zero for identical inputs."""
        loss_fn = ReconstructionLoss()
        x = torch.rand(2, 1, 64, 64)

        loss = loss_fn(x, x)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_reconstruction_loss_positive(self):
        """Test that loss is positive for different inputs."""
        loss_fn = ReconstructionLoss()
        pred = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.item() > 0.0


class TestDiceLoss:
    """Tests for Dice loss."""

    def test_dice_loss_perfect_match(self):
        """Test that loss is zero for perfect match."""
        loss_fn = DiceLoss()
        x = torch.ones(2, 1, 64, 64)

        loss = loss_fn(x, x)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_dice_loss_no_overlap(self):
        """Test that loss is ~1 for no overlap."""
        loss_fn = DiceLoss()
        pred = torch.zeros(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64)

        loss = loss_fn(pred, target)

        # Should be close to 1 (no overlap)
        assert loss.item() > 0.9

    def test_dice_loss_range(self):
        """Test that loss is in valid range [0, 1]."""
        loss_fn = DiceLoss()
        pred = torch.rand(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()

        loss = loss_fn(pred, target)

        assert 0.0 <= loss.item() <= 1.0


class TestSegmentationLoss:
    """Tests for combined segmentation loss."""

    def test_segmentation_loss_components(self):
        """Test that all loss components are returned."""
        loss_fn = SegmentationLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()

        total, dice, bce = loss_fn(pred, target)

        assert isinstance(total, torch.Tensor)
        assert isinstance(dice, torch.Tensor)
        assert isinstance(bce, torch.Tensor)
        assert total.item() >= 0.0


class TestAFNJointLoss:
    """Tests for the joint loss function."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function without Cellpose for testing."""
        return AFNJointLoss(
            lambda_rec=1.0,
            lambda_seg=10.0,
            lambda_percep=0.1,
            use_cellpose=False  # Use placeholder for testing
        )

    def test_joint_loss_forward(self, loss_fn):
        """Test forward pass of joint loss."""
        denoised_pred = torch.rand(2, 1, 64, 64)
        seg_pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        clean_gt = torch.rand(2, 1, 64, 64)
        seg_gt = (torch.rand(2, 1, 64, 64) > 0.5).float()

        total_loss, loss_dict = loss_fn(denoised_pred, seg_pred, clean_gt, seg_gt)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0.0

        # Check loss dictionary
        assert 'loss_rec' in loss_dict
        assert 'loss_seg' in loss_dict
        assert 'loss_percep' in loss_dict
        assert 'loss_total' in loss_dict

    def test_loss_weights(self, loss_fn):
        """Test that loss weights are applied correctly."""
        weights = loss_fn.get_loss_weights()

        assert weights['lambda_rec'] == 1.0
        assert weights['lambda_seg'] == 10.0
        assert weights['lambda_percep'] == 0.1

    def test_set_loss_weights(self, loss_fn):
        """Test setting loss weights."""
        loss_fn.set_loss_weights(lambda_rec=2.0, lambda_seg=5.0)

        weights = loss_fn.get_loss_weights()
        assert weights['lambda_rec'] == 2.0
        assert weights['lambda_seg'] == 5.0

    def test_backward_pass(self, loss_fn):
        """Test that gradients flow correctly."""
        denoised_pred = torch.rand(2, 1, 64, 64, requires_grad=True)
        seg_pred = torch.sigmoid(torch.randn(2, 1, 64, 64, requires_grad=True))
        clean_gt = torch.rand(2, 1, 64, 64)
        seg_gt = (torch.rand(2, 1, 64, 64) > 0.5).float()

        total_loss, _ = loss_fn(denoised_pred, seg_pred, clean_gt, seg_gt)
        total_loss.backward()

        assert denoised_pred.grad is not None
        assert seg_pred.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
