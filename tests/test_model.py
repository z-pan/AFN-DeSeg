"""
Unit tests for AFN-DeSeg model architecture.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import AFNDeSeg, DINOv3Encoder, UNetEncoder, FusionBlock


class TestAFNDeSeg:
    """Tests for the main AFN-DeSeg model."""

    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return AFNDeSeg(
            img_size=256,  # Smaller for faster testing
            in_channels=3,
            base_channels=32,  # Smaller for faster testing
            vit_embed_dim=384,  # Smaller for faster testing
            vit_depth=4,  # Fewer layers for faster testing
            vit_num_heads=6,
            lora_r=8,
            lora_alpha=8
        )

    def test_model_instantiation(self, model):
        """Test that model can be instantiated."""
        assert model is not None
        assert isinstance(model, AFNDeSeg)

    def test_forward_pass(self, model):
        """Test forward pass with correct input shape."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 256, 256)

        denoised, segmented = model(x)

        assert denoised.shape == (batch_size, 1, 256, 256)
        assert segmented.shape == (batch_size, 1, 256, 256)

    def test_output_ranges(self, model):
        """Test that outputs are in valid ranges."""
        x = torch.randn(1, 3, 256, 256)

        denoised, segmented = model(x)

        # Segmentation should be in [0, 1] due to sigmoid
        assert segmented.min() >= 0.0
        assert segmented.max() <= 1.0

    def test_preprocess_tpaf(self):
        """Test TPAF preprocessing function."""
        # Single channel input
        x = torch.rand(2, 1, 256, 256)
        preprocessed = AFNDeSeg.preprocess_tpaf(x, normalize=True)

        assert preprocessed.shape == (2, 3, 256, 256)

    def test_freeze_vit(self, model):
        """Test freezing ViT encoder."""
        # Count trainable params before
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.freeze_vit_pretrained()

        # Count trainable params after
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have fewer trainable params after freezing
        assert trainable_after < trainable_before

    def test_get_lora_params(self, model):
        """Test getting LoRA parameters."""
        lora_params = model.get_lora_params()

        assert len(lora_params) > 0
        for param in lora_params:
            assert param.requires_grad


class TestUNetEncoder:
    """Tests for U-Net encoder."""

    def test_encoder_output_shape(self):
        """Test encoder output shapes."""
        encoder = UNetEncoder(in_channels=3, base_channels=64)
        x = torch.randn(2, 3, 512, 512)

        bottleneck, skip_connections = encoder(x)

        # Bottleneck should be 512 channels, 32x32
        assert bottleneck.shape == (2, 512, 32, 32)

        # Should have 4 skip connections
        assert len(skip_connections) == 4

        # Check skip connection shapes
        expected_shapes = [
            (2, 64, 512, 512),
            (2, 128, 256, 256),
            (2, 256, 128, 128),
            (2, 512, 64, 64)
        ]
        for skip, expected in zip(skip_connections, expected_shapes):
            assert skip.shape == expected


class TestDINOv3Encoder:
    """Tests for DINOv3 encoder."""

    def test_encoder_output_shape(self):
        """Test encoder output shape."""
        encoder = DINOv3Encoder(
            img_size=256,
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6
        )
        x = torch.randn(2, 3, 256, 256)

        tokens = encoder(x)

        # Should have (256/16)^2 = 256 patches
        assert tokens.shape == (2, 256, 384)

    def test_positional_embedding_interpolation(self):
        """Test positional embedding interpolation for different sizes."""
        encoder = DINOv3Encoder(
            img_size=256,
            patch_size=16,
            embed_dim=384,
            depth=2,
            num_heads=6
        )

        # Test with different input size
        x = torch.randn(1, 3, 512, 512)
        tokens = encoder(x)

        # Should have (512/16)^2 = 1024 patches
        assert tokens.shape == (1, 1024, 384)


class TestFusionBlock:
    """Tests for feature fusion block."""

    def test_fusion_output_shape(self):
        """Test fusion block output shape."""
        fusion = FusionBlock(
            unet_channels=512,
            vit_channels=768,
            out_channels=512
        )

        unet_features = torch.randn(2, 512, 32, 32)
        vit_tokens = torch.randn(2, 1024, 768)  # 32x32 grid

        fused = fusion(unet_features, vit_tokens, grid_size=32)

        assert fused.shape == (2, 512, 32, 32)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
