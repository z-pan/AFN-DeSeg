"""
Unit Tests for Attention U-Net Model.

Tests the complete Attention U-Net architecture for KDA prediction,
including forward pass, output properties, and attention extraction.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.attention_unet import (
    AttentionUNet,
    ConvBlock,
    EncoderBlock,
    DecoderBlock,
    create_attention_unet
)


class TestConvBlock:
    """Tests for ConvBlock module."""

    def test_conv_block_output_shape(self):
        """Test convolution block output shape."""
        block = ConvBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 128, 128)

        output = block(x)

        assert output.shape == (2, 128, 128, 128)

    def test_conv_block_with_dropout(self):
        """Test convolution block with dropout."""
        block = ConvBlock(in_channels=64, out_channels=128, dropout=0.5)
        x = torch.randn(2, 64, 64, 64)

        # In training mode
        block.train()
        output_train = block(x)

        # In eval mode
        block.eval()
        output_eval = block(x)

        assert output_train.shape == output_eval.shape


class TestEncoderBlock:
    """Tests for EncoderBlock module."""

    def test_encoder_block_output_shapes(self):
        """Test encoder block output shapes."""
        block = EncoderBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 128, 128)

        pooled, features = block(x)

        # Pooled should be half the spatial size
        assert pooled.shape == (2, 128, 64, 64)
        # Features (skip connection) should have same spatial size
        assert features.shape == (2, 128, 128, 128)


class TestDecoderBlock:
    """Tests for DecoderBlock module."""

    def test_decoder_block_output_shape(self):
        """Test decoder block output shape."""
        block = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            attention_channels=64
        )

        x = torch.randn(2, 256, 32, 32)  # From previous decoder level
        skip = torch.randn(2, 128, 64, 64)  # From encoder

        output = block(x, skip)

        # Output should match skip connection spatial size
        assert output.shape == (2, 128, 64, 64)


class TestAttentionUNet:
    """Tests for complete AttentionUNet model."""

    def test_attention_unet_forward(self):
        """Test full Attention U-Net forward pass."""
        model = AttentionUNet(in_channels=1, base_filters=64)
        x = torch.randn(2, 1, 512, 512)

        output = model(x)

        assert output.shape == (2, 1, 512, 512), f"Expected (2, 1, 512, 512), got {output.shape}"

    def test_attention_unet_output_range(self):
        """Test that output is in [0, 1] range (sigmoid activation)."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        model.eval()

        x = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            output = model(x)

        assert (output >= 0).all(), "Output should be >= 0"
        assert (output <= 1).all(), "Output should be <= 1"

    def test_attention_unet_small_image(self):
        """Test with smaller image size."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        x = torch.randn(2, 1, 128, 128)

        output = model(x)

        assert output.shape == (2, 1, 128, 128)

    def test_attention_unet_large_image(self):
        """Test with larger image size."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        x = torch.randn(1, 1, 1024, 1024)

        output = model(x)

        assert output.shape == (1, 1, 1024, 1024)

    def test_attention_unet_non_square_image(self):
        """Test with non-square image."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        x = torch.randn(2, 1, 256, 512)

        output = model(x)

        assert output.shape == (2, 1, 256, 512)

    def test_attention_unet_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        x = torch.randn(2, 1, 128, 128, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradient should flow to input"

    def test_attention_unet_with_attention_extraction(self):
        """Test forward pass with attention map extraction."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        x = torch.randn(2, 1, 128, 128)

        output, attention_maps = model.forward_with_attention(x)

        assert output.shape == (2, 1, 128, 128)
        assert 'level1' in attention_maps
        assert 'level2' in attention_maps
        assert 'level3' in attention_maps
        assert 'level4' in attention_maps

    def test_attention_maps_properties(self):
        """Test properties of extracted attention maps."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        model.eval()

        x = torch.randn(2, 1, 128, 128)

        with torch.no_grad():
            _, attention_maps = model.forward_with_attention(x)

        for level, attn in attention_maps.items():
            # Attention maps should be in [0, 1]
            assert (attn >= 0).all(), f"Attention at {level} should be >= 0"
            assert (attn <= 1).all(), f"Attention at {level} should be <= 1"

    def test_attention_unet_eval_mode(self):
        """Test model in evaluation mode."""
        model = AttentionUNet(in_channels=1, base_filters=32, dropout=0.5)
        model.eval()

        x = torch.randn(2, 1, 128, 128)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 1, 128, 128)

    def test_attention_unet_num_parameters(self):
        """Test parameter counting."""
        model = AttentionUNet(in_channels=1, base_filters=64)

        num_params = model.get_num_parameters(trainable_only=True)

        assert num_params > 0, "Model should have parameters"
        # Typical Attention U-Net with base_filters=64 has ~30M parameters
        assert num_params > 1000000, "Model should have significant parameters"

    def test_attention_unet_custom_config(self):
        """Test model with custom configuration."""
        model = AttentionUNet(
            in_channels=1,
            base_filters=32,
            num_classes=1,
            attention_filters=128,
            dropout=0.2
        )

        x = torch.randn(2, 1, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)


class TestCreateAttentionUNet:
    """Tests for factory function."""

    def test_create_attention_unet_default(self):
        """Test factory function with default parameters."""
        model = create_attention_unet()

        assert isinstance(model, AttentionUNet)

    def test_create_attention_unet_custom_params(self):
        """Test factory function with custom parameters."""
        model = create_attention_unet(
            in_channels=1,
            base_filters=32,
            attention_filters=128
        )

        x = torch.randn(1, 1, 128, 128)
        output = model(x)

        assert output.shape == (1, 1, 128, 128)


class TestKDAWorkflow:
    """Integration tests for KDA prediction workflow."""

    def test_binary_mask_input(self):
        """Test with binary mask input (typical use case)."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        model.eval()

        # Simulate binary nuclear mask
        nuclear_mask = torch.zeros(1, 1, 256, 256)
        nuclear_mask[:, :, 50:100, 50:100] = 1.0
        nuclear_mask[:, :, 150:200, 150:200] = 1.0

        with torch.no_grad():
            kda_pred = model(nuclear_mask)

        assert kda_pred.shape == (1, 1, 256, 256)
        assert (kda_pred >= 0).all() and (kda_pred <= 1).all()

    def test_all_zeros_input(self):
        """Test with all-zeros input (no nuclei)."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        model.eval()

        x = torch.zeros(1, 1, 128, 128)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1, 128, 128)
        # Model should still produce valid output
        assert not torch.isnan(output).any()

    def test_all_ones_input(self):
        """Test with all-ones input (dense nuclei)."""
        model = AttentionUNet(in_channels=1, base_filters=32)
        model.eval()

        x = torch.ones(1, 1, 128, 128)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1, 128, 128)
        assert not torch.isnan(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
