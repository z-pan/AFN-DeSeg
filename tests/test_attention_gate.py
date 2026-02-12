"""
Unit Tests for Attention Gate Module.

Tests the AttentionGate and MultiScaleAttentionGate classes
for correct output shapes and attention coefficient properties.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.attention_gate import AttentionGate, MultiScaleAttentionGate


class TestAttentionGate:
    """Tests for AttentionGate module."""

    def test_attention_gate_output_shape(self):
        """Test that attention gate output has same shape as input features."""
        ag = AttentionGate(F_g=256, F_l=512, F_int=256)

        g = torch.randn(2, 256, 32, 32)  # Gating signal
        x = torch.randn(2, 512, 32, 32)  # Encoder features

        output = ag(g, x)

        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_attention_gate_different_spatial_sizes(self):
        """Test attention gate with different spatial sizes for g and x."""
        ag = AttentionGate(F_g=256, F_l=512, F_int=128)

        # g has smaller spatial size (common in U-Net decoders)
        g = torch.randn(2, 256, 16, 16)
        x = torch.randn(2, 512, 32, 32)

        output = ag(g, x)

        # Output should match x's spatial size
        assert output.shape == x.shape

    def test_attention_gate_single_sample(self):
        """Test with batch size of 1."""
        ag = AttentionGate(F_g=128, F_l=256, F_int=64)

        g = torch.randn(1, 128, 64, 64)
        x = torch.randn(1, 256, 64, 64)

        output = ag(g, x)

        assert output.shape == (1, 256, 64, 64)

    def test_attention_gate_large_batch(self):
        """Test with larger batch size."""
        ag = AttentionGate(F_g=256, F_l=512, F_int=256)

        g = torch.randn(16, 256, 32, 32)
        x = torch.randn(16, 512, 32, 32)

        output = ag(g, x)

        assert output.shape == (16, 512, 32, 32)

    def test_attention_coefficients_range(self):
        """Test that attention coefficients are in [0, 1] range."""
        ag = AttentionGate(F_g=256, F_l=512, F_int=256)

        g = torch.randn(2, 256, 32, 32)
        x = torch.randn(2, 512, 32, 32)

        alpha = ag.get_attention_map(g, x)

        assert alpha.min() >= 0.0, "Attention coefficients should be >= 0"
        assert alpha.max() <= 1.0, "Attention coefficients should be <= 1"

    def test_attention_map_shape(self):
        """Test attention map output shape."""
        ag = AttentionGate(F_g=256, F_l=512, F_int=256)

        g = torch.randn(4, 256, 32, 32)
        x = torch.randn(4, 512, 32, 32)

        alpha = ag.get_attention_map(g, x)

        # Attention map should be (B, 1, H, W)
        assert alpha.shape == (4, 1, 32, 32)

    def test_attention_gate_gradient_flow(self):
        """Test that gradients flow through attention gate."""
        ag = AttentionGate(F_g=256, F_l=512, F_int=256)

        g = torch.randn(2, 256, 32, 32, requires_grad=True)
        x = torch.randn(2, 512, 32, 32, requires_grad=True)

        output = ag(g, x)
        loss = output.sum()
        loss.backward()

        assert g.grad is not None, "Gradient should flow to g"
        assert x.grad is not None, "Gradient should flow to x"

    def test_attention_gate_without_bn(self):
        """Test attention gate without batch normalization."""
        ag = AttentionGate(F_g=256, F_l=512, F_int=256, use_bn=False)

        g = torch.randn(2, 256, 32, 32)
        x = torch.randn(2, 512, 32, 32)

        output = ag(g, x)

        assert output.shape == x.shape

    def test_attention_gate_small_channels(self):
        """Test with small channel counts."""
        ag = AttentionGate(F_g=32, F_l=64, F_int=16)

        g = torch.randn(2, 32, 128, 128)
        x = torch.randn(2, 64, 128, 128)

        output = ag(g, x)

        assert output.shape == (2, 64, 128, 128)


class TestMultiScaleAttentionGate:
    """Tests for MultiScaleAttentionGate module."""

    def test_multiscale_output_shape(self):
        """Test multi-scale attention gate output shape."""
        msag = MultiScaleAttentionGate(F_g=256, F_l=512, F_int=256)

        g = torch.randn(2, 256, 32, 32)
        x = torch.randn(2, 512, 32, 32)

        output = msag(g, x)

        assert output.shape == x.shape

    def test_multiscale_custom_scales(self):
        """Test with custom scale factors."""
        msag = MultiScaleAttentionGate(
            F_g=256, F_l=512, F_int=256,
            scales=[1, 2]
        )

        g = torch.randn(2, 256, 32, 32)
        x = torch.randn(2, 512, 32, 32)

        output = msag(g, x)

        assert output.shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
