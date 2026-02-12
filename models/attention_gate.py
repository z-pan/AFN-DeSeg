"""
Attention Gate Module for Attention U-Net.

Implements the attention mechanism for filtering encoder features
in skip connections, highlighting diagnostically relevant regions.

Reference:
    Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
    MIDL 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionGate(nn.Module):
    """
    Attention Gate for filtering encoder features in skip connections.

    The attention gate learns to suppress irrelevant regions while
    highlighting salient features useful for the task. It computes
    attention coefficients using additive attention mechanism.

    Mathematical formulation:
        alpha = sigmoid(psi(ReLU(W_x(x) + W_g(g))))
        output = x * alpha

    Args:
        F_g: Number of channels in gating signal (decoder features).
        F_l: Number of channels in encoder features (skip connection).
        F_int: Number of intermediate attention channels.
        use_bn: Whether to use batch normalization (default: True).

    Example:
        >>> ag = AttentionGate(F_g=256, F_l=512, F_int=256)
        >>> g = torch.randn(2, 256, 32, 32)  # Gating signal
        >>> x = torch.randn(2, 512, 32, 32)  # Encoder features
        >>> output = ag(g, x)
        >>> output.shape
        torch.Size([2, 512, 32, 32])
    """

    def __init__(
        self,
        F_g: int,
        F_l: int,
        F_int: int,
        use_bn: bool = True
    ):
        super(AttentionGate, self).__init__()

        self.F_g = F_g
        self.F_l = F_l
        self.F_int = F_int

        # Transform gating signal to intermediate dimension
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int) if use_bn else nn.Identity()
        )

        # Transform encoder features to intermediate dimension
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int) if use_bn else nn.Identity()
        )

        # Compute attention coefficients (reduce to single channel)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1) if use_bn else nn.Identity(),
            nn.Sigmoid()
        )

        # ReLU activation for combined features
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention gating to encoder features.

        Args:
            g: Gating signal from decoder, shape (B, F_g, H_g, W_g).
               Usually from a coarser resolution level.
            x: Encoder features from skip connection, shape (B, F_l, H_x, W_x).
               Usually at the same resolution as the decoder output.

        Returns:
            Attention-weighted encoder features, shape (B, F_l, H_x, W_x).
            Features are scaled by learned attention coefficients.

        Note:
            If g and x have different spatial dimensions, g is upsampled
            to match x's resolution before computing attention.
        """
        # Get spatial dimensions
        _, _, H_x, W_x = x.shape
        _, _, H_g, W_g = g.shape

        # Transform gating signal
        g_conv = self.W_g(g)

        # Upsample gating signal if needed to match encoder features
        if H_g != H_x or W_g != W_x:
            g_conv = F.interpolate(
                g_conv,
                size=(H_x, W_x),
                mode='bilinear',
                align_corners=True
            )

        # Transform encoder features
        x_conv = self.W_x(x)

        # Additive attention: combine gating and encoder features
        combined = self.relu(g_conv + x_conv)

        # Compute attention coefficients (values in [0, 1])
        alpha = self.psi(combined)

        # Apply attention: element-wise multiplication
        # Attention coefficient is broadcast across channels
        output = x * alpha

        return output

    def get_attention_map(
        self,
        g: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention coefficients without applying them.

        Useful for visualization and interpretability.

        Args:
            g: Gating signal from decoder.
            x: Encoder features from skip connection.

        Returns:
            Attention coefficients, shape (B, 1, H_x, W_x).
        """
        _, _, H_x, W_x = x.shape
        _, _, H_g, W_g = g.shape

        g_conv = self.W_g(g)

        if H_g != H_x or W_g != W_x:
            g_conv = F.interpolate(
                g_conv,
                size=(H_x, W_x),
                mode='bilinear',
                align_corners=True
            )

        x_conv = self.W_x(x)
        combined = self.relu(g_conv + x_conv)
        alpha = self.psi(combined)

        return alpha


class MultiScaleAttentionGate(nn.Module):
    """
    Multi-scale Attention Gate with parallel attention at different scales.

    Computes attention at multiple scales and combines them for
    more robust feature selection.

    Args:
        F_g: Number of channels in gating signal.
        F_l: Number of channels in encoder features.
        F_int: Number of intermediate attention channels.
        scales: List of scale factors for multi-scale attention.
    """

    def __init__(
        self,
        F_g: int,
        F_l: int,
        F_int: int,
        scales: list = [1, 2, 4]
    ):
        super(MultiScaleAttentionGate, self).__init__()

        self.scales = scales
        self.attention_gates = nn.ModuleList([
            AttentionGate(F_g, F_l, F_int // len(scales))
            for _ in scales
        ])

        # Combine multi-scale attention
        self.combine = nn.Conv2d(len(scales), 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply multi-scale attention gating.

        Args:
            g: Gating signal from decoder.
            x: Encoder features from skip connection.

        Returns:
            Attention-weighted encoder features.
        """
        _, _, H, W = x.shape
        attention_maps = []

        for scale, ag in zip(self.scales, self.attention_gates):
            if scale > 1:
                # Downsample for coarser scale
                x_scaled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
                g_scaled = F.avg_pool2d(g, kernel_size=scale, stride=scale)
                alpha = ag.get_attention_map(g_scaled, x_scaled)
                # Upsample attention map back
                alpha = F.interpolate(alpha, size=(H, W), mode='bilinear', align_corners=True)
            else:
                alpha = ag.get_attention_map(g, x)

            attention_maps.append(alpha)

        # Combine attention maps from all scales
        combined_attention = torch.cat(attention_maps, dim=1)
        final_attention = self.sigmoid(self.combine(combined_attention))

        return x * final_attention
