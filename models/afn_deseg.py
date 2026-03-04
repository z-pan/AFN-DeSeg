"""
AFN-DeSeg Model Architecture.

Auto-Fluorescence Nuclei Denoising and Segmentation framework implementing
a dual-encoder architecture with DINOv3 Vision Transformer and U-Net.

Based on sections 2.1 and 4.2.1 from the AFN-DeSeg paper.
"""

import math
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# LoRA (Low-Rank Adaptation) Module
# =============================================================================

class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation (LoRA).

    LoRA freezes the pre-trained weights and injects trainable low-rank
    decomposition matrices A and B, such that:
        W' = W + (alpha/r) * B @ A

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        r: Rank of the low-rank decomposition.
        alpha: Scaling factor for LoRA weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Original linear layer (frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features)

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        output = self.linear(x)
        # LoRA adaptation: x @ A^T @ B^T * scaling
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return output + lora_output

    def freeze_pretrained(self):
        """Freeze the original linear layer weights."""
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False


# =============================================================================
# U-Net Encoder Components
# =============================================================================

class ConvBlock(nn.Module):
    """
    Double convolution block for U-Net.

    Each block contains two 3x3 conv layers with BatchNorm and ReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetEncoder(nn.Module):
    """
    U-Net Encoder with 4 down-sampling blocks.

    Architecture:
        - 4 down-sampling blocks
        - Each block: 2x Conv3x3-BN-ReLU + MaxPool2x2
        - Channel progression: 64 -> 128 -> 256 -> 512
        - Output bottleneck: 512 channels at 32x32 (for 512x512 input)

    Args:
        in_channels: Number of input channels (default: 3 for pseudo-RGB).
        base_channels: Base channel count (default: 64).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        self.channels = [
            base_channels,      # 64
            base_channels * 2,  # 128
            base_channels * 4,  # 256
            base_channels * 8   # 512
        ]

        # Initial convolution
        self.init_conv = ConvBlock(in_channels, self.channels[0])

        # Encoder blocks with down-sampling
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = self.channels[0]
        for out_ch in self.channels[1:]:
            self.encoder_blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(self.channels[-1], self.channels[-1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tuple of:
                - bottleneck: Bottleneck features (B, 512, H/16, W/16).
                - skip_connections: List of encoder features for skip connections.
        """
        skip_connections = []

        # Initial conv
        x = self.init_conv(x)
        skip_connections.append(x)

        # Encoder blocks
        for block in self.encoder_blocks:
            x = self.pool(x)
            x = block(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.pool(x)
        x = self.bottleneck(x)

        return x, skip_connections


# =============================================================================
# DINOv3 Vision Transformer Encoder with LoRA
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with LoRA on Query and Value projections.

    Args:
        embed_dim: Embedding dimension (768 for ViT-Base).
        num_heads: Number of attention heads (12 for ViT-Base).
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        lora_r: int = 16,
        lora_alpha: int = 16
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query and Value with LoRA adaptation
        self.q_proj = LoRALinear(embed_dim, embed_dim, r=lora_r, alpha=lora_alpha)
        self.v_proj = LoRALinear(embed_dim, embed_dim, r=lora_r, alpha=lora_alpha)

        # Key without LoRA (standard linear)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with MHSA and MLP.

    Uses LayerNorm and GELU activation as per ViT specification.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        lora_r: int = 16,
        lora_alpha: int = 16
    ):
        super().__init__()

        # Layer Normalization (as per ViT)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention with LoRA
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, lora_r, lora_alpha
        )

        # MLP with GELU activation
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for Vision Transformer.

    Converts image into sequence of patch embeddings.

    Args:
        img_size: Input image size.
        patch_size: Size of each patch.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DINOv3Encoder(nn.Module):
    """
    DINOv3 Vision Transformer Encoder with LoRA adaptation.

    Architecture (ViT-Base):
        - Patch size: 16x16 (32x32 grid for 512x512 input)
        - Hidden dimension: 768
        - 12 transformer layers with MHSA
        - LoRA (r=16, alpha=16) on Query and Value matrices

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        lora_r: int = 16,
        lora_alpha: int = 16
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )

        # Learnable positional embeddings
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # CLS token (optional, can be used for classification tasks)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks with LoRA
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, lora_r=lora_r, lora_alpha=lora_alpha)
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def interpolate_pos_embed(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Interpolate positional embeddings for different input sizes.

        Supports bicubic interpolation as mentioned in the paper.
        """
        num_patches = x.shape[1]
        N = self.pos_embed.shape[1]

        if num_patches == N and h == w == self.grid_size:
            return self.pos_embed

        # Reshape and interpolate
        pos_embed = self.pos_embed.reshape(1, self.grid_size, self.grid_size, -1)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # (1, C, H, W)
        pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)

        return pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv3 encoder.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Patch tokens of shape (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size

        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding (with interpolation if needed)
        pos_embed = self.interpolate_pos_embed(x, grid_h, grid_w)
        x = x + pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x

    def freeze_pretrained(self):
        """Freeze all weights except LoRA parameters."""
        for name, param in self.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                param.requires_grad = False


# =============================================================================
# Feature Fusion Module
# =============================================================================

class FusionBlock(nn.Module):
    """
    Feature Fusion Block.

    Combines U-Net bottleneck features with DINOv3 patch tokens:
        1. Reshape ViT tokens to spatial format
        2. Bicubic upsample to match U-Net resolution
        3. Concatenate along channel dimension
        4. Fuse via 1x1 conv + 3x3 conv

    Args:
        unet_channels: U-Net bottleneck channels (512).
        vit_channels: ViT embedding dimension (768).
        out_channels: Output channels after fusion (512).
    """

    def __init__(
        self,
        unet_channels: int = 512,
        vit_channels: int = 768,
        out_channels: int = 512
    ):
        super().__init__()

        combined_channels = unet_channels + vit_channels  # 1280

        # Fusion bottleneck: 1x1 conv followed by 3x3 conv
        self.fusion = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        unet_features: torch.Tensor,
        vit_tokens: torch.Tensor,
        grid_size: int
    ) -> torch.Tensor:
        """
        Fuse U-Net and ViT features.

        Args:
            unet_features: U-Net bottleneck features (B, 512, H, W).
            vit_tokens: ViT patch tokens (B, num_patches, 768).
            grid_size: ViT grid size (e.g., 32 for 512x512 input with patch 16).

        Returns:
            Fused features (B, out_channels, H, W).
        """
        B = unet_features.shape[0]
        H, W = unet_features.shape[2:]

        # Reshape ViT tokens to spatial format: (B, N, C) -> (B, C, grid, grid)
        vit_spatial = vit_tokens.transpose(1, 2).reshape(B, -1, grid_size, grid_size)

        # Bicubic upsample to match U-Net bottleneck resolution
        if vit_spatial.shape[2:] != (H, W):
            vit_spatial = F.interpolate(
                vit_spatial, size=(H, W), mode='bicubic', align_corners=False
            )

        # Concatenate along channel dimension
        combined = torch.cat([unet_features, vit_spatial], dim=1)

        # Apply fusion block
        fused = self.fusion(combined)

        return fused


# =============================================================================
# Decoder Components
# =============================================================================

class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connection.

    Args:
        in_channels: Input channels.
        skip_channels: Skip connection channels.
        out_channels: Output channels.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        # Bilinear upsampling followed by convolution
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Convolution after concatenation with skip connection
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DenoisingDecoder(nn.Module):
    """
    Denoising Decoder.

    Outputs single-channel continuous intensity map for image reconstruction.

    Args:
        in_channels: Input channels from fusion block.
        encoder_channels: List of encoder channel counts for skip connections.
    """

    def __init__(
        self,
        in_channels: int = 512,
        encoder_channels: List[int] = None
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [512, 256, 128, 64]  # From bottleneck to first

        self.decoder_blocks = nn.ModuleList()
        current_ch = in_channels

        # Build decoder blocks (reverse of encoder)
        for i, skip_ch in enumerate(encoder_channels):
            out_ch = skip_ch if i < len(encoder_channels) - 1 else encoder_channels[-1]
            self.decoder_blocks.append(DecoderBlock(current_ch, skip_ch, out_ch))
            current_ch = out_ch

        # Final output layer (single channel intensity map, sigmoid for [0,1] range)
        self.output = nn.Sequential(
            nn.Conv2d(current_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through denoising decoder.

        Args:
            x: Fused features from fusion block.
            skip_connections: List of encoder features (in order from shallow to deep).

        Returns:
            Denoised intensity map (B, 1, H, W).
        """
        # Reverse skip connections (deep to shallow)
        skips = skip_connections[::-1]

        for i, (block, skip) in enumerate(zip(self.decoder_blocks, skips)):
            x = block(x, skip)

        return self.output(x)


class SegmentationDecoder(nn.Module):
    """
    Segmentation Decoder.

    Outputs binary segmentation mask with sigmoid activation.

    Args:
        in_channels: Input channels from fusion block.
        encoder_channels: List of encoder channel counts for skip connections.
    """

    def __init__(
        self,
        in_channels: int = 512,
        encoder_channels: List[int] = None
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [512, 256, 128, 64]

        self.decoder_blocks = nn.ModuleList()
        current_ch = in_channels

        # Build decoder blocks
        for i, skip_ch in enumerate(encoder_channels):
            out_ch = skip_ch if i < len(encoder_channels) - 1 else encoder_channels[-1]
            self.decoder_blocks.append(DecoderBlock(current_ch, skip_ch, out_ch))
            current_ch = out_ch

        # Final output layer with sigmoid (binary mask)
        self.output = nn.Sequential(
            nn.Conv2d(current_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through segmentation decoder.

        Args:
            x: Fused features from fusion block.
            skip_connections: List of encoder features.

        Returns:
            Segmentation probability map (B, 1, H, W).
        """
        skips = skip_connections[::-1]

        for block, skip in zip(self.decoder_blocks, skips):
            x = block(x, skip)

        return self.output(x)


# =============================================================================
# Main AFN-DeSeg Model
# =============================================================================

class AFNDeSeg(nn.Module):
    """
    AFN-DeSeg: Auto-Fluorescence Nuclei Denoising and Segmentation.

    A dual-encoder architecture combining U-Net and DINOv3 Vision Transformer
    for joint denoising and segmentation of TPAF microscopy images.

    Architecture:
        - Dual Encoders: U-Net (local features) + DINOv3 ViT (global semantic)
        - Feature Fusion: Concatenate and fuse encoder outputs
        - Dual Decoders: Separate denoising and segmentation branches

    Input: Single-channel TPAF image (replicated to 3 channels, normalized)
    Output: Denoised intensity map + Binary segmentation mask

    Args:
        img_size: Input image size (default: 512).
        in_channels: Input channels (default: 3 for pseudo-RGB).
        base_channels: U-Net base channels (default: 64).
        vit_embed_dim: ViT embedding dimension (default: 768).
        vit_depth: Number of ViT layers (default: 12).
        vit_num_heads: Number of ViT attention heads (default: 12).
        vit_patch_size: ViT patch size (default: 16).
        lora_r: LoRA rank (default: 16).
        lora_alpha: LoRA alpha (default: 16).

    Example:
        >>> model = AFNDeSeg(img_size=512)
        >>> x = torch.randn(1, 3, 512, 512)
        >>> denoised, segmented = model(x)
        >>> print(denoised.shape, segmented.shape)
        torch.Size([1, 1, 512, 512]) torch.Size([1, 1, 512, 512])
    """

    # ImageNet normalization statistics (for DINOv3 compatibility)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        img_size: int = 512,
        in_channels: int = 3,
        base_channels: int = 64,
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_num_heads: int = 12,
        vit_patch_size: int = 16,
        lora_r: int = 16,
        lora_alpha: int = 16
    ):
        super().__init__()

        self.img_size = img_size
        self.vit_patch_size = vit_patch_size
        self.vit_grid_size = img_size // vit_patch_size

        # U-Net Encoder
        self.unet_encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels
        )

        # DINOv3 Encoder with LoRA
        self.vit_encoder = DINOv3Encoder(
            img_size=img_size,
            patch_size=vit_patch_size,
            in_channels=in_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        # Feature Fusion
        unet_bottleneck_channels = base_channels * 8  # 512
        self.fusion = FusionBlock(
            unet_channels=unet_bottleneck_channels,
            vit_channels=vit_embed_dim,
            out_channels=unet_bottleneck_channels
        )

        # Encoder channel list for skip connections
        encoder_channels = [
            base_channels * 8,  # 512 (deepest)
            base_channels * 4,  # 256
            base_channels * 2,  # 128
            base_channels       # 64 (shallowest)
        ]

        # Dual Decoders
        self.denoising_decoder = DenoisingDecoder(
            in_channels=unet_bottleneck_channels,
            encoder_channels=encoder_channels
        )

        self.segmentation_decoder = SegmentationDecoder(
            in_channels=unet_bottleneck_channels,
            encoder_channels=encoder_channels
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize decoder weights using Kaiming initialization."""
        for module in [self.denoising_decoder, self.segmentation_decoder, self.fusion]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through AFN-DeSeg.

        Args:
            x: Input tensor of shape (B, C, H, W).
               For single-channel TPAF, replicate to 3 channels before input.

        Returns:
            Tuple of:
                - denoised: Denoised intensity map (B, 1, H, W).
                - segmented: Segmentation probability map (B, 1, H, W).
        """
        # U-Net encoder
        unet_bottleneck, skip_connections = self.unet_encoder(x)

        # DINOv3 encoder
        vit_tokens = self.vit_encoder(x)

        # Feature fusion
        fused = self.fusion(unet_bottleneck, vit_tokens, self.vit_grid_size)

        # Dual decoders
        denoised = self.denoising_decoder(fused, skip_connections)
        segmented = self.segmentation_decoder(fused, skip_connections)

        return denoised, segmented

    def freeze_vit_pretrained(self):
        """Freeze ViT encoder except LoRA parameters."""
        self.vit_encoder.freeze_pretrained()

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (useful for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_lora_params(self) -> List[nn.Parameter]:
        """Get LoRA parameters only."""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_params.append(param)
        return lora_params

    @staticmethod
    def preprocess_tpaf(
        image: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess single-channel TPAF image for model input.

        Replicates to 3 channels and applies ImageNet normalization.

        Args:
            image: Single-channel image (B, 1, H, W) or (B, H, W).
            normalize: Whether to apply ImageNet normalization.

        Returns:
            Preprocessed image (B, 3, H, W).
        """
        # Ensure 4D tensor
        if image.dim() == 3:
            image = image.unsqueeze(1)

        # Replicate to 3 channels
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Apply ImageNet normalization
        if normalize:
            mean = torch.tensor(AFNDeSeg.IMAGENET_MEAN, device=image.device).view(1, 3, 1, 1)
            std = torch.tensor(AFNDeSeg.IMAGENET_STD, device=image.device).view(1, 3, 1, 1)
            image = (image - mean) / std

        return image


def load_dinov3_weights(
    encoder: DINOv3Encoder,
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    device: str = "cpu"
) -> DINOv3Encoder:
    """
    Load pretrained DINOv3 weights from HuggingFace.

    Args:
        encoder: DINOv3Encoder instance to load weights into.
        model_name: HuggingFace model identifier.
                   Default: "facebook/dinov3-vitb16-pretrain-lvd1689m"
        device: Device to load weights on.

    Returns:
        Encoder with pretrained weights loaded.

    Note:
        This function maps DINOv3 weights to our custom encoder architecture.
        LoRA parameters are initialized fresh (not from pretrained).
    """
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "transformers library required for loading HuggingFace weights. "
            "Install with: pip install transformers"
        )

    print(f"Loading pretrained weights from {model_name}...")

    # Load pretrained model from HuggingFace
    pretrained = AutoModel.from_pretrained(model_name)
    pretrained_state = pretrained.state_dict()

    # Create mapping from HuggingFace keys to our encoder keys
    encoder_state = encoder.state_dict()
    new_state = {}

    for key in encoder_state.keys():
        # Skip LoRA parameters (these should be trained fresh)
        if 'lora_A' in key or 'lora_B' in key:
            new_state[key] = encoder_state[key]
            continue

        # Map patch embedding
        if 'patch_embed.proj' in key:
            hf_key = key.replace('patch_embed.proj', 'embeddings.patch_embeddings.projection')
            if hf_key in pretrained_state:
                new_state[key] = pretrained_state[hf_key]
            else:
                new_state[key] = encoder_state[key]

        # Map positional embeddings
        elif 'pos_embed' in key:
            if 'embeddings.position_embeddings' in pretrained_state:
                # May need interpolation if sizes differ
                pretrained_pos = pretrained_state['embeddings.position_embeddings']
                if pretrained_pos.shape == encoder_state[key].shape:
                    new_state[key] = pretrained_pos
                else:
                    # Interpolate positional embeddings
                    new_state[key] = _interpolate_pos_embed(
                        pretrained_pos, encoder_state[key].shape
                    )
            else:
                new_state[key] = encoder_state[key]

        # Map transformer blocks
        elif 'blocks.' in key:
            block_idx = int(key.split('.')[1])
            hf_prefix = f'encoder.layer.{block_idx}'

            # Map attention layers
            if '.attn.q_proj.linear' in key:
                hf_key = f'{hf_prefix}.attention.attention.query'
                if hf_key + '.weight' in pretrained_state:
                    if 'weight' in key:
                        new_state[key] = pretrained_state[hf_key + '.weight']
                    elif 'bias' in key:
                        new_state[key] = pretrained_state[hf_key + '.bias']
                else:
                    new_state[key] = encoder_state[key]

            elif '.attn.k_proj' in key:
                hf_key = f'{hf_prefix}.attention.attention.key'
                if hf_key + '.weight' in pretrained_state:
                    if 'weight' in key:
                        new_state[key] = pretrained_state[hf_key + '.weight']
                    elif 'bias' in key:
                        new_state[key] = pretrained_state[hf_key + '.bias']
                else:
                    new_state[key] = encoder_state[key]

            elif '.attn.v_proj.linear' in key:
                hf_key = f'{hf_prefix}.attention.attention.value'
                if hf_key + '.weight' in pretrained_state:
                    if 'weight' in key:
                        new_state[key] = pretrained_state[hf_key + '.weight']
                    elif 'bias' in key:
                        new_state[key] = pretrained_state[hf_key + '.bias']
                else:
                    new_state[key] = encoder_state[key]

            elif '.attn.out_proj' in key:
                hf_key = f'{hf_prefix}.attention.output.dense'
                if hf_key + '.weight' in pretrained_state:
                    if 'weight' in key:
                        new_state[key] = pretrained_state[hf_key + '.weight']
                    elif 'bias' in key:
                        new_state[key] = pretrained_state[hf_key + '.bias']
                else:
                    new_state[key] = encoder_state[key]

            # Map layer norms
            elif '.norm1' in key:
                hf_key = f'{hf_prefix}.layernorm_before'
                if hf_key + '.weight' in pretrained_state:
                    if 'weight' in key:
                        new_state[key] = pretrained_state[hf_key + '.weight']
                    elif 'bias' in key:
                        new_state[key] = pretrained_state[hf_key + '.bias']
                else:
                    new_state[key] = encoder_state[key]

            elif '.norm2' in key:
                hf_key = f'{hf_prefix}.layernorm_after'
                if hf_key + '.weight' in pretrained_state:
                    if 'weight' in key:
                        new_state[key] = pretrained_state[hf_key + '.weight']
                    elif 'bias' in key:
                        new_state[key] = pretrained_state[hf_key + '.bias']
                else:
                    new_state[key] = encoder_state[key]

            # Map MLP
            elif '.mlp.0' in key:  # First linear
                hf_key = f'{hf_prefix}.intermediate.dense'
                if hf_key + '.weight' in pretrained_state:
                    if 'weight' in key:
                        new_state[key] = pretrained_state[hf_key + '.weight']
                    elif 'bias' in key:
                        new_state[key] = pretrained_state[hf_key + '.bias']
                else:
                    new_state[key] = encoder_state[key]

            elif '.mlp.2' in key:  # Second linear
                hf_key = f'{hf_prefix}.output.dense'
                if hf_key + '.weight' in pretrained_state:
                    if 'weight' in key:
                        new_state[key] = pretrained_state[hf_key + '.weight']
                    elif 'bias' in key:
                        new_state[key] = pretrained_state[hf_key + '.bias']
                else:
                    new_state[key] = encoder_state[key]

            else:
                new_state[key] = encoder_state[key]

        # Map final layer norm
        elif 'norm.weight' in key or 'norm.bias' in key:
            hf_key = 'layernorm'
            if hf_key + '.weight' in pretrained_state:
                if 'weight' in key:
                    new_state[key] = pretrained_state[hf_key + '.weight']
                elif 'bias' in key:
                    new_state[key] = pretrained_state[hf_key + '.bias']
            else:
                new_state[key] = encoder_state[key]

        else:
            # Keep original for unmapped keys
            new_state[key] = encoder_state[key]

    # Load the mapped state dict
    encoder.load_state_dict(new_state, strict=False)
    print(f"Successfully loaded pretrained weights (LoRA parameters initialized fresh)")

    return encoder


def _interpolate_pos_embed(
    pos_embed: torch.Tensor,
    target_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Interpolate positional embeddings to target shape.

    Args:
        pos_embed: Original positional embeddings.
        target_shape: Target shape.

    Returns:
        Interpolated positional embeddings.
    """
    if pos_embed.shape == target_shape:
        return pos_embed

    # Assume shape is (1, N, D) where N = H*W
    _, N, D = pos_embed.shape
    _, target_N, _ = target_shape

    # Compute grid sizes
    src_size = int(N ** 0.5)
    tgt_size = int(target_N ** 0.5)

    # Reshape to (1, D, H, W)
    pos_embed = pos_embed.transpose(1, 2).reshape(1, D, src_size, src_size)

    # Interpolate
    pos_embed = F.interpolate(
        pos_embed,
        size=(tgt_size, tgt_size),
        mode='bicubic',
        align_corners=False
    )

    # Reshape back to (1, N, D)
    pos_embed = pos_embed.reshape(1, D, -1).transpose(1, 2)

    return pos_embed


def create_afn_deseg(
    img_size: int = 512,
    pretrained_vit: Optional[str] = None,
    pretrained_dinov3: bool = False,
    dinov3_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    **kwargs
) -> AFNDeSeg:
    """
    Factory function to create AFN-DeSeg model.

    Args:
        img_size: Input image size.
        pretrained_vit: Path to local pretrained ViT weights (optional).
        pretrained_dinov3: Whether to load DINOv3 weights from HuggingFace.
        dinov3_model_name: HuggingFace model identifier for DINOv3.
        **kwargs: Additional arguments for AFNDeSeg.

    Returns:
        AFNDeSeg model instance.

    Example:
        >>> # Load with pretrained DINOv3 weights from HuggingFace
        >>> model = create_afn_deseg(
        ...     img_size=512,
        ...     pretrained_dinov3=True,
        ...     dinov3_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m"
        ... )

        >>> # Load with local pretrained weights
        >>> model = create_afn_deseg(
        ...     img_size=512,
        ...     pretrained_vit="/path/to/weights.pth"
        ... )
    """
    model = AFNDeSeg(img_size=img_size, **kwargs)

    # Load pretrained DINOv3 from HuggingFace
    if pretrained_dinov3:
        load_dinov3_weights(
            model.vit_encoder,
            model_name=dinov3_model_name
        )

    # Load local pretrained ViT weights (takes precedence over HuggingFace)
    elif pretrained_vit is not None:
        state_dict = torch.load(pretrained_vit, map_location='cpu')
        # Filter and load only ViT encoder weights
        vit_state_dict = {
            k.replace('vit_encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('vit_encoder.')
        }
        model.vit_encoder.load_state_dict(vit_state_dict, strict=False)
        print(f"Loaded pretrained ViT weights from {pretrained_vit}")

    return model
