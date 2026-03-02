"""
Joint Loss Functions for AFN-DeSeg.

Based on section 4.2.2 from the AFN-DeSeg paper.

Total Loss:
    L_total = λ_rec * L_rec + λ_seg * L_seg + λ_percep * L_percep

Where:
    - L_rec: Reconstruction loss (L1/MAE)
    - L_seg: Segmentation loss (Dice + BCE)
    - L_percep: Perceptual loss (MSE on Cellpose features)

Default weights: λ_rec=1.0, λ_seg=10.0, λ_percep=0.1
"""

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Reconstruction Loss
# =============================================================================

class ReconstructionLoss(nn.Module):
    """
    Reconstruction Loss using L1 (Mean Absolute Error).

    Measures pixel-wise fidelity between predicted denoised image
    and ground truth clean image.
    """

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L1 reconstruction loss.

        Args:
            pred: Predicted denoised image (B, 1, H, W).
            target: Ground truth clean image (B, 1, H, W).

        Returns:
            Scalar L1 loss value.
        """
        return self.l1_loss(pred, target)


# =============================================================================
# Segmentation Loss
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice

    Addresses class imbalance for sparse nuclear targets.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred: Predicted probability map (B, 1, H, W), values in [0, 1].
            target: Ground truth binary mask (B, 1, H, W).

        Returns:
            Scalar Dice loss value.
        """
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return loss (1 - dice), averaged over batch
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    Combined Segmentation Loss: Dice Loss + Binary Cross Entropy.

    This combination addresses class imbalance in sparse nuclear targets
    while maintaining pixel-level accuracy.
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        """
        Args:
            dice_weight: Weight for Dice loss component.
            bce_weight: Weight for BCE loss component.
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined segmentation loss.

        Args:
            pred: Predicted probability map (B, 1, H, W), values in [0, 1].
            target: Ground truth binary mask (B, 1, H, W).

        Returns:
            Tuple of (total_loss, dice_loss, bce_loss).
        """
        dice = self.dice_loss(pred, target)
        # PyTorch 2.x blocks binary_cross_entropy inside any autocast context at
        # the dispatcher level (regardless of tensor dtype).  Explicitly disable
        # autocast for this single op; the model already applies sigmoid so we
        # cannot use BCEWithLogitsLoss without also changing the architecture.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            bce = F.binary_cross_entropy(pred.float(), target.float())

        total = self.dice_weight * dice + self.bce_weight * bce

        return total, dice, bce


# =============================================================================
# Perceptual Loss with Cellpose
# =============================================================================

class CellposeFeatureExtractor(nn.Module):
    """
    Feature extractor using frozen Cellpose cyto2 model.

    Extracts features from the bottleneck and first up-sampling layer
    of the Cellpose CPnet for perceptual loss computation.

    Note: Inputs must be z-score normalized to match cyto2 training distribution.
    """

    def __init__(self, model_type: str = 'cyto2', device: Optional[str] = None):
        """
        Initialize Cellpose feature extractor.

        Args:
            model_type: Cellpose model type (default: 'cyto2').
            device: Device to load model on. If None, uses CUDA if available.
        """
        super().__init__()

        self.model_type = model_type
        self._device = device

        # Lazy initialization flag
        self._initialized = False
        self._cpnet = None
        self._incompatible = False   # True when Cellpose ≥4.0 SAM backbone detected

        # Feature hooks storage
        self._features: Dict[str, torch.Tensor] = {}

    def _ensure_initialized(self, device: torch.device):
        """Lazy initialization of Cellpose model."""
        if self._initialized:
            return

        try:
            from cellpose import models
        except ImportError:
            raise ImportError(
                "Cellpose is required for perceptual loss. "
                "Install with: pip install cellpose"
            )

        # Load Cellpose model
        cp_model = models.CellposeModel(
            model_type=self.model_type,
            device=device
        )

        # Get the CPnet (the neural network component)
        self._cpnet = cp_model.net

        # Cellpose ≥4.0 replaced CPnet with a SAM-based ViT backbone.
        # That architecture has no downsample/upsample blocks, so our
        # feature hooks cannot be registered.  Mark as incompatible and
        # skip hook registration; forward() will return an empty dict,
        # making the perceptual loss contribution zero for this run.
        if not (hasattr(self._cpnet, 'downsample') and hasattr(self._cpnet, 'upsample')):
            print(
                "[PerceptualLoss] Warning: Cellpose ≥4.0 SAM backbone detected. "
                "Feature hooks are not supported in this version. "
                "Cellpose perceptual loss will be zero for this training run. "
                "Pass --no_cellpose to use the placeholder perceptual loss instead."
            )
            self._incompatible = True
            self._initialized = True
            return

        # Freeze all weights
        for param in self._cpnet.parameters():
            param.requires_grad = False

        # Set to evaluation mode
        self._cpnet.eval()

        # Register hooks for feature extraction
        self._register_hooks()

        self._initialized = True

    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""
        # The Cellpose CPnet architecture has:
        # - downsample blocks (encoder)
        # - upsample blocks (decoder)
        # We need bottleneck (end of encoder) and first upsample layer

        def get_hook(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook

        # Hook into the bottleneck (last downsample output)
        # In CPnet, this is typically after the encoder before upsampling
        if hasattr(self._cpnet, 'downsample'):
            # Get the last downsample block for bottleneck features
            num_down = len(self._cpnet.downsample)
            if num_down > 0:
                self._cpnet.downsample[-1].register_forward_hook(
                    get_hook('bottleneck')
                )

        # Hook into the first upsample layer
        if hasattr(self._cpnet, 'upsample'):
            if len(self._cpnet.upsample) > 0:
                self._cpnet.upsample[0].register_forward_hook(
                    get_hook('upsample1')
                )

    def _z_score_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score normalization to match Cellpose training distribution.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Z-score normalized tensor.
        """
        # Compute per-image statistics
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-8
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from Cellpose network.

        Args:
            x: Input image tensor (B, C, H, W).
               Should be single channel or will be averaged.

        Returns:
            Dictionary with 'bottleneck' and 'upsample1' features.
        """
        device = x.device
        self._ensure_initialized(device)

        # SAM-based Cellpose 4.x: hooks unavailable, return empty dict so
        # PerceptualLoss.forward produces zero loss without crashing.
        if self._incompatible:
            return {}

        # Clear previous features
        self._features = {}

        # Ensure input is properly formatted for Cellpose
        # Cellpose expects (B, C, H, W) with C=2 for cyto2 (typically)
        if x.shape[1] == 1:
            # Duplicate channel for cyto2 input format
            x = x.repeat(1, 2, 1, 1)
        elif x.shape[1] == 3:
            # Convert RGB to 2-channel (average RGB for one channel)
            x = torch.stack([x.mean(dim=1), x.mean(dim=1)], dim=1)

        # Z-score normalize
        x_norm = self._z_score_normalize(x)

        # Forward pass through Cellpose (no grad needed)
        with torch.no_grad():
            _ = self._cpnet(x_norm)

        return self._features.copy()


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using frozen Cellpose cyto2 features.

    Computes MSE between feature maps of predicted and ground truth images
    extracted from the bottleneck and first up-sampling layer of Cellpose.

    As per paper: "Features are extracted from the bottleneck and the first
    up-sampling layer. Inputs to the perceptual network are z-score normalized
    to match the cyto2 training distribution."
    """

    def __init__(
        self,
        model_type: str = 'cyto2',
        feature_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            model_type: Cellpose model type.
            feature_weights: Optional weights for each feature layer.
                           Default: equal weights for bottleneck and upsample1.
        """
        super().__init__()

        self.feature_extractor = CellposeFeatureExtractor(model_type=model_type)

        if feature_weights is None:
            self.feature_weights = {
                'bottleneck': 1.0,
                'upsample1': 1.0
            }
        else:
            self.feature_weights = feature_weights

        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted denoised image (B, 1, H, W).
            target: Ground truth clean image (B, 1, H, W).

        Returns:
            Tuple of (total_loss, dict of per-layer losses).
        """
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        # Compute MSE for each feature layer
        layer_losses = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        for layer_name, weight in self.feature_weights.items():
            if layer_name in pred_features and layer_name in target_features:
                layer_loss = self.mse_loss(
                    pred_features[layer_name],
                    target_features[layer_name]
                )
                layer_losses[layer_name] = layer_loss
                total_loss = total_loss + weight * layer_loss

        return total_loss, layer_losses


class PerceptualLossPlaceholder(nn.Module):
    """
    Placeholder Perceptual Loss for when Cellpose is not available.

    Uses a simple VGG-style feature extraction as fallback.
    This allows training without Cellpose dependency.
    """

    def __init__(self):
        super().__init__()

        # Simple feature extractor mimicking early conv layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Freeze weights
        for param in self.features.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute placeholder perceptual loss.

        Args:
            pred: Predicted denoised image (B, 1, H, W).
            target: Ground truth clean image (B, 1, H, W).

        Returns:
            Tuple of (total_loss, empty dict).
        """
        with torch.no_grad():
            pred_feat = self.features(pred)
            target_feat = self.features(target)

        loss = self.mse_loss(pred_feat, target_feat)
        return loss, {'placeholder': loss}


# =============================================================================
# Joint Loss (Main Class)
# =============================================================================

class AFNJointLoss(nn.Module):
    """
    AFN-DeSeg Joint Loss Function.

    Combines reconstruction, segmentation, and perceptual losses for
    joint optimization of denoising and segmentation tasks.

    Total Loss:
        L_total = λ_rec * L_rec + λ_seg * L_seg + λ_percep * L_percep

    Where:
        - L_rec = L1(denoised_pred, clean_gt)
        - L_seg = Dice(seg_pred, seg_gt) + BCE(seg_pred, seg_gt)
        - L_percep = MSE(Cellpose_features(denoised_pred), Cellpose_features(clean_gt))

    Default weights (from paper):
        - λ_rec = 1.0
        - λ_seg = 10.0 (higher to prevent restoration from dominating)
        - λ_percep = 0.1 (soft constraint to suppress artifacts)

    Args:
        lambda_rec: Weight for reconstruction loss.
        lambda_seg: Weight for segmentation loss.
        lambda_percep: Weight for perceptual loss.
        use_cellpose: Whether to use Cellpose for perceptual loss.
                     If False, uses a placeholder network.

    Example:
        >>> loss_fn = AFNJointLoss()
        >>> denoised_pred = torch.rand(2, 1, 512, 512)
        >>> seg_pred = torch.sigmoid(torch.randn(2, 1, 512, 512))
        >>> clean_gt = torch.rand(2, 1, 512, 512)
        >>> seg_gt = torch.randint(0, 2, (2, 1, 512, 512)).float()
        >>> total_loss, loss_dict = loss_fn(denoised_pred, seg_pred, clean_gt, seg_gt)
    """

    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_seg: float = 10.0,
        lambda_percep: float = 0.1,
        use_cellpose: bool = True
    ):
        super().__init__()

        self.lambda_rec = lambda_rec
        self.lambda_seg = lambda_seg
        self.lambda_percep = lambda_percep

        # Loss components
        self.reconstruction_loss = ReconstructionLoss()
        self.segmentation_loss = SegmentationLoss()

        # Perceptual loss (Cellpose or placeholder)
        if use_cellpose:
            try:
                self.perceptual_loss = PerceptualLoss(model_type='cyto2')
                self._use_cellpose = True
            except ImportError:
                print("Warning: Cellpose not available. Using placeholder perceptual loss.")
                self.perceptual_loss = PerceptualLossPlaceholder()
                self._use_cellpose = False
        else:
            self.perceptual_loss = PerceptualLossPlaceholder()
            self._use_cellpose = False

    def forward(
        self,
        denoised_pred: torch.Tensor,
        seg_pred: torch.Tensor,
        clean_gt: torch.Tensor,
        seg_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute joint loss.

        Args:
            denoised_pred: Predicted denoised image (B, 1, H, W).
            seg_pred: Predicted segmentation mask (B, 1, H, W), values in [0, 1].
            clean_gt: Ground truth clean image (B, 1, H, W).
            seg_gt: Ground truth segmentation mask (B, 1, H, W), binary.

        Returns:
            Tuple of:
                - total_loss: Weighted sum of all losses.
                - loss_dict: Dictionary containing individual loss components:
                    - 'loss_rec': Reconstruction loss
                    - 'loss_seg': Total segmentation loss
                    - 'loss_dice': Dice loss component
                    - 'loss_bce': BCE loss component
                    - 'loss_percep': Perceptual loss
                    - 'loss_total': Total weighted loss
        """
        # Reconstruction loss (L1)
        loss_rec = self.reconstruction_loss(denoised_pred, clean_gt)

        # Segmentation loss (Dice + BCE)
        loss_seg, loss_dice, loss_bce = self.segmentation_loss(seg_pred, seg_gt)

        # Perceptual loss (Cellpose features)
        loss_percep, percep_details = self.perceptual_loss(denoised_pred, clean_gt)

        # Total loss with weights
        total_loss = (
            self.lambda_rec * loss_rec +
            self.lambda_seg * loss_seg +
            self.lambda_percep * loss_percep
        )

        # Build loss dictionary for logging
        loss_dict = {
            'loss_rec': loss_rec.detach(),
            'loss_seg': loss_seg.detach(),
            'loss_dice': loss_dice.detach(),
            'loss_bce': loss_bce.detach(),
            'loss_percep': loss_percep.detach(),
            'loss_total': total_loss.detach(),
            # Weighted components (for analysis)
            'weighted_rec': (self.lambda_rec * loss_rec).detach(),
            'weighted_seg': (self.lambda_seg * loss_seg).detach(),
            'weighted_percep': (self.lambda_percep * loss_percep).detach(),
        }

        # Add perceptual layer details
        for layer_name, layer_loss in percep_details.items():
            loss_dict[f'percep_{layer_name}'] = layer_loss.detach()

        return total_loss, loss_dict

    def get_loss_weights(self) -> Dict[str, float]:
        """Return current loss weights."""
        return {
            'lambda_rec': self.lambda_rec,
            'lambda_seg': self.lambda_seg,
            'lambda_percep': self.lambda_percep
        }

    def set_loss_weights(
        self,
        lambda_rec: Optional[float] = None,
        lambda_seg: Optional[float] = None,
        lambda_percep: Optional[float] = None
    ):
        """Update loss weights dynamically."""
        if lambda_rec is not None:
            self.lambda_rec = lambda_rec
        if lambda_seg is not None:
            self.lambda_seg = lambda_seg
        if lambda_percep is not None:
            self.lambda_percep = lambda_percep


def create_joint_loss(
    lambda_rec: float = 1.0,
    lambda_seg: float = 10.0,
    lambda_percep: float = 0.1,
    use_cellpose: bool = True
) -> AFNJointLoss:
    """
    Factory function to create AFN-DeSeg joint loss.

    Args:
        lambda_rec: Weight for reconstruction loss.
        lambda_seg: Weight for segmentation loss.
        lambda_percep: Weight for perceptual loss.
        use_cellpose: Whether to use Cellpose for perceptual loss.

    Returns:
        AFNJointLoss instance.
    """
    return AFNJointLoss(
        lambda_rec=lambda_rec,
        lambda_seg=lambda_seg,
        lambda_percep=lambda_percep,
        use_cellpose=use_cellpose
    )
