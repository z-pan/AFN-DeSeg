"""
Stage 2: Joint Denoising and Segmentation Training for AFN-DeSeg.

Based on section 4.2.1 "Joint Optimization of Denoising and Segmentation Branches".

Training Configuration:
    - Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
    - Scheduler: CosineAnnealingLR (decay to 1e-6)
    - Gradient Clipping: max_norm=1.0
    - Epochs: 150
    - Early Stopping: 20 epochs patience on validation Dice
    - Batch Size: 4

Usage:
    python training/train_stage2_joint.py --data_dir /path/to/data --output_dir /path/to/output
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.afn_deseg import AFNDeSeg
from losses.joint_loss import AFNJointLoss


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image (B, C, H, W).
        target: Ground truth image (B, C, H, W).
        max_val: Maximum pixel value.

    Returns:
        PSNR value in dB.
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice Coefficient.

    Args:
        pred: Predicted probability map (B, 1, H, W).
        target: Ground truth binary mask (B, 1, H, W).
        threshold: Threshold to binarize predictions.

    Returns:
        Dice coefficient value.
    """
    pred_binary = (pred > threshold).float()

    intersection = (pred_binary * target).sum().item()
    union = pred_binary.sum().item() + target.sum().item()

    if union < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0

    dice = (2.0 * intersection) / union
    return dice


# =============================================================================
# Dataset
# =============================================================================

class TPAFDataset(Dataset):
    """
    TPAF Dataset for joint denoising and segmentation.

    Expected directory structure:
        data_dir/
            noisy/      # Noisy TPAF images
            clean/      # Clean reference images (ground truth for denoising)
            masks/      # Segmentation masks (ground truth for segmentation)

    All images should be .npy or .png/.tif files with matching names.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        img_size: int = 512,
        augment: bool = True
    ):
        """
        Args:
            data_dir: Root directory containing noisy/, clean/, masks/ folders.
            split: Dataset split ('train', 'val', 'test').
            img_size: Image size (assumes square images).
            augment: Whether to apply data augmentation.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        # Find image files
        self.noisy_dir = self.data_dir / 'noisy'
        self.clean_dir = self.data_dir / 'clean'
        self.mask_dir = self.data_dir / 'masks'

        # Get list of samples (by filename without extension)
        self.samples = self._get_samples()

    def _get_samples(self):
        """Get list of sample names from noisy directory."""
        samples = []

        if self.noisy_dir.exists():
            for f in self.noisy_dir.iterdir():
                if f.suffix in ['.npy', '.png', '.tif', '.tiff']:
                    samples.append(f.stem)

        return sorted(samples)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from various formats."""
        if path.suffix == '.npy':
            return np.load(path).astype(np.float32)
        else:
            try:
                import tifffile
                if path.suffix in ['.tif', '.tiff']:
                    return tifffile.imread(path).astype(np.float32)
            except ImportError:
                pass

            from skimage import io
            return io.imread(path).astype(np.float32)

    def _find_file(self, directory: Path, name: str) -> Optional[Path]:
        """Find file with given name in directory (any extension)."""
        for ext in ['.npy', '.png', '.tif', '.tiff']:
            path = directory / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1]."""
        if img.max() > 1.0:
            img = img / 255.0 if img.max() <= 255 else img / img.max()
        return img

    def _augment(self, noisy: np.ndarray, clean: np.ndarray, mask: np.ndarray):
        """Apply geometric augmentations (flips and rotations)."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            noisy = np.flip(noisy, axis=-1).copy()
            clean = np.flip(clean, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            noisy = np.flip(noisy, axis=-2).copy()
            clean = np.flip(clean, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            noisy = np.rot90(noisy, k, axes=(-2, -1)).copy()
            clean = np.rot90(clean, k, axes=(-2, -1)).copy()
            mask = np.rot90(mask, k, axes=(-2, -1)).copy()

        return noisy, clean, mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_name = self.samples[idx]

        # Load images
        noisy_path = self._find_file(self.noisy_dir, sample_name)
        clean_path = self._find_file(self.clean_dir, sample_name)
        mask_path = self._find_file(self.mask_dir, sample_name)

        noisy = self._load_image(noisy_path) if noisy_path else np.zeros((self.img_size, self.img_size))
        clean = self._load_image(clean_path) if clean_path else noisy.copy()
        mask = self._load_image(mask_path) if mask_path else np.zeros((self.img_size, self.img_size))

        # Ensure 2D
        if noisy.ndim == 3:
            noisy = noisy.mean(axis=-1) if noisy.shape[-1] <= 4 else noisy[..., 0]
        if clean.ndim == 3:
            clean = clean.mean(axis=-1) if clean.shape[-1] <= 4 else clean[..., 0]
        if mask.ndim == 3:
            mask = mask[..., 0] if mask.shape[-1] <= 4 else mask.mean(axis=-1)

        # Normalize
        noisy = self._normalize(noisy)
        clean = self._normalize(clean)
        mask = (mask > 0.5).astype(np.float32)  # Binary mask

        # Apply augmentation
        if self.augment:
            noisy, clean, mask = self._augment(noisy, clean, mask)

        # Convert to tensors and add channel dimension
        noisy = torch.from_numpy(noisy).unsqueeze(0)  # (1, H, W)
        clean = torch.from_numpy(clean).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)

        return noisy, clean, mask


# =============================================================================
# Training Utilities
# =============================================================================

class EarlyStopping:
    """
    Early stopping handler.

    Stops training if validation metric doesn't improve for `patience` epochs.
    """

    def __init__(self, patience: int = 20, mode: str = 'max', delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait for improvement.
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize.
            delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: AFNDeSeg model.
        dataloader: Training dataloader.
        criterion: AFNJointLoss criterion.
        optimizer: Optimizer.
        device: Training device.
        epoch: Current epoch number.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Dictionary of average training metrics.
    """
    model.train()

    total_loss = 0.0
    loss_components = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for noisy, clean, mask in pbar:
        # Move to device
        noisy = noisy.to(device)
        clean = clean.to(device)
        mask = mask.to(device)

        # Convert to pseudo-RGB and normalize (for DINOv3 compatibility)
        noisy_rgb = AFNDeSeg.preprocess_tpaf(noisy, normalize=True)

        # Forward pass
        denoised, seg_pred = model(noisy_rgb)

        # Compute loss
        loss, loss_dict = criterion(denoised, seg_pred, clean, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # Update weights
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k not in loss_components:
                loss_components[k] = 0.0
            loss_components[k] += v.item()

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    # Average metrics
    metrics = {'train_loss': total_loss / num_batches}
    for k, v in loss_components.items():
        metrics[f'train_{k}'] = v / num_batches

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Validate model.

    Args:
        model: AFNDeSeg model.
        dataloader: Validation dataloader.
        criterion: AFNJointLoss criterion.
        device: Device.
        epoch: Current epoch number.

    Returns:
        Dictionary of validation metrics.
    """
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_psnr = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)

    for noisy, clean, mask in pbar:
        # Move to device
        noisy = noisy.to(device)
        clean = clean.to(device)
        mask = mask.to(device)

        # Convert to pseudo-RGB and normalize
        noisy_rgb = AFNDeSeg.preprocess_tpaf(noisy, normalize=True)

        # Forward pass
        denoised, seg_pred = model(noisy_rgb)

        # Compute loss
        loss, _ = criterion(denoised, seg_pred, clean, mask)

        # Compute metrics
        dice = compute_dice(seg_pred, mask)
        psnr = compute_psnr(denoised, clean)

        # Accumulate
        total_loss += loss.item()
        total_dice += dice
        total_psnr += psnr
        num_batches += 1

        pbar.set_postfix({
            'dice': f"{dice:.4f}",
            'psnr': f"{psnr:.2f}"
        })

    # Average metrics
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_dice': total_dice / num_batches,
        'val_psnr': total_psnr / num_batches
    }

    return metrics


# =============================================================================
# Main Training Function
# =============================================================================

def train(args):
    """Main training function."""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training configuration
    config = vars(args)
    config['device'] = str(device)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize model
    print("Initializing AFN-DeSeg model...")
    model = AFNDeSeg(
        img_size=args.img_size,
        in_channels=3,
        base_channels=64,
        vit_embed_dim=768,
        vit_depth=12,
        vit_num_heads=12,
        vit_patch_size=16,
        lora_r=16,
        lora_alpha=16
    )
    model = model.to(device)

    # Load pretrained weights if provided
    if args.pretrained_vit:
        print(f"Loading pretrained ViT weights from {args.pretrained_vit}")
        state_dict = torch.load(args.pretrained_vit, map_location=device)
        model.vit_encoder.load_state_dict(state_dict, strict=False)

    # Optionally freeze ViT backbone (keep only LoRA trainable)
    if args.freeze_vit:
        print("Freezing ViT backbone (only LoRA parameters trainable)")
        model.freeze_vit_pretrained()

    # Initialize loss function
    print("Initializing joint loss function...")
    criterion = AFNJointLoss(
        lambda_rec=args.lambda_rec,
        lambda_seg=args.lambda_seg,
        lambda_percep=args.lambda_percep,
        use_cellpose=args.use_cellpose
    )

    # Initialize optimizer (AdamW with weight decay)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Initialize scheduler (CosineAnnealing to min_lr)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # Create datasets
    print("Loading datasets...")
    train_dataset = TPAFDataset(
        args.data_dir,
        split='train',
        img_size=args.img_size,
        augment=True
    )
    val_dataset = TPAFDataset(
        args.data_dir,
        split='val',
        img_size=args.img_size,
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    best_dice = 0.0
    start_epoch = 1

    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['metrics'].get('val_dice', 0.0)

    # Training history
    history = []

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Early stopping patience: {args.patience} epochs")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_grad_norm=args.max_grad_norm
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Update scheduler
        scheduler.step()

        # Combine metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        history.append(metrics)

        # Print epoch summary
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {metrics['train_loss']:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | "
              f"Val Dice: {metrics['val_dice']:.4f} | "
              f"Val PSNR: {metrics['val_psnr']:.2f} dB | "
              f"LR: {metrics['lr']:.2e}")

        # Save best model
        if val_metrics['val_dice'] > best_dice:
            best_dice = val_metrics['val_dice']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                str(output_dir / 'best_model.pth')
            )
            print(f"  -> New best model saved (Dice: {best_dice:.4f})")

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics,
            str(output_dir / 'latest_checkpoint.pth')
        )

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Check early stopping
        if early_stopping(val_metrics['val_dice']):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation Dice: {best_dice:.4f}")
            break

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='AFN-DeSeg Stage 2: Joint Denoising and Segmentation Training'
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/stage2',
                        help='Output directory for checkpoints')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')

    # Model arguments
    parser.add_argument('--pretrained_vit', type=str, default=None,
                        help='Path to pretrained ViT weights')
    parser.add_argument('--freeze_vit', action='store_true',
                        help='Freeze ViT backbone (only train LoRA)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay for AdamW')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')

    # Loss weights
    parser.add_argument('--lambda_rec', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--lambda_seg', type=float, default=10.0,
                        help='Weight for segmentation loss')
    parser.add_argument('--lambda_percep', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--use_cellpose', action='store_true', default=True,
                        help='Use Cellpose for perceptual loss')
    parser.add_argument('--no_cellpose', action='store_false', dest='use_cellpose',
                        help='Disable Cellpose perceptual loss')

    # Early stopping
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs)')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def main():
    """Entry point for Stage 2 joint training."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == '__main__':
    main()
