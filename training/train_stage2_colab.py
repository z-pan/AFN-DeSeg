"""
Stage 2: Joint Denoising and Segmentation Training for AFN-DeSeg.
Optimized for Google Colab (A100, 40 GB).

Key differences from train_stage2_joint.py
-------------------------------------------
1. Fixes a bug in the original script where train and val datasets both
   read from the same directory (split parameter only toggled augmentation).
   This script expects dedicated train/ and val/ subdirectories.
2. batch_size=8  (A100 40 GB handles 512×512 dual-encoder at batch 8)
3. num_workers=2  (Colab typically exposes 2 CPU cores)
4. lora_r=32  (matches default_config.yaml for A100)
5. Automatic Mixed Precision (AMP) via torch.cuda.amp — A100 has native
   BF16 tensor cores, giving ~1.5–2× throughput with no accuracy loss.
6. Robust Stage 1 ViT weight loading that handles both a raw state dict
   and a full checkpoint dict (with / without 'vit_encoder.' prefix).

Expected data directory layout
--------------------------------
    data_dir/
        train/
            noisy/    # ~480 noisy TPAF images  (512×512, .tif / .npy)
            clean/    # ~480 clean reference images
            masks/    # ~480 binary segmentation masks
        val/
            noisy/    # ~60 noisy images
            clean/    # ~60 clean images
            masks/    # ~60 masks

Colab usage
-----------
    from google.colab import drive
    drive.mount('/content/drive')

    !python /content/AFN-DeSeg/training/train_stage2_colab.py \\
        --data_dir      /content/drive/MyDrive/AFN-DeSeg/data \\
        --output_dir    /content/drive/MyDrive/AFN-DeSeg/checkpoints/stage2 \\
        --pretrained_vit /content/drive/MyDrive/AFN-DeSeg/checkpoints/stage1/best_vit.pth

Resume after a Colab session drop
-----------------------------------
    !python /content/AFN-DeSeg/training/train_stage2_colab.py \\
        --data_dir      /content/drive/MyDrive/AFN-DeSeg/data \\
        --output_dir    /content/drive/MyDrive/AFN-DeSeg/checkpoints/stage2 \\
        --resume        /content/drive/MyDrive/AFN-DeSeg/checkpoints/stage2/latest_checkpoint.pth
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure the project root is on the path regardless of where the script
# is launched from (handles both  !python training/...  and  %run  in Colab).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from losses.joint_loss import AFNJointLoss
from models.afn_deseg import AFNDeSeg


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10((max_val ** 2) / mse)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum().item()
    union = pred_bin.sum().item() + target.sum().item()
    if union < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0
    return (2.0 * intersection) / union


# =============================================================================
# Dataset
# =============================================================================

_SUPPORTED_EXT = ['.npy', '.png', '.tif', '.tiff']


def _find_file(directory: Path, stem: str) -> Optional[Path]:
    for ext in _SUPPORTED_EXT:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _load_image(path: Path) -> np.ndarray:
    if path.suffix == '.npy':
        return np.load(str(path)).astype(np.float32)
    try:
        import tifffile
        if path.suffix in ('.tif', '.tiff'):
            return tifffile.imread(str(path)).astype(np.float32)
    except ImportError:
        pass
    from skimage import io as skio
    return skio.imread(str(path)).astype(np.float32)


def _to_2d(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    # Channels-last  (H, W, C)  or channels-first  (C, H, W)
    if img.shape[-1] <= 4:
        return img.mean(axis=-1)
    if img.shape[0] <= 4:
        return img.mean(axis=0)
    return img[..., 0]


def _normalize_percentile(img: np.ndarray) -> np.ndarray:
    """Clip 1–99 percentile then rescale to [0, 1]."""
    lo, hi = np.percentile(img, [1, 99])
    if hi - lo > 1e-8:
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo)
    return img.astype(np.float32)


def _augment(noisy: np.ndarray, clean: np.ndarray, mask: np.ndarray):
    """Geometric augmentation: random H/V flip + random 90° rotation."""
    arrays = [noisy, clean, mask]

    if np.random.rand() > 0.5:                          # H-flip
        arrays = [np.flip(a, axis=-1).copy() for a in arrays]
    if np.random.rand() > 0.5:                          # V-flip
        arrays = [np.flip(a, axis=-2).copy() for a in arrays]

    k = np.random.randint(0, 4)                          # 0°/90°/180°/270°
    if k:
        arrays = [np.rot90(a, k, axes=(-2, -1)).copy() for a in arrays]

    return arrays[0], arrays[1], arrays[2]


class TPAFDataset(Dataset):
    """
    TPAF triplet dataset.

    Reads from  split_dir/{noisy,clean,masks}/  where every filename stem
    present in noisy/ is expected to also appear in clean/ and masks/.
    Missing clean or mask files fall back to the noisy image and zero mask
    respectively (with a warning on first occurrence).

    Args:
        split_dir: Directory that contains noisy/, clean/, masks/ subfolders.
        augment:   Whether to apply geometric augmentation (train=True).
    """

    def __init__(self, split_dir: str, augment: bool = False):
        self.split_dir = Path(split_dir)
        self.augment = augment

        self.noisy_dir = self.split_dir / 'noisy'
        self.clean_dir = self.split_dir / 'clean'
        self.mask_dir  = self.split_dir / 'masks'

        if not self.noisy_dir.exists():
            raise FileNotFoundError(
                f"noisy/ directory not found inside {split_dir}. "
                "Create data_dir/train/noisy/ and data_dir/val/noisy/."
            )

        self.samples: List[str] = sorted(
            f.stem for f in self.noisy_dir.iterdir()
            if f.suffix in _SUPPORTED_EXT
        )

        if not self.samples:
            raise ValueError(f"No images found in {self.noisy_dir}")

        self._warned_clean = False
        self._warned_mask  = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.samples[idx]

        noisy_path = _find_file(self.noisy_dir, stem)
        clean_path = _find_file(self.clean_dir, stem)
        mask_path  = _find_file(self.mask_dir,  stem)

        noisy = _normalize_percentile(_to_2d(_load_image(noisy_path)))

        if clean_path:
            clean = _normalize_percentile(_to_2d(_load_image(clean_path)))
        else:
            if not self._warned_clean:
                print(f"[Dataset] clean/ image not found for '{stem}'; using noisy as fallback.")
                self._warned_clean = True
            clean = noisy.copy()

        if mask_path:
            mask = (_to_2d(_load_image(mask_path)) > 0.5).astype(np.float32)
        else:
            if not self._warned_mask:
                print(f"[Dataset] masks/ image not found for '{stem}'; using zero mask as fallback.")
                self._warned_mask = True
            mask = np.zeros_like(noisy)

        if self.augment:
            noisy, clean, mask = _augment(noisy, clean, mask)

        noisy = torch.from_numpy(noisy).unsqueeze(0)   # (1, H, W)
        clean = torch.from_numpy(clean).unsqueeze(0)
        mask  = torch.from_numpy(mask).unsqueeze(0)

        return noisy, clean, mask


# =============================================================================
# Early stopping
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 20, mode: str = 'max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best: Optional[float] = None

    def __call__(self, score: float) -> bool:
        if self.best is None:
            self.best = score
            return False
        improved = (score > self.best) if self.mode == 'max' else (score < self.best)
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    metrics: Dict,
    path: str,
) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'metrics':              metrics,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
) -> Dict:
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt


def load_vit_weights(model: AFNDeSeg, path: str, device: torch.device) -> None:
    """
    Load Stage 1 ViT weights into model.vit_encoder.

    Handles three formats produced by different Stage 1 scripts:
      (a) Full Stage 1 checkpoint dict  →  'model_state_dict' key present
      (b) Raw full-model state dict     →  keys begin with 'vit_encoder.'
      (c) Raw ViT-only state dict       →  keys begin with 'patch_embed.' etc.
    """
    raw = torch.load(path, map_location=device)

    # Unwrap checkpoint dict if present
    state = raw.get('model_state_dict', raw)

    # Try to strip 'vit_encoder.' prefix (format b)
    vit_state = {
        k[len('vit_encoder.'):]: v
        for k, v in state.items()
        if k.startswith('vit_encoder.')
    }

    if vit_state:
        missing, unexpected = model.vit_encoder.load_state_dict(vit_state, strict=False)
    else:
        # Assume the entire state dict belongs to the ViT encoder (format c)
        missing, unexpected = model.vit_encoder.load_state_dict(state, strict=False)
        vit_state = state

    print(f"  Loaded {len(vit_state)} tensors into vit_encoder  "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"  Missing keys (first 5): {missing[:5]}")


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    max_grad_norm: float,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", leave=False)
    for noisy, clean, mask in pbar:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        mask  = mask.to(device, non_blocking=True)

        noisy_rgb = AFNDeSeg.preprocess_tpaf(noisy, normalize=True)

        with autocast('cuda'):
            denoised, seg_pred = model(noisy_rgb)
            loss, loss_dict = criterion(denoised, seg_pred, clean, mask)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return {f"train_{k}": v / n_batches for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = total_dice = total_psnr = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Val]  ", leave=False)
    for noisy, clean, mask in pbar:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        mask  = mask.to(device, non_blocking=True)

        noisy_rgb = AFNDeSeg.preprocess_tpaf(noisy, normalize=True)

        with autocast('cuda'):
            denoised, seg_pred = model(noisy_rgb)
            loss, _ = criterion(denoised, seg_pred, clean, mask)

        total_loss += loss.item()
        total_dice += compute_dice(seg_pred, mask)
        total_psnr += compute_psnr(denoised, clean)
        n_batches  += 1

        pbar.set_postfix(
            dice=f"{total_dice / n_batches:.4f}",
            psnr=f"{total_psnr / n_batches:.2f}",
        )

    return {
        'val_loss': total_loss / n_batches,
        'val_dice': total_dice / n_batches,
        'val_psnr': total_psnr / n_batches,
    }


# =============================================================================
# Main
# =============================================================================

def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # Device & GPU info
    # ------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
    print()

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("Initializing AFN-DeSeg model ...")
    model = AFNDeSeg(
        img_size=512,
        in_channels=3,
        base_channels=64,
        vit_embed_dim=768,
        vit_depth=12,
        vit_num_heads=12,
        vit_patch_size=16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r,   # keep scaling = alpha/r = 1.0
    ).to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_total:,} total  |  {n_trainable:,} trainable")

    # ------------------------------------------------------------------
    # Load Stage 1 pretrained ViT weights
    # ------------------------------------------------------------------
    if args.pretrained_vit:
        print(f"\nLoading Stage 1 ViT weights: {args.pretrained_vit}")
        load_vit_weights(model, args.pretrained_vit, device)

    # Freeze ViT backbone; only LoRA adapters + U-Net + decoders remain trainable
    if args.freeze_vit:
        model.vit_encoder.freeze_pretrained()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ViT backbone frozen.  Trainable params now: {n_trainable:,}")

    # ------------------------------------------------------------------
    # Loss, optimiser, scheduler, AMP scaler
    # ------------------------------------------------------------------
    criterion = AFNJointLoss(
        lambda_rec=args.lambda_rec,
        lambda_seg=args.lambda_seg,
        lambda_percep=args.lambda_percep,
        use_cellpose=args.use_cellpose,
    ).to(device)

    optimizer = AdamW(
        model.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler    = GradScaler('cuda')   # AMP — A100 BF16/FP16 tensor cores

    # ------------------------------------------------------------------
    # Datasets  (train/ and val/ live inside args.data_dir)
    # ------------------------------------------------------------------
    data_root = Path(args.data_dir)
    train_dir = data_root / 'train'
    val_dir   = data_root / 'val'

    print(f"\nLoading datasets from {data_root}")
    train_ds = TPAFDataset(str(train_dir), augment=True)
    val_ds   = TPAFDataset(str(val_dir),   augment=False)

    # persistent_workers avoids re-spawning worker processes each epoch
    # (beneficial in Colab where process creation is slow).
    _pw = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=_pw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=_pw,
    )

    print(f"  Train : {len(train_ds):4d} samples  ({len(train_loader)} batches/epoch)")
    print(f"  Val   : {len(val_ds):4d} samples")

    # ------------------------------------------------------------------
    # Optionally resume
    # ------------------------------------------------------------------
    best_dice  = 0.0
    start_epoch = 1

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = ckpt['epoch'] + 1
        best_dice   = ckpt['metrics'].get('val_dice', 0.0)
        print(f"  Continuing from epoch {start_epoch}  "
              f"(best val_dice so far: {best_dice:.4f})")

    early_stop = EarlyStopping(patience=args.patience, mode='max')
    history: List[Dict] = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Stage 2 training  —  {args.epochs} epochs  |  "
          f"batch={args.batch_size}  |  lora_r={args.lora_r}  |  "
          f"lr={args.lr:.0e}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.max_grad_norm,
        )
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'lr': current_lr}
        history.append(metrics)

        # Print epoch summary with loss breakdown
        t_loss = train_metrics.get('train_loss_total', float('nan'))
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={t_loss:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_dice={val_metrics['val_dice']:.4f}  "
            f"val_psnr={val_metrics['val_psnr']:.2f} dB  "
            f"lr={current_lr:.2e}"
        )
        # Per-component loss breakdown for diagnostics
        l_rec = train_metrics.get('train_loss_rec', 0)
        l_seg = train_metrics.get('train_loss_seg', 0)
        l_percep = train_metrics.get('train_loss_percep', 0)
        if l_rec or l_seg:
            print(f"  Loss breakdown: L_rec={l_rec:.4f}  L_seg={l_seg:.4f}  L_percep={l_percep:.4f}")

        # Always save latest checkpoint so Colab sessions can resume
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, val_metrics,
            str(output_dir / 'latest_checkpoint.pth'),
        )

        # Save best model (monitored on val_dice)
        if val_metrics['val_dice'] > best_dice:
            best_dice = val_metrics['val_dice']
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, val_metrics,
                str(output_dir / 'best_model.pth'),
            )
            print(f"  -> Best model saved  (val_dice={best_dice:.4f})")

        # Persist history to Drive so it survives session drops
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if early_stop(val_metrics['val_dice']):
            print(f"\nEarly stopping triggered at epoch {epoch}  "
                  f"(patience={args.patience})")
            break

    print(f"\n{'='*70}")
    print(f"Training complete.  Best val_dice = {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='AFN-DeSeg Stage 2 training — Colab A100',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- data ----
    p.add_argument('--data_dir', required=True,
                   help='Root dir with train/ and val/ subdirectories')
    p.add_argument('--output_dir', default='./checkpoints/stage2',
                   help='Where to save checkpoints and logs')

    # ---- model ----
    p.add_argument('--pretrained_vit', default=None,
                   help='Path to Stage 1 ViT checkpoint (.pth)')
    p.add_argument('--freeze_vit', action='store_true',
                   help='Freeze ViT backbone; only LoRA adapters + U-Net train')
    p.add_argument('--lora_r', type=int, default=32,
                   help='LoRA rank. 32 is the config default for A100.')

    # ---- training ----
    p.add_argument('--epochs',       type=int,   default=150)
    p.add_argument('--batch_size',   type=int,   default=8,
                   help='8 fits 512×512 dual-encoder model on A100 40 GB')
    p.add_argument('--num_workers',  type=int,   default=2,
                   help='Colab exposes 2 CPU cores')
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--min_lr',       type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--max_grad_norm',type=float, default=1.0)
    p.add_argument('--patience',     type=int,   default=20,
                   help='Early stopping patience in epochs')

    # ---- loss ----
    p.add_argument('--lambda_rec',    type=float, default=1.0,
                   help='Reconstruction loss weight')
    p.add_argument('--lambda_seg',    type=float, default=1.0,
                   help='Segmentation loss weight')
    p.add_argument('--lambda_percep', type=float, default=0.1)
    p.add_argument('--use_cellpose',  action='store_true', default=True)
    p.add_argument('--no_cellpose',   action='store_false', dest='use_cellpose',
                   help='Disable Cellpose perceptual loss (faster, slightly lower quality)')

    # ---- misc ----
    p.add_argument('--resume', default=None,
                   help='Path to latest_checkpoint.pth to resume after a session drop')
    p.add_argument('--seed',   type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == '__main__':
    main()
