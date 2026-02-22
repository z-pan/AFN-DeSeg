"""
Stage 1: DINOv3 Domain Adaptation for AFN-DeSeg.

Based on section 4.2.3 "Microscopy Domain Adaptation".

This stage performs self-supervised training on unlabeled TPAF patches
to adapt the DINOv3 backbone to the TPAF domain before joint training.

Training Configuration:
    - LoRA rank r=32, alpha=32 on Query and Value matrices
    - 50 epochs with DINO student-teacher distillation
    - Teacher EMA decay: 0.998
    - Auto-detects bfloat16 on A100/H100 (falls back to float16 + GradScaler)
    - Multi-crop augmentation with intensity perturbations

Usage:
    python training/train_stage1_dino.py \
        --data_dir /path/to/unlabeled_tpaf \
        --output_dir /path/to/output
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.afn_deseg import DINOv3Encoder
from data.dataset import TPAFUnlabeledDataset
from data.augmentations import get_dino_transform


# =============================================================================
# DINO Loss
# =============================================================================

class DINOLoss(nn.Module):
    """
    DINO self-supervised loss.

    Implements student-teacher distillation with centering.
    The student learns to match the teacher's output distribution
    across different augmented views of the same image.
    """

    def __init__(
        self,
        out_dim: int = 768,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9
    ):
        """
        Args:
            out_dim: Output dimension of the encoder.
            teacher_temp: Temperature for teacher softmax.
            student_temp: Temperature for student softmax.
            center_momentum: Momentum for center update.
        """
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # Center for teacher outputs (running mean)
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DINO loss.

        Args:
            student_output: Student encoder output (B, D).
            teacher_output: Teacher encoder output (B, D).

        Returns:
            Cross-entropy loss between student and teacher.
        """
        # Teacher output with centering and sharpening
        teacher_probs = F.softmax(
            (teacher_output - self.center) / self.teacher_temp,
            dim=-1
        )

        # Student output with temperature
        student_log_probs = F.log_softmax(
            student_output / self.student_temp,
            dim=-1
        )

        # Cross-entropy loss
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()

        # Update center with EMA
        self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Update center with exponential moving average."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)


# =============================================================================
# DINO Trainer
# =============================================================================

class DINOTrainer:
    """
    Trainer for DINO self-supervised learning.

    Implements student-teacher training with EMA teacher updates.
    Uses mixed precision and sequential view processing to minimize
    GPU memory usage.
    """

    def __init__(
        self,
        student: DINOv3Encoder,
        teacher: DINOv3Encoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: DINOLoss,
        device: torch.device,
        ema_decay: float = 0.996
    ):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.ema_decay = ema_decay

        # Auto-detect bfloat16 support (A100, H100, etc.)
        self.use_bf16 = (
            device.type == 'cuda'
            and torch.cuda.is_bf16_supported()
        )
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16

        if self.use_bf16:
            print("  -> Using bfloat16 (A100/H100 detected). GradScaler disabled.")
        else:
            print("  -> Using float16 with GradScaler.")

        # GradScaler is only needed for float16; bfloat16 has enough
        # exponent range that loss scaling is unnecessary.
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(device.type == 'cuda' and not self.use_bf16)
        )

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher with EMA of student weights."""
        for student_param, teacher_param in zip(
            self.student.parameters(),
            self.teacher.parameters()
        ):
            teacher_param.data = (
                self.ema_decay * teacher_param.data +
                (1 - self.ema_decay) * student_param.data
            )

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Uses mixed precision (bfloat16 on A100/H100, float16 otherwise)
        and sequential view processing to minimize GPU memory. Each
        view's loss is computed and back-propagated independently so
        only one ViT computation graph is held in memory at a time.

        Args:
            dataloader: Training dataloader.
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.student.train()
        self.teacher.eval()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [DINO]", leave=False)

        for batch in pbar:
            # Get image batch
            images = batch.to(self.device)

            # Create two augmented views
            view1 = images
            view2 = torch.flip(images, dims=[-1])

            # Convert to pseudo-RGB (replicate channel)
            view1_rgb = view1.repeat(1, 3, 1, 1) if view1.shape[1] == 1 else view1
            view2_rgb = view2.repeat(1, 3, 1, 1) if view2.shape[1] == 1 else view2

            # Teacher forward for both views (no grad, low memory)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.amp_dtype):
                teacher_feat1 = self.teacher(view1_rgb).mean(dim=1)
                teacher_feat2 = self.teacher(view2_rgb).mean(dim=1)

            # Process views sequentially to avoid holding two full
            # student computation graphs in memory simultaneously.
            self.optimizer.zero_grad()

            # View 1: student predicts teacher of view 2
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                student_feat1 = self.student(view1_rgb).mean(dim=1)
                loss1 = self.criterion(student_feat1, teacher_feat2) / 2
            self.scaler.scale(loss1).backward()
            del student_feat1, loss1

            # View 2: student predicts teacher of view 1
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                student_feat2 = self.student(view2_rgb).mean(dim=1)
                loss2 = self.criterion(student_feat2, teacher_feat1) / 2
            self.scaler.scale(loss2).backward()

            # Step optimizer with scaled gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update teacher EMA
            self.update_teacher()

            # Accumulate (reconstruct total loss for logging)
            batch_loss = loss2.item() * 2  # approx; both halves are similar
            del student_feat2, loss2
            total_loss += batch_loss
            num_batches += 1

            pbar.set_postfix({'loss': f"{batch_loss:.4f}"})

        # Step scheduler
        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }


# =============================================================================
# Main Training Function
# =============================================================================

def train_stage1(args):
    """Main Stage 1 training function."""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['device'] = str(device)
    with open(output_dir / 'config_stage1.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize student encoder
    print("Initializing DINOv3 encoder (student)...")
    student = DINOv3Encoder(
        img_size=args.img_size,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    student = student.to(device)

    # Initialize teacher (copy of student)
    print("Initializing DINOv3 encoder (teacher)...")
    teacher = DINOv3Encoder(
        img_size=args.img_size,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    teacher.load_state_dict(student.state_dict())
    teacher = teacher.to(device)

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    # Load pretrained weights if provided
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        if args.pretrained.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(args.pretrained, device=str(device))
        else:
            state_dict = torch.load(args.pretrained, map_location=device)
        student.load_state_dict(state_dict, strict=False)
        teacher.load_state_dict(state_dict, strict=False)

    # Create dataset
    print("Loading dataset...")
    transform = get_dino_transform(args.img_size)
    dataset = TPAFUnlabeledDataset(
        args.data_dir,
        img_size=args.img_size,
        transform=transform,
        normalize=args.normalize,
        split_channels=args.split_channels
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Initialize loss
    criterion = DINOLoss(
        out_dim=768,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum
    ).to(device)

    # Initialize optimizer (only train LoRA parameters by default)
    if args.train_full:
        params = student.parameters()
    else:
        # Only LoRA parameters
        params = [p for n, p in student.named_parameters()
                 if 'lora_A' in n or 'lora_B' in n]
        print(f"Training {len(params)} LoRA parameter groups")

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # Initialize trainer
    trainer = DINOTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        ema_decay=args.ema_decay
    )

    # Training history
    history = []
    best_loss = float('inf')

    # Training loop
    print(f"\nStarting Stage 1 DINO training for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(dataloader, epoch)
        history.append(metrics)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Loss: {metrics['loss']:.4f} | "
              f"LR: {metrics['lr']:.2e}")

        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save(
                student.state_dict(),
                output_dir / 'best_dino_encoder.pth'
            )
            print(f"  -> New best model saved (Loss: {best_loss:.4f})")

        # Save latest
        torch.save(
            student.state_dict(),
            output_dir / 'latest_dino_encoder.pth'
        )

        # Save history
        with open(output_dir / 'history_stage1.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Stage 1 training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Encoder saved to: {output_dir / 'best_dino_encoder.pth'}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='AFN-DeSeg Stage 1: DINO Domain Adaptation'
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to unlabeled TPAF images')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/stage1',
                        help='Output directory')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--split_channels', action='store_true',
                        help='Split multi-channel images (e.g. FAD+NADH) into '
                             'separate grayscale samples. Empty channels are '
                             'auto-skipped.')
    parser.add_argument('--normalize', type=str, default='percentile',
                        choices=['minmax', 'percentile', 'fixed'],
                        help='Per-image normalization method (default: percentile)')

    # Model arguments
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained DINOv3 weights')
    parser.add_argument('--lora_r', type=int, default=32,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--train_full', action='store_true',
                        help='Train full encoder (not just LoRA)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')

    # DINO arguments
    parser.add_argument('--ema_decay', type=float, default=0.998,
                        help='EMA decay for teacher')
    parser.add_argument('--teacher_temp', type=float, default=0.04,
                        help='Teacher temperature')
    parser.add_argument('--student_temp', type=float, default=0.1,
                        help='Student temperature')
    parser.add_argument('--center_momentum', type=float, default=0.9,
                        help='Center momentum')

    # Other
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def main():
    """Entry point for Stage 1 DINO domain adaptation training."""
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_stage1(args)


if __name__ == '__main__':
    main()
