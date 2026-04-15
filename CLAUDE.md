# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AFN-DeSeg is a dual-encoder deep learning framework for joint denoising and nuclear segmentation of Two-Photon Autofluorescence (TPAF) microscopy images, with downstream Key Diagnostic Area (KDA) prediction. Published in Nature Communications.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_model.py -v

# Run a specific test
pytest tests/test_model.py::test_forward_pass -v

# Stage 1: DINO domain adaptation (self-supervised, unlabeled data)
python training/train_stage1_dino.py --data_dir <unlabeled_images> --output_dir <output>

# Stage 2: Joint denoising + segmentation (supervised)
python training/train_stage2_joint.py --data_dir <data> --output_dir <output> --pretrained_vit <stage1_weights>

# Stage 2 on Colab (AMP-enabled variant)
python training/train_stage2_colab.py --data_dir <data> --output_dir <output> --pretrained_vit <stage1_weights> --freeze_vit --lora_r 32

# Inference
python inference/predict.py --checkpoint <model.pth> --input <images> --output <results>
```

## Architecture

### Dual-Encoder Design

The core model (`models/afn_deseg.py`) has two parallel encoders feeding into two parallel decoders via a fusion block:

```
Input (1ch TPAF) → preprocess_tpaf() → pseudo-RGB (3ch, ImageNet-normalized)
                         ↓
         ┌───────────────┴───────────────┐
    U-Net Encoder                  DINOv2 ViT Encoder
   (local features)              (global context, LoRA)
    64→128→256→512                768-dim tokens, r=32
         │ skip connections              │
         └───────────┬──────────────────┘
                Fusion Block
          (concat 512+768 → conv → fused)
                     ↓
         ┌───────────┴───────────┐
   Seg Decoder              Denoising Decoder
   (raw logits)          (Tanh residual learning)
         ↓                       ↓
   Binary mask           denoised = clamp(input_gray + residual, 0, 1)
```

### Gradient Isolation

The denoising decoder receives `fused.detach()` and `[s.detach() for s in skip_connections]`. This means:
- The shared encoder is trained **only** by segmentation loss
- The denoising decoder trains independently on frozen encoder features
- This prevents multi-task gradient conflict between L_rec and L_seg

### LoRA on ViT

LoRA (`models/afn_deseg.py: LoRALinear`) adapts the frozen DINOv2 ViT by adding low-rank matrices to Q/V projections. Stage 1 pretrains these LoRA weights via DINO self-supervised learning on unlabeled TPAF images. Stage 2 fine-tunes them jointly.

### Loss Functions

`losses/joint_loss.py` computes: `L_total = λ_rec * L1(denoised, clean) + λ_seg * (Dice + BCE)(logits, mask)`

Key details:
- SegmentationDecoder outputs **raw logits** (no sigmoid) — `BCEWithLogitsLoss` is used for AMP numerical stability (FP16 sigmoid can produce exact 0.0 → log(0)=inf)
- `DiceLoss` applies sigmoid internally
- DenoisingDecoder uses **residual learning**: output is Tanh-bounded noise residual, not the clean image directly
- Perceptual loss (Cellpose features) is disabled by default (`lambda_percep=0.0`) due to hook issues

### Two-Stage Training Pipeline

1. **Stage 1** (`training/train_stage1_dino.py`): Self-supervised DINO distillation on unlabeled TPAF images. Trains LoRA weights + student/teacher ViT. Output: `best_dino_encoder.pth`
2. **Stage 2** (`training/train_stage2_joint.py` or `train_stage2_colab.py`): Supervised joint training. Loads Stage 1 ViT weights, freezes ViT backbone (only LoRA trainable), trains U-Net encoder + both decoders. Output: `best_model.pth`

### KDA Pipeline (Downstream)

`models/attention_unet.py` — Separate Attention U-Net that takes binary nuclear masks from AFN-DeSeg and predicts Key Diagnostic Areas. Trained independently via `training/kda_trainer.py`.

## Data Format

Stage 2 expects triplets (noisy image, clean image, binary mask) in either layout:

```
# Split-aware (preferred)          # Flat (auto-splits 85/15)
data_dir/                           data_dir/
├── train/                          ├── noisy/
│   ├── noisy/                      ├── clean/
│   ├── clean/                      └── masks/
│   └── masks/
└── val/
    ├── noisy/
    ├── clean/
    └── masks/
```

Supported formats: `.npy`, `.png`, `.tif`. Files are matched by filename across subdirectories.

## Key Conventions

- All images are normalized to ImageNet stats before entering the model (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Single-channel TPAF images are converted to pseudo-RGB by `preprocess_tpaf()` (replicate across 3 channels)
- The denoising decoder recovers original [0,1] grayscale by un-normalizing the input before adding the residual
- Config defaults are in `configs/default_config.yaml` — training scripts override these via CLI args
- `model.get_trainable_params()` returns only the parameters that should be optimized (respects frozen ViT + LoRA)
- The Colab training script uses AMP (FP16) with `GradScaler`; the non-Colab script does not
- Validation metric for early stopping and best-model saving is `val_dice`
- `torch.load` calls use `weights_only=False` for PyTorch 2.6+ compatibility
