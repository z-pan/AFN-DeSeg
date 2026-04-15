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

The core model (`models/afn_deseg.py`) has two parallel encoders feeding into two parallel decoders via a fusion block.

**U-Net encoder** — 4 downsampling blocks, channel progression 64 → 128 → 256 → 512. Produces a bottleneck tensor at 1/16 input resolution and a `skip_connections` list used by both decoders.

**DINOv2 ViT encoder** — ViT-Base/16 (768-dim, 12 heads, depth 12). Processes pseudo-RGB input (3-channel repeat of the grayscale TPAF image with ImageNet normalisation). Only Q and V projection matrices receive LoRA adapters (r=32, α=32 by default); the rest of the backbone is frozen.

**Feature fusion block** — Concatenates U-Net bottleneck (512 ch) with spatially-reshaped ViT patch tokens (768 ch, bicubic-upsampled to match spatial dims), then merges via 1×1 conv → 3×3 conv.

**Denoising decoder** — Residual learning: predicts a Tanh-bounded signed residual added to the recovered input grayscale, then clamped to [0, 1]. Single-channel output.

**Segmentation decoder** — Predicts a per-pixel probability map via sigmoid. Binary mask at inference by thresholding at 0.5.

### Gradient Isolation

The denoising decoder receives `fused.detach()` and `[s.detach() for s in skip_connections]`. This means:
- The shared encoder is trained **only** by segmentation loss
- The denoising decoder trains independently on frozen encoder features
- This prevents multi-task gradient conflict between L_rec and L_seg
- **Do NOT remove the `.detach()` calls on `fused` and `skip_connections` in the denoising decoder — removing them causes segmentation loss to corrupt the denoising decoder.**

### LoRA on ViT

LoRA (`models/afn_deseg.py: LoRALinear`) adapts the frozen DINOv2 ViT by adding low-rank matrices to Q/V projections. Stage 1 pretrains these LoRA weights via DINO self-supervised learning on unlabeled TPAF images. Stage 2 fine-tunes them jointly.

### Loss Functions

`losses/joint_loss.py` computes: `L_total = λ_rec * L1(denoised, clean) + λ_seg * (Dice + BCE)(probs, mask)`

Key details:
- SegmentationDecoder outputs **sigmoid probabilities** — `BCELoss` is used; the loss must be wrapped in `torch.amp.autocast(device_type='cuda', enabled=False)` to avoid the PyTorch 2.x AMP dispatcher block
- `DiceLoss` also receives sigmoid probabilities directly
- DenoisingDecoder uses **residual learning**: output is Tanh-bounded noise residual, not the clean image directly
- **Perceptual loss is permanently disabled (`lambda_percep=0.0`) — do NOT enable it. The Cellpose feature hook causes training crashes.**

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

Supported formats: `.npy`, `.png`, `.tif`. Files are matched by filename across subdirectories. On Windows, use `--copy` with `data_prep/step3_build_splits.py` to avoid symlink privilege errors (WinError 1314).

## Key Conventions

- All images are normalised to ImageNet stats before entering the model (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Single-channel TPAF images are converted to pseudo-RGB by `preprocess_tpaf()` (replicate across 3 channels)
- The denoising decoder recovers original [0,1] grayscale by un-normalising the input before adding the residual
- `model.get_trainable_params()` returns only the parameters that should be optimised (respects frozen ViT + LoRA)
- Masks are stored as uint8 TIFF (patch-selected training images) or uint16 PNG (raw Cellpose outputs); `TPAFDataset._load_mask` routes by extension (tifffile for `.tif/.tiff`, skimage for `.png`)
- AMP uses the non-deprecated API: `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda', ...)`
- The Colab training script (`train_stage2_colab.py`) uses AMP (FP16) with `GradScaler`; the non-Colab script does not
- Validation metric for early stopping and best-model saving is `val_dice`
- `torch.load` calls use `weights_only=False` for PyTorch 2.6+ compatibility

## Noise Model

Mixed Poisson-Gaussian: `y = (1/a) · Poisson(a · x) + N(0, b²)`

Variance scales as `Var[y_N] = (x/a + b²) / N` for N-frame averages. Parameters `a` (Poisson scale) and `b` (Gaussian std) are estimated by Multi-Level Mean-Variance regression (IRLS, Huber loss) over multi-frame TPAF calibration data.

- Estimation: `noise_scripts/estimate_noise.py` — outputs `noise_params.json`
- Synthesis: `noise_scripts/generate_noisy.py` — supports `--random_level` and `--level_weights`
- Correction factor `(1/N + 1/16)` is applied during estimation to account for calibration data at 1–16 frame averages

## Reproducing the Data Preparation Pipeline

1. **`data_prep/step1_estimate_noise.py`** — estimate noise parameters from per-level TPAF calibration data and reorganise to per-FOV symlink layout.
2. **`data_prep/step1b_select_patches.py`** — sliding-window patch selection (512×512, stride 256) keeping patches with ≥ 50 nucleus pixels.
3. **`data_prep/step2_generate_noisy.py`** — synthesise noisy images at target frame-count levels (`--n_frames 1 4 8 16` or `--random_level`).
4. **`data_prep/step3_build_splits.py`** — assemble train/val splits with clean images, masks, and noisy images (use `--copy` on Windows).
