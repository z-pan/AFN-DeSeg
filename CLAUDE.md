# AFN-DeSeg — Developer Guide

## Architecture

AFN-DeSeg uses a **dual-encoder** design that combines local spatial features from a U-Net with global semantic context from a frozen DINOv3 Vision Transformer.

**U-Net encoder** — 4 downsampling blocks with channel progression 64 → 128 → 256 → 512. Produces a bottleneck tensor at 1/16 input resolution and a `skip_connections` list (one entry per encoder level) used by both decoders.

**DINOv3 ViT encoder** — ViT-Base/16 (768-dim, 12 heads, depth 12). Processes pseudo-RGB input (3-channel repeat of the grayscale TPAF image with ImageNet normalisation). Only the Q and V projection matrices in each attention block receive LoRA adapters (r=32, α=32 by default); the rest of the backbone is frozen.

**Feature fusion block** — Concatenates the U-Net bottleneck (512 ch) with spatially-reshaped ViT patch tokens (768 ch, bicubic-upsampled to match spatial dims), then merges via a 1×1 conv followed by a 3×3 conv.

**Denoising decoder** — Residual learning: predicts a signed residual (Tanh-bounded), which is added to the recovered input grayscale and clamped to [0, 1]. Single-channel output. Shares `fused` and `skip_connections` with the segmentation decoder.

**Segmentation decoder** — Predicts a per-pixel probability map via sigmoid. Binary mask at inference is obtained by thresholding at 0.5. Shares `fused` and `skip_connections` with the denoising decoder.

## Key Conventions

- Single-channel TPAF images are repeated across 3 channels to form pseudo-RGB before ViT input; ImageNet mean/std normalisation is applied (`data.imagenet_mean/std` in `configs/default_config.yaml`).
- LoRA adapters are the only trainable ViT parameters by default (`freeze_vit=True`); adapter rank and alpha are set at model construction time via `lora_r` / `lora_alpha`.
- The sigmoid on the segmentation output is applied inside `SegmentationDecoder.forward()` — it is **not** re-applied inside the loss. The denoising output uses `torch.clamp` after residual addition, not sigmoid.
- Masks are stored as uint8 TIFF (patch-selected training images) or uint16 PNG (raw outputs from Cellpose); `TPAFDataset._load_mask` routes by file extension (tifffile for `.tif/.tiff`, skimage for `.png`).
- Checkpoints saved by `train_stage2_colab.py` contain the full model `state_dict` and optimiser state; resume training with `--resume <checkpoint_path>`.
- AMP uses the non-deprecated API: `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda', ...)`. BCE loss must be wrapped in `torch.amp.autocast(device_type='cuda', enabled=False)` to avoid the PyTorch 2.x dispatcher block.

## Gradient Isolation

Both decoders receive the same `fused` features and `skip_connections` from the U-Net encoder. Gradient flow between the two task heads through these shared tensors is **intentional** and enables joint optimisation.

- **Do NOT remove the `.detach()` calls on `fused` and `skip_connections` in the denoising decoder — removing them causes segmentation loss to corrupt the denoising decoder.**

## Loss Weights

Stage 2 joint loss: `L_total = λ_rec · L_rec + λ_seg · L_seg + λ_percep · L_percep`

| Weight | Value | Notes |
|---|---|---|
| `lambda_rec` | 1.0 | L1 reconstruction loss |
| `lambda_seg` | 10.0 | Dice + BCE segmentation loss |
| `lambda_percep` | 0.0 | **Perceptual loss is permanently disabled (`lambda_percep=0.0`) — do NOT enable it. The Cellpose feature hook causes training crashes.** |

## Data Layout (Stage 2)

Training scripts auto-detect split-aware layout (preferred) or fall back to a flat layout with 85/15 filename-based split:

```
data_dir/
  train/
    noisy/    *.tif
    clean/    *.tif
    masks/    *.tif  or  *.png
  val/
    noisy/
    clean/
    masks/
```

On Windows, create splits with `--copy` (step3) to avoid symlink privilege errors (WinError 1314).

## Noise Model

Mixed Poisson-Gaussian: `y = (1/a) · Poisson(a · x) + N(0, b²)`

Variance scales as `Var[y_N] = (x/a + b²) / N` for N-frame averages. Parameters `a` (Poisson scale) and `b` (Gaussian std) are estimated by Multi-Level Mean-Variance regression (IRLS, Huber loss) over multi-frame TPAF calibration data.

- Estimation: `noise_scripts/estimate_noise.py` — outputs `noise_params.json`
- Synthesis: `noise_scripts/generate_noisy.py` — supports `--random_level` and `--level_weights`
- Correction factor `(1/N + 1/16)` is applied during estimation to account for calibration data at 1–16 frame averages

## Reproducing the Training Pipeline

1. **`data_prep/step1_estimate_noise.py`** — estimate noise parameters from per-level TPAF calibration data and reorganise to per-FOV symlink layout.
2. **`data_prep/step1b_select_patches.py`** — sliding-window patch selection (512×512, stride 256) keeping patches with ≥ 50 nucleus pixels.
3. **`data_prep/step2_generate_noisy.py`** — synthesise noisy images at target frame-count levels (`--n_frames 1 4 8 16` or `--random_level`).
4. **`data_prep/step3_build_splits.py`** — assemble train/val splits with clean images, masks, and noisy images (use `--copy` on Windows).
5. **`training/train_stage2_colab.py`** — joint Stage 2 training with Colab A100 settings (batch_size=8, lora_r=32, num_workers=2).
