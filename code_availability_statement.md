# Code Availability Statement

## Overview

The complete source code for **AFN-DeSeg** (Auto-Fluorescence Nuclei Denoising and Segmentation) is publicly available to ensure full reproducibility of the results presented in this study.

## Repository

**GitHub Repository**: https://github.com/z-pan/AFN-DeSeg

## Contents

The repository includes:

1. **Model Implementation**
   - AFN-DeSeg dual-encoder architecture (U-Net + DINOv3 ViT)
   - LoRA adaptation modules for efficient fine-tuning
   - Attention U-Net for Key Diagnostic Area (KDA) prediction

2. **Training Scripts**
   - Stage 1: DINO domain adaptation for TPAF microscopy
   - Stage 2: Joint denoising and segmentation training
   - KDA model training

3. **Data Processing**
   - Mixed Poisson-Gaussian (MPG) noise synthesis
   - Data augmentation pipelines
   - Dataset loaders for TPAF images

4. **Evaluation**
   - Comprehensive metrics (PSNR, SSIM, Dice, IoU, mAP, HD95)
   - KDA-specific metrics (KAF, nuclear density correlation)
   - Bland-Altman analysis for clinical validation

5. **Inference**
   - Prediction scripts for new TPAF images
   - Batch processing utilities

## System Requirements

### Software Dependencies
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.7 (for GPU acceleration)

### Hardware Requirements
- GPU: NVIDIA GPU with >= 8GB VRAM (recommended: RTX 3090 or A100)
- RAM: >= 32GB system memory
- Storage: >= 10GB for code, models, and sample data

### Typical Installation Time
- Fresh installation: ~10-15 minutes
- With pre-downloaded model weights: ~5 minutes

## Reproducibility

### Pretrained Weights
Pretrained model weights are available at:
- DINOv3 backbone: `facebook/dinov3-vitb16-pretrain-lvd1689m` (HuggingFace)
- AFN-DeSeg checkpoint: Available in the `checkpoint/` directory

### Random Seeds
All experiments use fixed random seeds for reproducibility:
- PyTorch seed: 42
- NumPy seed: 42
- CUDA deterministic mode: enabled

### Expected Runtime
- Training (Stage 2, 150 epochs): ~12-24 hours on single A100 GPU
- Inference (per 512×512 image): ~50-100ms on GPU

## License

This code is released under the **MIT License**, allowing free use, modification, and distribution for both academic and commercial purposes with proper attribution.

## Citation

If you use this code in your research, please cite:

```
Pan, Z., Song, N., Cheng, S., et al. A Joint Denoising and Segmentation Framework
for Ovarian Cancer Diagnosis based on Two-Photon Autofluorescence Microscopy.
Nature Communications (2024).
```

## Contact

For questions regarding the code implementation:
- **Corresponding Author**: [Author Name]
- **Email**: [Contact Email]
- **Issues**: https://github.com/z-pan/AFN-DeSeg/issues
