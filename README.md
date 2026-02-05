# AFN-DeSeg

**Auto-Fluorescence Nuclei Denoising and Segmentation**

A physics-informed deep learning framework for joint image denoising and nuclear segmentation of Two-Photon Autofluorescence (TPAF) microscopy images.

## Overview

AFN-DeSeg addresses the challenge of extracting diagnostic-grade nuclear features from noisy TPAF images where nuclei appear as negative contrast (signal voids). The framework combines:

- **Dual-Encoder Architecture**: U-Net for local features + DINOv3 Vision Transformer for global semantic context
- **LoRA Adaptation**: Efficient fine-tuning of the ViT backbone with Low-Rank Adaptation
- **Joint Optimization**: Simultaneous denoising and segmentation with mutual reinforcement
- **Physics-Informed Noise Model**: Mixed Poisson-Gaussian (MPG) noise synthesis based on real TPAF imaging characteristics

## Project Structure

```
AFN-DeSeg/
├── configs/          # Configuration files
├── data/             # Data processing and noise synthesis
│   └── mpg_noise_synthesis.py
├── inference/        # Inference scripts
├── losses/           # Loss functions
│   └── joint_loss.py
├── models/           # Model architectures
│   └── afn_deseg.py
├── paper/            # Reference paper and supplementary info
├── training/         # Training scripts
│   └── train_stage2_joint.py
├── utils/            # Utility functions
├── requirements.txt  # Python dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/z-pan/AFN-DeSeg.git
cd AFN-DeSeg

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended)

Key dependencies:
- `torch`, `torchvision` - Deep learning framework
- `transformers` - DINOv3 ViT backbone
- `cellpose` - Perceptual loss computation
- `albumentations` - Data augmentation
- `scikit-image`, `tifffile` - Image processing

## Usage

### Training

Prepare your dataset with the following structure:
```
data_dir/
├── noisy/    # Noisy TPAF images (.npy, .png, or .tif)
├── clean/    # Clean reference images (ground truth)
└── masks/    # Segmentation masks (ground truth)
```

Run Stage 2 joint training:
```bash
python training/train_stage2_joint.py \
    --data_dir /path/to/data \
    --output_dir ./checkpoints \
    --epochs 150 \
    --batch_size 4 \
    --lr 1e-4
```

Training parameters (from paper):
- Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
- Scheduler: CosineAnnealingLR (decay to 1e-6)
- Gradient clipping: max_norm=1.0
- Early stopping: 20 epochs patience on validation Dice

### Model Architecture

```python
from models.afn_deseg import AFNDeSeg

# Initialize model
model = AFNDeSeg(
    img_size=512,
    in_channels=3,
    base_channels=64,
    vit_embed_dim=768,
    vit_depth=12,
    vit_num_heads=12,
    lora_r=16,
    lora_alpha=16
)

# Forward pass
denoised, segmentation = model(input_image)
```

### Loss Functions

```python
from losses.joint_loss import AFNJointLoss

# Initialize joint loss
criterion = AFNJointLoss(
    lambda_rec=1.0,      # Reconstruction loss weight
    lambda_seg=10.0,     # Segmentation loss weight
    lambda_percep=0.1    # Perceptual loss weight
)

# Compute loss
total_loss, loss_dict = criterion(denoised_pred, seg_pred, clean_gt, seg_gt)
```

### Noise Synthesis

```python
from data.mpg_noise_synthesis import MPGNoiseSynthesizer, add_mpg_noise

# Add MPG noise to clean image
noisy = add_mpg_noise(clean_image, gain=2.0, sigma_read=5.0)

# Or use synthesizer with multiple noise levels
synthesizer = MPGNoiseSynthesizer()
noisy, level = synthesizer.synthesize(clean_image)
```

## Model Components

### Dual-Encoder
- **U-Net Encoder**: 4 down-sampling blocks (64→128→256→512 channels)
- **DINOv3 Encoder**: ViT-Base with patch size 16, 768-dim embeddings, 12 transformer layers

### Feature Fusion
- Reshape ViT tokens to spatial format
- Bicubic upsample to match U-Net resolution
- Concatenate (512 + 768 = 1280 channels)
- Fuse via 1×1 + 3×3 convolutions

### Dual-Decoders
- **Denoising Decoder**: Outputs single-channel intensity map
- **Segmentation Decoder**: Outputs binary mask with sigmoid activation

### Loss Function
$$L_{total} = \lambda_{rec} \cdot L_{rec} + \lambda_{seg} \cdot L_{seg} + \lambda_{percep} \cdot L_{percep}$$

Where:
- $L_{rec}$: L1 reconstruction loss
- $L_{seg}$: Dice + BCE segmentation loss
- $L_{percep}$: MSE on frozen Cellpose cyto2 features

## Citation

If you use this code, please cite:

```bibtex
@article{afndeseg2024,
  title={A Joint Denoising and Segmentation Framework for Ovarian Cancer Diagnosis based on Two-Photon Autofluorescence Microscopy},
  author={Pan, Zhengyuan and Song, Naikun and Cheng, Shanshan and Pang, Wen and Liao, Hongen and Wang, Yu and Gu, Bobo},
  journal={},
  year={2024}
}
```

## License

This project is for research purposes.
