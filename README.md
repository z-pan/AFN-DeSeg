# AFN-DeSeg

**Auto-Fluorescence Nuclei Denoising and Segmentation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

A physics-informed deep learning framework for joint image denoising and nuclear segmentation of Two-Photon Autofluorescence (TPAF) microscopy images, with downstream Key Diagnostic Area (KDA) prediction for clinical cancer diagnosis.

## Overview

AFN-DeSeg addresses the challenge of extracting diagnostic-grade nuclear features from noisy TPAF images where nuclei appear as negative contrast (signal voids). The framework combines:

- **Dual-Encoder Architecture**: U-Net for local features + DINOv3 Vision Transformer for global semantic context
- **LoRA Adaptation**: Efficient fine-tuning of the ViT backbone with Low-Rank Adaptation (r=16, α=16)
- **Joint Optimization**: Simultaneous denoising and segmentation with mutual reinforcement
- **Physics-Informed Noise Model**: Mixed Poisson-Gaussian (MPG) noise synthesis based on real TPAF imaging characteristics
- **KDA Prediction**: Attention U-Net for identifying high-tumor-burden regions

## Project Structure

```
AFN-DeSeg/
├── checkpoint/                 # Model weights and checkpoints
│   └── README.md               # Checkpoint documentation
├── configs/                    # Configuration files
│   ├── __init__.py
│   ├── default_config.yaml     # AFN-DeSeg configuration
│   └── kda_config.yaml         # KDA model configuration
├── data/                       # Data processing and noise synthesis
│   ├── __init__.py
│   ├── augmentations.py        # Data augmentation pipelines
│   ├── dataset.py              # Dataset classes
│   └── mpg_noise_synthesis.py  # Physics-informed noise model
├── evaluation/                 # Evaluation scripts
│   ├── __init__.py
│   └── evaluate.py             # Comprehensive evaluation
├── inference/                  # Inference scripts
│   ├── __init__.py
│   ├── predict.py              # AFN-DeSeg prediction
│   └── kda_predictor.py        # KDA prediction
├── losses/                     # Loss functions
│   ├── __init__.py
│   └── joint_loss.py           # Joint denoising-segmentation loss
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── afn_deseg.py            # Main AFN-DeSeg model
│   ├── attention_gate.py       # Attention gate module
│   └── attention_unet.py       # Attention U-Net for KDA
├── paper/                      # Reference paper and supplementary info
├── training/                   # Training scripts
│   ├── __init__.py
│   ├── train_stage1_dino.py    # Stage 1: DINO domain adaptation
│   ├── train_stage2_joint.py   # Stage 2: Joint training
│   └── kda_trainer.py          # KDA model training
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics
│   ├── kda_metrics.py          # KDA-specific metrics
│   └── visualization.py        # Visualization utilities
├── tests/                      # Unit tests
│   ├── test_attention_gate.py
│   ├── test_attention_unet.py
│   ├── test_data.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   └── test_model.py
├── LICENSE                     # MIT License
├── README.md
├── requirements.txt
└── code_availability_statement.md
```

## System Requirements

### Software Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.7 (for GPU acceleration)

Key dependencies:
- `torch`, `torchvision` - Deep learning framework
- `transformers`, `huggingface_hub` - DINOv3 ViT backbone
- `cellpose` - Perceptual loss computation
- `albumentations` - Data augmentation
- `scikit-image`, `tifffile`, `scipy` - Image processing
- `matplotlib` - Visualization

### Hardware Requirements

- **GPU**: NVIDIA GPU with >= 8GB VRAM (recommended: RTX 3090 or A100)
- **RAM**: >= 32GB system memory
- **Storage**: >= 10GB for code, models, and sample data

### Installation Time

- Fresh installation: ~10-15 minutes
- With pre-downloaded model weights: ~5 minutes

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

## Usage

### Quick Start

```python
from models import AFNDeSeg, create_afn_deseg

# Create model with pretrained DINOv3 weights from HuggingFace
model = create_afn_deseg(
    img_size=512,
    pretrained_dinov3=True,
    dinov3_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m"
)

# Or create model without pretrained weights
model = AFNDeSeg(img_size=512)

# Forward pass
denoised, segmentation = model(input_image)
```

### Training

#### Stage 1: DINO Domain Adaptation

```bash
python training/train_stage1_dino.py \
    --data_dir /path/to/unlabeled_tpaf \
    --output_dir ./checkpoint/stage1 \
    --epochs 50 \
    --batch_size 16
```

#### Stage 2: Joint Training

Prepare your dataset with the following structure:
```
data_dir/
├── noisy/    # Noisy TPAF images (.npy, .png, or .tif)
├── clean/    # Clean reference images (ground truth)
└── masks/    # Segmentation masks (ground truth)
```

Run training:
```bash
python training/train_stage2_joint.py \
    --data_dir /path/to/data \
    --output_dir ./checkpoint/stage2 \
    --pretrained_vit ./checkpoint/stage1/best_dino_encoder.pth \
    --epochs 150 \
    --batch_size 4 \
    --lr 1e-4
```

Training parameters (from paper):
- Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
- Scheduler: CosineAnnealingLR (decay to 1e-6)
- Gradient clipping: max_norm=1.0
- Early stopping: 20 epochs patience on validation Dice

#### KDA Model Training

```bash
python training/kda_trainer.py \
    --data_dir /path/to/kda_data \
    --output_dir ./checkpoint/kda \
    --epochs 100 \
    --batch_size 8
```

### Inference

#### AFN-DeSeg Inference

```bash
python inference/predict.py \
    --checkpoint ./checkpoint/stage2/best_model.pth \
    --input /path/to/images \
    --output /path/to/results \
    --visualize
```

#### KDA Prediction

```bash
python inference/kda_predictor.py \
    --checkpoint ./checkpoint/kda/kda_best.pth \
    --input /path/to/nuclear_masks \
    --output /path/to/kda_results
```

### Evaluation

```bash
python evaluation/evaluate.py \
    --checkpoint ./checkpoint/stage2/best_model.pth \
    --data_dir /path/to/test_data \
    --output_dir ./evaluation_results
```

### Running Tests

```bash
pytest tests/ -v
```

## Model Architecture

### AFN-DeSeg: Dual-Encoder

- **U-Net Encoder**: 4 down-sampling blocks (64→128→256→512 channels)
- **DINOv3 Encoder**: ViT-Base with patch size 16, 768-dim embeddings, 12 transformer layers
  - Pretrained weights: `facebook/dinov3-vitb16-pretrain-lvd1689m`
  - LoRA adaptation on Query and Value matrices (r=16, α=16)

### Feature Fusion

- Reshape ViT tokens to spatial format (32×32×768)
- Bicubic upsample to match U-Net resolution
- Concatenate (512 + 768 = 1280 channels)
- Fuse via 1×1 + 3×3 convolutions

### Dual-Decoders

- **Denoising Decoder**: Outputs single-channel intensity map
- **Segmentation Decoder**: Outputs binary mask with sigmoid activation

### Loss Function

$$L_{total} = \lambda_{rec} \cdot L_{rec} + \lambda_{seg} \cdot L_{seg} + \lambda_{percep} \cdot L_{percep}$$

Where:
- $L_{rec}$: L1 reconstruction loss (λ=1.0)
- $L_{seg}$: Dice + BCE segmentation loss (λ=10.0)
- $L_{percep}$: MSE on frozen Cellpose cyto2 features (λ=0.1)

### Attention U-Net for KDA

- **Input**: Binary nuclear segmentation masks from AFN-DeSeg
- **Architecture**: 4-level encoder-decoder with attention gates
- **Output**: Key Diagnostic Area probability map
- **Reference**: Oktay et al., "Attention U-Net", MIDL 2018

## Metrics

The evaluation scripts compute:

**Image Restoration**:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

**Segmentation**:
- Dice Coefficient
- IoU (Intersection over Union)
- mAP@IoU=0.5

**Nuclei Morphology**:
- Nuclear Area
- Circularity
- Density

**Clinical Validation**:
- HD95 (95th percentile Hausdorff Distance)
- Bland-Altman analysis
- Key Area Fraction (KAF)
- Nuclear density correlation (r=0.96)

## Reproducibility

### Random Seeds

All experiments use fixed random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
```

### Expected Runtime

- Training (Stage 2, 150 epochs): ~12-24 hours on single A100 GPU
- Inference (per 512×512 image): ~50-100ms on GPU

## Code Availability

The complete source code is publicly available under the MIT License. See [code_availability_statement.md](code_availability_statement.md) for detailed information about:
- Repository contents
- System requirements
- Reproducibility guidelines
- Pretrained model weights

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DINOv3 pretrained weights from [Facebook Research](https://huggingface.co/collections/facebook/dinov3)
- Cellpose for perceptual loss computation
- Attention U-Net architecture from Oktay et al., MIDL 2018
