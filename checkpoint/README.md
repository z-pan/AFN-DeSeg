# Model Checkpoints

This directory contains pretrained model weights for AFN-DeSeg.

## Expected Files

| File | Description |
|------|-------------|
| `afn_deseg_best.pth` | Best AFN-DeSeg model (Stage 2 joint training) |
| `dino_encoder_stage1.pth` | DINOv3 encoder after Stage 1 domain adaptation |
| `dinov3-vitb16-pretrain-lvd1689m/` | DINOv3 pretrained weights from HuggingFace |

## Checkpoint Format

Model checkpoints are saved in PyTorch format with the following structure:

```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'best_loss': best_loss,
    'best_dice': best_dice,
    'config': {
        'img_size': 512,
        'in_channels': 3,
        'base_channels': 64,
        'vit_embed_dim': 768,
        'vit_depth': 12,
        'vit_num_heads': 12,
        'lora_r': 16,
        'lora_alpha': 16
    }
}
```

## Loading Checkpoints

```python
from models import AFNDeSeg

# Load model
model = AFNDeSeg(img_size=512)
checkpoint = torch.load('checkpoint/afn_deseg_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Downloading DINOv3 Weights

DINOv3 weights are automatically downloaded from HuggingFace when using `pretrained_dinov3=True`:

```python
from models import create_afn_deseg

model = create_afn_deseg(
    img_size=512,
    pretrained_dinov3=True,
    dinov3_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m"
)
```

Alternatively, download manually:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
model.save_pretrained("checkpoint/dinov3-vitb16-pretrain-lvd1689m")
```

## Note

Large checkpoint files (`.pth`, `.safetensors`) are excluded from git tracking.
Download or generate them using the training scripts.
