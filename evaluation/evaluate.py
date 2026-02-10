"""
Evaluation Script for AFN-DeSeg.

Computes all metrics reported in the paper:
- Image Restoration: PSNR, SSIM
- Segmentation: Dice Coefficient, mAP@IoU=0.5
- Nuclei Morphology: Nuclear Area, Circularity, Density
- Clinical Validation: KAF correlation, Bland-Altman analysis, HD95

Usage:
    python evaluation/evaluate.py \
        --checkpoint /path/to/best_model.pth \
        --data_dir /path/to/test_data \
        --output_dir /path/to/results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import AFNDeSeg
from data import TPAFDataset
from utils.metrics import (
    compute_psnr,
    compute_ssim,
    compute_dice,
    compute_iou,
    compute_map_at_iou,
    compute_hausdorff_distance,
    compute_morphology_stats,
    compute_bland_altman,
    MetricTracker
)


class AFNDeSegEvaluator:
    """
    Evaluator for AFN-DeSeg model.

    Computes comprehensive metrics on test dataset.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        img_size: int = 512
    ):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint.
            device: Device for evaluation ('cuda' or 'cpu').
            img_size: Input image size.
        """
        self.img_size = img_size

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

    def _load_model(self, checkpoint_path: str) -> AFNDeSeg:
        """Load model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")

        model = AFNDeSeg(
            img_size=self.img_size,
            in_channels=3,
            base_channels=64,
            vit_embed_dim=768,
            vit_depth=12,
            vit_num_heads=12,
            lora_r=16,
            lora_alpha=16
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        print(f"Model loaded on {self.device}")

        return model

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_morphology: bool = True,
        compute_clinical: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.

        Args:
            dataloader: Test dataloader.
            compute_morphology: Whether to compute morphology metrics.
            compute_clinical: Whether to compute clinical validation metrics.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()

        # Metric trackers
        psnr_values = []
        ssim_values = []
        dice_values = []
        iou_values = []
        map_values = []
        hd95_values = []

        # Morphology statistics
        pred_areas = []
        pred_circularities = []
        pred_counts = []
        gt_areas = []
        gt_circularities = []
        gt_counts = []

        print("Evaluating...")
        for batch in tqdm(dataloader, desc="Evaluation"):
            noisy, clean, mask = batch[:3]

            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            mask = mask.to(self.device)

            # Preprocess and forward
            noisy_rgb = AFNDeSeg.preprocess_tpaf(noisy, normalize=True)
            denoised, seg_pred = self.model(noisy_rgb)

            # Compute metrics for each sample in batch
            for i in range(noisy.shape[0]):
                # Image restoration metrics
                psnr = compute_psnr(denoised[i], clean[i])
                ssim = compute_ssim(denoised[i], clean[i])
                psnr_values.append(psnr)
                ssim_values.append(ssim)

                # Segmentation metrics
                pred_np = seg_pred[i].cpu().numpy().squeeze()
                mask_np = mask[i].cpu().numpy().squeeze()

                dice = compute_dice(pred_np, mask_np)
                iou = compute_iou(pred_np, mask_np)
                map_score = compute_map_at_iou(pred_np, mask_np, iou_threshold=0.5)

                dice_values.append(dice)
                iou_values.append(iou)
                map_values.append(map_score)

                # Clinical metrics
                if compute_clinical:
                    hd95 = compute_hausdorff_distance(pred_np, mask_np, percentile=95)
                    if not np.isnan(hd95):
                        hd95_values.append(hd95)

                # Morphology metrics
                if compute_morphology:
                    pred_stats = compute_morphology_stats(pred_np)
                    gt_stats = compute_morphology_stats(mask_np)

                    if 'area_mean' in pred_stats:
                        pred_areas.append(pred_stats['area_mean'])
                    if 'circularity_mean' in pred_stats:
                        pred_circularities.append(pred_stats['circularity_mean'])
                    pred_counts.append(pred_stats['count'])

                    if 'area_mean' in gt_stats:
                        gt_areas.append(gt_stats['area_mean'])
                    if 'circularity_mean' in gt_stats:
                        gt_circularities.append(gt_stats['circularity_mean'])
                    gt_counts.append(gt_stats['count'])

        # Aggregate metrics
        results = {
            # Image Restoration
            'psnr_mean': float(np.mean(psnr_values)),
            'psnr_std': float(np.std(psnr_values)),
            'ssim_mean': float(np.mean(ssim_values)),
            'ssim_std': float(np.std(ssim_values)),

            # Segmentation
            'dice_mean': float(np.mean(dice_values)),
            'dice_std': float(np.std(dice_values)),
            'iou_mean': float(np.mean(iou_values)),
            'iou_std': float(np.std(iou_values)),
            'map_mean': float(np.mean(map_values)),
            'map_std': float(np.std(map_values)),
        }

        # Clinical metrics
        if compute_clinical and hd95_values:
            results['hd95_mean'] = float(np.mean(hd95_values))
            results['hd95_std'] = float(np.std(hd95_values))
            results['hd95_median'] = float(np.median(hd95_values))

        # Morphology correlation
        if compute_morphology and pred_counts and gt_counts:
            # Nuclear density correlation
            pred_counts = np.array(pred_counts)
            gt_counts = np.array(gt_counts)
            if len(pred_counts) > 1:
                density_corr = np.corrcoef(pred_counts, gt_counts)[0, 1]
                results['density_correlation'] = float(density_corr)

                # Bland-Altman for density
                ba_density = compute_bland_altman(pred_counts, gt_counts)
                results['density_bias'] = ba_density['bias']
                results['density_loa_lower'] = ba_density['loa_lower']
                results['density_loa_upper'] = ba_density['loa_upper']

            # Area statistics
            if pred_areas and gt_areas:
                results['pred_area_mean'] = float(np.mean(pred_areas))
                results['gt_area_mean'] = float(np.mean(gt_areas))

            # Circularity statistics
            if pred_circularities and gt_circularities:
                results['pred_circularity_mean'] = float(np.mean(pred_circularities))
                results['gt_circularity_mean'] = float(np.mean(gt_circularities))

        return results

    def evaluate_by_noise_level(
        self,
        dataloaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model across different noise levels.

        Args:
            dataloaders: Dictionary mapping noise level names to dataloaders.

        Returns:
            Dictionary of results per noise level.
        """
        results = {}
        for level_name, dataloader in dataloaders.items():
            print(f"\nEvaluating noise level: {level_name}")
            results[level_name] = self.evaluate(
                dataloader,
                compute_morphology=False,
                compute_clinical=False
            )
        return results


def print_results(results: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

    # Group by category
    categories = {
        "Image Restoration": ['psnr', 'ssim'],
        "Segmentation": ['dice', 'iou', 'map'],
        "Clinical (HD95)": ['hd95'],
        "Morphology": ['density', 'area', 'circularity']
    }

    for category, prefixes in categories.items():
        relevant = {k: v for k, v in results.items()
                   if any(k.startswith(p) for p in prefixes)}
        if relevant:
            print(f"\n{category}:")
            for key, value in relevant.items():
                print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='AFN-DeSeg Evaluation Script'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--no_morphology', action='store_true',
                        help='Skip morphology metrics')
    parser.add_argument('--no_clinical', action='store_true',
                        help='Skip clinical metrics')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = AFNDeSegEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        img_size=args.img_size
    )

    # Create test dataset
    test_dataset = TPAFDataset(
        args.data_dir,
        split='test',
        img_size=args.img_size,
        transform=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test samples: {len(test_dataset)}")

    # Run evaluation
    results = evaluator.evaluate(
        test_loader,
        compute_morphology=not args.no_morphology,
        compute_clinical=not args.no_clinical
    )

    # Print results
    print_results(results)

    # Save results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save summary table (for paper)
    summary_file = output_dir / 'summary_table.txt'
    with open(summary_file, 'w') as f:
        f.write("AFN-DeSeg Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB\n")
        f.write(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}\n")
        f.write(f"Dice: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}\n")
        f.write(f"mAP@0.5: {results['map_mean']:.4f} ± {results['map_std']:.4f}\n")
        if 'hd95_median' in results:
            f.write(f"HD95: {results['hd95_median']:.2f} pixels (median)\n")
        if 'density_correlation' in results:
            f.write(f"Density Correlation: {results['density_correlation']:.4f}\n")
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
