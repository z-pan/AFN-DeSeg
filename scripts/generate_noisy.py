#!/usr/bin/env python3
"""
CLI entry point for synthetic Poisson-Gaussian noise generation.

Reads clean TPAF images from a directory, loads previously estimated noise
parameters, and saves synthetic noisy images at one or more equivalent
frame-count levels.

Usage
-----
::

    python scripts/generate_noisy.py \\
        --clean_dir  data/clean_images/ \\
        --params_file  outputs/noise_params.json \\
        --n_frames  1 4 8 16 \\
        --output_dir  outputs/synthetic_noisy/ \\
        --seed  42

Arguments
---------
See ``python scripts/generate_noisy.py --help`` for the full list.

Output layout
-------------
For each clean image and each requested *n_frames* value, the script writes::

    output_dir/
        n_frames_1/
            image_001.tif
            image_002.tif
            …
        n_frames_4/
            image_001.tif
            …
        n_frames_8/
            …
        n_frames_16/
            …

If ``--save_comparison`` is given, a visual comparison PNG is written
next to each clean image.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.noise_generator import PoissonGaussianGenerator
from src.visualization import plot_noise_level_comparison


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
        datefmt='%H:%M:%S',
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger('generate_noisy')

# Supported input extensions
_INPUT_EXTENSIONS = {'.tif', '.tiff', '.npy'}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='generate_noisy',
        description='Generate synthetic Poisson-Gaussian noisy images from clean TPAF images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- I/O ----
    p.add_argument(
        '--clean_dir', required=True,
        help='Directory containing clean (noise-free) TPAF images (.tif / .npy).',
    )
    p.add_argument(
        '--params_file', required=True,
        help='Path to noise_params.json produced by estimate_noise.py.',
    )
    p.add_argument(
        '--output_dir', default='outputs/synthetic_noisy',
        help='Root directory for synthetic noisy images.',
    )

    # ---- Noise settings ----
    p.add_argument(
        '--n_frames', nargs='+', type=float, default=[1, 4, 8, 16],
        metavar='N',
        help='Equivalent frame-count levels to simulate (space-separated). '
             'Non-integer values are accepted (e.g. --n_frames 1 2.5 8).',
    )
    p.add_argument(
        '--seed', type=int, default=None,
        help='Base random seed for reproducibility. Per-image seeds are '
             'derived as seed + image_index.',
    )

    # ---- Output options ----
    p.add_argument(
        '--output_format', choices=['tif', 'npy'], default='tif',
        help='Output file format for synthetic noisy images.',
    )
    p.add_argument(
        '--save_comparison', action='store_true', default=False,
        help='For each clean image, save a side-by-side PNG comparison of '
             'all noise levels alongside the clean reference.',
    )
    p.add_argument(
        '--comparison_crop', type=int, default=None,
        metavar='PIXELS',
        help='If set, crop to a central square of this size for comparison plots.',
    )

    # ---- Misc ----
    p.add_argument(
        '--recursive', action='store_true', default=False,
        help='Recursively search clean_dir for image files.',
    )
    p.add_argument('--verbose', action='store_true', default=False,
                   help='Enable DEBUG-level logging.')

    return p


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def _collect_image_paths(clean_dir: str, recursive: bool) -> List[Path]:
    """Return all supported image files in *clean_dir* (sorted)."""
    root = Path(clean_dir)
    paths: List[Path] = []
    pattern = '**/*' if recursive else '*'
    for ext in _INPUT_EXTENSIONS:
        paths.extend(root.glob(f'{pattern}{ext}'))
        paths.extend(root.glob(f'{pattern}{ext.upper()}'))
    return sorted(set(paths))


def _load_image(path: Path) -> np.ndarray:
    """Load a TIFF or .npy image as float64."""
    if path.suffix.lower() == '.npy':
        return np.load(str(path))
    return tifffile.imread(str(path))


def _save_image(arr: np.ndarray, path: Path, fmt: str) -> None:
    """Save *arr* to *path* in the requested format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'npy':
        np.save(str(path.with_suffix('.npy')), arr)
    else:
        tifffile.imwrite(str(path.with_suffix('.tif')), arr, compression='deflate')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Run noise generation.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Poisson-Gaussian Noise Generation")
    logger.info("=" * 60)
    logger.info("clean_dir    : %s", args.clean_dir)
    logger.info("params_file  : %s", args.params_file)
    logger.info("output_dir   : %s", output_dir)
    logger.info("n_frames     : %s", args.n_frames)
    logger.info("seed         : %s", args.seed)
    logger.info("format       : %s", args.output_format)

    # ------------------------------------------------------------------
    # Load generator
    # ------------------------------------------------------------------
    try:
        gen = PoissonGaussianGenerator.from_params_file(args.params_file)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Failed to load params from '%s': %s", args.params_file, exc)
        return 1

    logger.info("Generator: a=%.4f, b=%.4f", gen.a, gen.b)

    # Print predicted SNR table
    logger.info("Predicted SNR at mean=5000 ADU:")
    for nf in sorted(args.n_frames):
        snr = gen.theoretical_snr(5000.0, n_frames=nf)
        logger.info("  n_frames=%-5.0f  SNR=%.1f", nf, snr)

    # ------------------------------------------------------------------
    # Collect input images
    # ------------------------------------------------------------------
    image_paths = _collect_image_paths(args.clean_dir, args.recursive)
    if not image_paths:
        logger.error(
            "No image files found in '%s' (extensions: %s)",
            args.clean_dir, _INPUT_EXTENSIONS,
        )
        return 1

    logger.info("Found %d clean image(s) to process.", len(image_paths))

    # ------------------------------------------------------------------
    # Create output subdirectory per n_frames level
    # ------------------------------------------------------------------
    n_frames_sorted = sorted(set(float(v) for v in args.n_frames))
    level_dirs: Dict[float, Path] = {}
    for nf in n_frames_sorted:
        label = f'n_frames_{nf:.0f}' if nf == int(nf) else f'n_frames_{nf:.2f}'
        d = output_dir / label
        d.mkdir(parents=True, exist_ok=True)
        level_dirs[nf] = d

    # ------------------------------------------------------------------
    # Process each image
    # ------------------------------------------------------------------
    n_ok = 0
    n_err = 0
    clean_dir_root = Path(args.clean_dir)

    for img_idx, img_path in enumerate(image_paths):
        # Derive relative stem (preserve sub-structure when recursive)
        try:
            rel_stem = img_path.relative_to(clean_dir_root).with_suffix('')
        except ValueError:
            rel_stem = Path(img_path.stem)

        logger.info("[%d/%d] Processing '%s' …", img_idx + 1, len(image_paths), img_path.name)

        try:
            clean = _load_image(img_path)
        except Exception as exc:
            logger.warning("  Failed to load '%s': %s — skipping.", img_path, exc)
            n_err += 1
            continue

        orig_dtype = clean.dtype

        # Per-image seed derived from base seed
        per_image_seed = None if args.seed is None else args.seed + img_idx

        # Generate noisy images at all requested levels
        noisy_dict: Dict[float, np.ndarray] = gen.generate_noise_levels(
            clean_image=clean,
            n_frames_list=n_frames_sorted,
            seed=per_image_seed,
        )

        # Save per-level images
        for nf, noisy in noisy_dict.items():
            out_path = level_dirs[nf] / rel_stem
            try:
                _save_image(noisy, out_path, args.output_format)
                logger.debug("  Saved n_frames=%.0f → '%s'", nf, out_path)
            except Exception as exc:
                logger.warning("  Failed to save n_frames=%.0f '%s': %s", nf, out_path, exc)
                n_err += 1

        # Optional side-by-side comparison plot
        if args.save_comparison:
            try:
                import matplotlib.pyplot as plt
                fig = plot_noise_level_comparison(
                    clean_image=clean.astype(np.float64),
                    noisy_images=noisy_dict,
                    output_path=str(output_dir / 'comparisons' / f'{rel_stem}.png'),
                    crop=args.comparison_crop,
                )
                plt.close(fig)
            except Exception as exc:
                logger.warning("  Comparison plot failed for '%s': %s", img_path.name, exc)

        n_ok += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Generation complete.")
    logger.info("  Processed : %d image(s)", n_ok)
    logger.info("  Errors    : %d", n_err)
    logger.info("  Outputs   : %s", output_dir)
    logger.info("  Levels    : %s", [f'n={nf:.0f}' for nf in n_frames_sorted])
    logger.info("=" * 60)

    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
