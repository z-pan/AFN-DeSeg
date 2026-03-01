#!/usr/bin/env python3
"""
Step 2: Generate synthetic noisy training images from clean TPAF images.

Wraps noise_scripts/generate_noisy.py with pre-flight validation and a
post-generation file-count / range sanity check.

What this step does
-------------------
For each clean image (typically the avg16 frame used as the denoising ground
truth), this script synthesises one or more noisy copies by injecting
Poisson-Gaussian noise calibrated to your specific microscope (the ``a`` and
``b`` parameters estimated in Step 1).

Multiple ``--n_frames`` levels are supported simultaneously:

    n_frames=1   → noisiest  (simulates a single-frame acquisition)
    n_frames=4   → moderate noise
    n_frames=8   → low noise (still visibly noisier than the clean reference)

For Stage 2 training it is conventional to use **n_frames=1** as the noisy
input to maximise denoising difficulty.  Levels 4 and 8 may also be added
to the training set for data augmentation.

Usage
-----
::

    python data_prep/step2_generate_noisy.py \\
        --clean_dir   /path/to/clean_images \\
        --params_file outputs/noise_estimation/noise_params.json \\
        --output_dir  outputs/noisy_images \\
        --n_frames 1 4 8 \\
        --seed 42

Output layout
-------------
::

    output_dir/
        n_frames_1/       # one noisy copy per clean image, highest noise
            img_001.tif
            img_002.tif
            …
        n_frames_4/
            …
        n_frames_8/
            …
        comparisons/      # side-by-side PNG panels (requires --save_comparison)
            img_001.png
            …

After running, MANUALLY check
------------------------------
1. **Visual QC** — Open 3–5 comparison PNGs from ``output_dir/comparisons/``.

   - The ``n_frames_1`` column must look visibly noisy (granular/speckled
     texture matching real 1-frame acquisitions from your microscope).
   - Noise should be spatially uncorrelated — no structured bands or stripes.
   - Brighter image regions should look noisier than dark regions
     (this is the Poisson property: σ² ∝ signal).
   - The ``n_frames_8`` column should look noticeably cleaner than
     ``n_frames_1``, though still noisier than the clean reference.

2. **Count check** — The script prints a count table; verify that every clean
   image has a corresponding noisy file in every requested level directory.

3. If the noise looks too weak or too strong, revisit Step 1 and check that
   the estimated ``a`` and ``b`` match your imaging conditions.  A common
   mistake is providing images in the wrong intensity unit (photon counts vs.
   ADU, or uint16 vs. float32 normalised to [0, 1]).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_SUPPORTED_EXT = {'.tif', '.tiff', '.npy'}

BANNER = "=" * 64


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='step2_generate_noisy',
        description='Generate synthetic noisy images from clean TPAF images (Step 2 of data prep).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--clean_dir', required=True,
        help='Directory containing clean (avg16) TPAF images (.tif / .npy).',
    )
    p.add_argument(
        '--params_file', required=True,
        help='Path to noise_params.json produced by Step 1.',
    )
    p.add_argument(
        '--output_dir', default='outputs/noisy_images',
        help='Root directory for synthetic noisy images.',
    )
    p.add_argument(
        '--n_frames', nargs='+', type=float, default=[1, 4, 8],
        metavar='N',
        help='Equivalent frame-count levels to simulate (space-separated). '
             'Use 1 for Stage 2 training; additional levels give augmentation data.',
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help='Base random seed for reproducibility.',
    )
    p.add_argument(
        '--save_comparison', action='store_true', default=True,
        help='Save side-by-side PNG panels for visual QC (strongly recommended).',
    )
    p.add_argument(
        '--comparison_crop', type=int, default=256,
        metavar='PIXELS',
        help='Crop to this central square for comparison panels.',
    )
    p.add_argument('--verbose', action='store_true', default=False,
                   help='Pass --verbose to the underlying generation script.')
    return p


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _preflight(args) -> bool:
    """Validate inputs before starting the (potentially slow) generation."""
    ok = True

    # Check clean_dir
    clean_path = Path(args.clean_dir)
    if not clean_path.exists():
        print(f"[ERROR] --clean_dir does not exist: {clean_path}")
        ok = False
    else:
        images = [f for f in clean_path.iterdir()
                  if f.suffix.lower() in _SUPPORTED_EXT]
        if not images:
            print(f"[ERROR] No .tif / .npy images found in {clean_path}")
            ok = False
        else:
            print(f"\n{BANNER}")
            print("PRE-FLIGHT CHECK")
            print(f"  Clean images found : {len(images)}")

    # Check params_file
    params_path = Path(args.params_file)
    if not params_path.exists():
        print(f"  [ERROR] --params_file does not exist: {params_path}")
        ok = False
    else:
        with open(params_path) as fh:
            p = json.load(fh)
        print(f"  Noise params       : a={p['a']:.4f}  b={p['b']:.4f}  R²={p['r_squared']:.4f}")
        if p['r_squared'] < 0.80:
            print(f"  [WARN] R² = {p['r_squared']:.4f} < 0.80 — noise parameters are unreliable.")
            print("         Re-run Step 1 with more calibration images before continuing.")

    if ok:
        print(f"  Output dir         : {args.output_dir}")
        print(f"  n_frames levels    : {args.n_frames}")
        print(f"  Seed               : {args.seed}")
        print(BANNER)

    return ok


# ---------------------------------------------------------------------------
# Run generation
# ---------------------------------------------------------------------------

def _run_generation(args) -> int:
    script = _PROJECT_ROOT / 'noise_scripts' / 'generate_noisy.py'
    cmd = [
        sys.executable, str(script),
        '--clean_dir', args.clean_dir,
        '--params_file', args.params_file,
        '--output_dir', args.output_dir,
        '--n_frames', *[str(n) for n in args.n_frames],
        '--seed', str(args.seed),
        '--output_format', 'tif',
    ]
    if args.save_comparison:
        cmd += ['--save_comparison', '--comparison_crop', str(args.comparison_crop)]
    if args.verbose:
        cmd.append('--verbose')

    print(f"\n{BANNER}")
    print("RUNNING: noise_scripts/generate_noisy.py")
    print(BANNER)

    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Post-generation checks
# ---------------------------------------------------------------------------

def _post_check(clean_dir: str, output_dir: str, n_frames: list) -> bool:
    """Count generated files and flag mismatches."""
    clean_path = Path(clean_dir)
    out_path   = Path(output_dir)

    clean_stems = {
        f.stem for f in clean_path.iterdir()
        if f.suffix.lower() in _SUPPORTED_EXT
    }
    n_clean = len(clean_stems)

    print(f"\n{BANNER}")
    print("POST-GENERATION FILE COUNT")
    print(f"  Clean images (source) : {n_clean}")
    print()

    all_ok = True
    for nf in sorted(n_frames):
        label = f'n_frames_{nf:.0f}' if nf == int(nf) else f'n_frames_{nf:.2f}'
        level_dir = out_path / label

        if not level_dir.exists():
            print(f"  [MISSING] {label}/  — directory was not created!")
            all_ok = False
            continue

        noisy_stems = {
            f.stem for f in level_dir.iterdir()
            if f.suffix.lower() in _SUPPORTED_EXT
        }
        n_noisy   = len(noisy_stems)
        missing   = clean_stems - noisy_stems
        extra     = noisy_stems - clean_stems

        if not missing and not extra:
            print(f"  [OK]      {label}/  {n_noisy} images  (all matched)")
        else:
            all_ok = False
            print(f"  [WARN]    {label}/  {n_noisy}/{n_clean} images")
            if missing:
                shown = sorted(missing)[:5]
                print(f"            Missing: {shown}"
                      + ("  …" if len(missing) > 5 else ""))
            if extra:
                shown = sorted(extra)[:5]
                print(f"            Extra  : {shown}"
                      + ("  …" if len(extra) > 5 else ""))

    comp_dir = out_path / 'comparisons'
    print(BANNER)
    print("\n*** MANUAL CHECKS REQUIRED ***")
    print()
    if comp_dir.exists():
        print(f"1. Open comparison panels in:  {comp_dir}/")
        print("   - n_frames_1 column must look visibly noisy (granular/speckled).")
        print("   - Noise should be spatially uncorrelated (no bands or stripes).")
        print("   - Bright regions should appear noisier than dark regions")
        print("     (Poisson property: higher signal → higher variance).")
        print("   - n_frames_8 should look cleaner than n_frames_1.")
    else:
        print("1. Re-run with --save_comparison to generate visual QC panels.")
    print()
    print("2. Compare a few noisy images against real single-frame acquisitions")
    print("   from your microscope (if available) to verify the noise texture")
    print("   looks realistic.  Use Fiji / napari for inspection.")
    print()
    print("3. If noise looks too strong or too weak:")
    print("   - Check that clean images are in the same intensity units as the")
    print("     calibration images used in Step 1.")
    print("   - A common mistake: clean images normalised to [0, 1] while")
    print("     calibration images were in raw ADU counts.")
    print(BANNER)

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not _preflight(args):
        return 1

    rc = _run_generation(args)
    if rc != 0:
        print(f"\n[ERROR] generate_noisy.py exited with code {rc}.")
        print("        Check the output above for error messages.")
        return rc

    ok = _post_check(args.clean_dir, args.output_dir, args.n_frames)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
