#!/usr/bin/env python3
"""
Step 1b: Select qualifying 512×512 patches from TPAF images.

Slides a fixed-size window over each source TPAF image and keeps only patches
where the Cellpose instance mask contains at least ``--min_nuclei`` distinct
nuclei (unique non-zero label values in the mask patch).

Sources
-------
**WSI folders** (``--wsi_dirs``):
    Co-located clean TPAF images (``.tif``) and Cellpose masks
    (``*_cp_masks.png``).  Qualifying patches are saved to ``clean/`` and
    ``masks/`` under ``--output_dir``.

**Calibration avg_16 folder** (``--avg16_dir``):
    Clean TPAF images with co-located Cellpose masks.  For each qualifying
    window the matching file from ``--avg1_dir`` (the noisy single-frame
    acquisition) is also extracted.  All three outputs are saved:

    * avg_16 patch → ``clean/``
    * avg_16 mask patch → ``masks/``
    * avg_1 patch (same spatial window) → ``noisy/``

This produces the pool of labelled patches consumed by Step 3
(``step3_build_splits.py``).

Stride selection
----------------
Default stride is **256 px** (50 % overlap with the 512 px window).

* A stride of 512 (no overlap) risks missing cell-dense clusters that
  straddle adjacent window boundaries, which would cause qualifying regions
  to be silently excluded.
* A stride of 256 ensures every 512×512 area appears intact in at least one
  window.  The ≥50-nucleus filter discards sparse / background patches, so
  50 % overlap does not cause prohibitive dataset bloat in practice.
* A stride of 384 (25 % overlap) is an acceptable lower-redundancy
  alternative for very large images (> 4096 px) where runtime matters.

File naming
-----------
Patches are named ``{source}_{img_stem}_x{col:04d}_y{row:04d}`` where
``col`` and ``row`` are the zero-indexed pixel coordinates of the window's
top-left corner in the original image.  The source prefix (WSI directory
name or ``avg16``) prevents stem collisions across input directories.

Usage
-----
::

    python data_prep/step1b_select_patches.py \\
        --wsi_dirs    /path/to/260227_WSI_00 \\
                      /path/to/260227_WSI_01 \\
                      /path/to/260227_WSI_02 \\
                      /path/to/260227_WSI_03 \\
        --avg16_dir   /path/to/avg_16 \\
        --avg1_dir    /path/to/avg_1 \\
        --output_dir  outputs/patches

    # Inspect counts without writing any files:
    python data_prep/step1b_select_patches.py \\
        --wsi_dirs ...  --avg16_dir ...  --avg1_dir ... \\
        --output_dir outputs/patches --dry_run

Outputs
-------
``output_dir/clean/``
    8-bit grayscale TIFF patches from WSI and avg_16 sources.
``output_dir/masks/``
    16-bit PNG instance-mask patches (Cellpose labels preserved).
``output_dir/noisy/``
    8-bit grayscale TIFF patches from avg_1 (aligned with avg_16 windows).

Downstream pipeline
--------------------
Step 1b outputs feed into **Step 2** before Step 3.  ``clean/`` patches serve
as the clean source for synthetic noisy generation; ``noisy/`` (real avg_1
patches) can be merged with the Step 2 output as an additional noisy source::

    # Step 2 — generate synthetic noisy from all clean patches
    python data_prep/step2_generate_noisy.py \\
        --clean_dir   outputs/patches/clean \\
        --params_file outputs/noise_estimation/noise_params.json \\
        --output_dir  outputs/patches_noisy \\
        --n_frames 1

    # Step 3 — assemble train/val splits
    #   noisy source A: synthetic noisy (from Step 2, covers all clean patches)
    #   noisy source B: real avg_1 patches (noisy/, covers avg_16 subset only)
    #   Run Step 3 twice with the same --output_dir to combine both noisy sources:
    python data_prep/step3_build_splits.py \\
        --noisy_dir  outputs/patches_noisy/n_frames_1 \\
        --clean_dir  outputs/patches/clean \\
        --mask_dir   outputs/patches/masks \\
        --output_dir outputs/stage2_data --seed 42

    python data_prep/step3_build_splits.py \\
        --noisy_dir  outputs/patches/noisy \\
        --clean_dir  outputs/patches/clean \\
        --mask_dir   outputs/patches/masks \\
        --output_dir outputs/stage2_data --seed 42

After running, MANUALLY check
------------------------------
1. File counts match expectations::

       ls outputs/patches/clean | wc -l   # total clean patches (WSI + avg_16)
       ls outputs/patches/masks | wc -l   # must equal clean count
       ls outputs/patches/noisy | wc -l   # avg_16 contribution only (no WSI)

2. Open 3–5 patch triplets in Fiji or Python and confirm:

   - Images are 512×512 uint8 grayscale TIFF.
   - Masks are 512×512 uint16 PNG (Cellpose instance labels intact).
   - avg_1 (noisy/) and avg_16 (clean/) patches are spatially aligned.

3. Proceed to Step 2 to generate synthetic noisy images from clean/ patches
   before building train/val splits with Step 3.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    from skimage import io as skimage_io
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

BANNER = "=" * 64

# Suffixes Cellpose appends to image stems — these are not TPAF images.
_CELLPOSE_STEM_SUFFIXES = ('_seg', '_cp_masks', '_cp_output', '_flows')

# Recognised TPAF image extensions.
_IMG_EXT = {'.tif', '.tiff'}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _is_tpaf_image(path: Path) -> bool:
    """Return True if *path* is a TPAF image (not a Cellpose output file)."""
    return (
        path.suffix.lower() in _IMG_EXT
        and not any(path.stem.endswith(s) for s in _CELLPOSE_STEM_SUFFIXES)
    )


def _find_mask(img_path: Path, mask_suffix: str) -> Optional[Path]:
    """Return the Cellpose mask path for *img_path*, or None if not found."""
    for ext in ('.png', '.tif', '.tiff'):
        candidate = img_path.with_name(img_path.stem + mask_suffix + ext)
        if candidate.exists():
            return candidate
    return None


def _load_image(path: Path) -> np.ndarray:
    """Load a TIFF image, preserving its native dtype (typically uint8)."""
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required — install with: pip install tifffile")
    return tifffile.imread(str(path))


def _load_mask(path: Path) -> np.ndarray:
    """Load a Cellpose mask (PNG or TIFF), preserving native uint16 labels."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        if HAS_TIFFFILE:
            return tifffile.imread(str(path))
        if HAS_SKIMAGE:
            return skimage_io.imread(str(path))
    else:  # .png or any other format
        if HAS_SKIMAGE:
            return skimage_io.imread(str(path))
        if HAS_TIFFFILE:
            # tifffile cannot read PNG; fall back to PIL if available
            try:
                from PIL import Image
                return np.array(Image.open(str(path)))
            except ImportError:
                pass
    raise ImportError(
        "tifffile or scikit-image is required — "
        "install with: pip install tifffile scikit-image"
    )


def _save_image(patch: np.ndarray, path: Path) -> None:
    """Save a patch as uint8 TIFF (8-bit grayscale), matching the original format."""
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required — install with: pip install tifffile")
    tifffile.imwrite(str(path), patch.astype(np.uint8))


def _save_mask(patch: np.ndarray, path: Path) -> None:
    """Save a mask patch as uint16 PNG, preserving Cellpose instance labels."""
    if not HAS_SKIMAGE:
        raise ImportError(
            "scikit-image is required — install with: pip install scikit-image"
        )
    skimage_io.imsave(str(path), patch.astype(np.uint16), check_contrast=False)


# ---------------------------------------------------------------------------
# Nucleus counting and sliding window
# ---------------------------------------------------------------------------

def _count_nuclei(mask_patch: np.ndarray) -> int:
    """Count distinct nuclei (unique non-zero label values) in a mask patch."""
    unique = np.unique(mask_patch)
    return int(np.count_nonzero(unique))  # background label 0 is excluded


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Return a 2-D view of *arr* (take first channel if multi-channel)."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # (H, W, C) → take first channel; (C, H, W) handled by first-channel too
        if arr.shape[-1] <= 4:      # channels-last
            return arr[..., 0]
        return arr[0]               # channels-first
    raise ValueError(f"Unexpected array shape: {arr.shape}")


def _sliding_windows(
    H: int, W: int, window_size: int, stride: int
) -> Iterator[Tuple[int, int]]:
    """
    Yield (row, col) top-left corners of all sliding windows.

    An edge-aligned window is appended along each axis when the last
    stride-aligned position does not reach the image boundary, guaranteeing
    full spatial coverage without zero-padding.

    Nothing is yielded for images smaller than *window_size* in either
    dimension (the caller is responsible for the warning/skip).
    """
    if H < window_size or W < window_size:
        return

    rows = list(range(0, H - window_size + 1, stride))
    cols = list(range(0, W - window_size + 1, stride))

    last_row = H - window_size
    if rows[-1] != last_row:
        rows.append(last_row)

    last_col = W - window_size
    if cols[-1] != last_col:
        cols.append(last_col)

    for r in rows:
        for c in cols:
            yield r, c


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

def _preflight(args: argparse.Namespace) -> bool:
    """Validate all inputs before touching the filesystem. Returns False on error."""
    ok = True

    if not HAS_TIFFFILE:
        print("[ERROR] tifffile not found — install with: pip install tifffile")
        ok = False
    if not HAS_SKIMAGE:
        print("[ERROR] scikit-image not found — install with: pip install scikit-image")
        ok = False

    for raw in args.wsi_dirs:
        d = Path(raw)
        if not d.is_dir():
            print(f"[ERROR] WSI directory not found: {d}")
            ok = False

    for flag, raw in [('--avg16_dir', args.avg16_dir), ('--avg1_dir', args.avg1_dir)]:
        d = Path(raw)
        if not d.is_dir():
            print(f"[ERROR] {flag} directory not found: {d}")
            ok = False

    if args.window_size < 1:
        print(f"[ERROR] --window_size must be ≥ 1 (got {args.window_size})")
        ok = False
    if args.stride < 1:
        print(f"[ERROR] --stride must be ≥ 1 (got {args.stride})")
        ok = False
    if args.min_nuclei < 0:
        print(f"[ERROR] --min_nuclei must be ≥ 0 (got {args.min_nuclei})")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Core processing — WSI directories
# ---------------------------------------------------------------------------

def _process_wsi_dirs(
    wsi_dirs: List[Path],
    clean_dir: Path,
    masks_dir: Path,
    window_size: int,
    stride: int,
    min_nuclei: int,
    mask_suffix: str,
    dry_run: bool,
    verbose: bool,
) -> Tuple[int, int, int]:
    """
    Process WSI source directories → clean/ and masks/.

    Returns
    -------
    (n_images_processed, n_windows_evaluated, n_patches_saved)
    """
    n_images = n_checked = n_saved = 0

    for wsi_dir in wsi_dirs:
        tpaf_files = sorted(f for f in wsi_dir.iterdir() if _is_tpaf_image(f))
        if not tpaf_files:
            print(f"  [WARN] No TPAF images found in {wsi_dir} — skipped")
            continue

        print(f"\n  {wsi_dir.name}  ({len(tpaf_files)} TPAF image(s))")

        for img_path in tpaf_files:
            mask_path = _find_mask(img_path, mask_suffix)
            if mask_path is None:
                print(f"    [WARN] No mask for {img_path.name} — skipped")
                continue

            img  = _to_2d(_load_image(img_path))
            mask = _to_2d(_load_mask(mask_path))
            H, W = img.shape

            if H < window_size or W < window_size:
                print(f"    [WARN] {img_path.name} ({H}×{W}) is smaller than "
                      f"the {window_size}×{window_size} window — skipped")
                continue

            n_images += 1
            windows = list(_sliding_windows(H, W, window_size, stride))
            img_saved = 0

            for row, col in windows:
                n_checked += 1
                img_patch  = img [row:row + window_size, col:col + window_size]
                mask_patch = mask[row:row + window_size, col:col + window_size]
                n_nuc = _count_nuclei(mask_patch)

                if verbose:
                    kept = "  kept" if n_nuc >= min_nuclei else ""
                    print(f"      x={col:04d} y={row:04d}  nuclei={n_nuc}{kept}")

                if n_nuc < min_nuclei:
                    continue

                stem = f"{wsi_dir.name}_{img_path.stem}_x{col:04d}_y{row:04d}"
                if not dry_run:
                    _save_image(img_patch,  clean_dir / f"{stem}.tif")
                    _save_mask (mask_patch, masks_dir / f"{stem}.png")

                n_saved   += 1
                img_saved += 1

            print(f"    {img_path.name}: {img_saved}/{len(windows)} "
                  f"windows kept (≥{min_nuclei} nuclei)")

    return n_images, n_checked, n_saved


# ---------------------------------------------------------------------------
# Core processing — avg_16 / avg_1 directories
# ---------------------------------------------------------------------------

def _process_avg_dirs(
    avg16_dir: Path,
    avg1_dir: Path,
    clean_dir: Path,
    masks_dir: Path,
    noisy_dir: Path,
    window_size: int,
    stride: int,
    min_nuclei: int,
    mask_suffix: str,
    dry_run: bool,
    verbose: bool,
) -> Tuple[int, int, int]:
    """
    Process avg_16 / avg_1 calibration directories → clean/, masks/, noisy/.

    For each qualifying window:
      * avg_16 patch  → clean/
      * mask patch    → masks/
      * avg_1 patch   → noisy/

    Returns
    -------
    (n_images_processed, n_windows_evaluated, n_triplets_saved)
    """
    n_images = n_checked = n_saved = 0

    tpaf_files = sorted(f for f in avg16_dir.iterdir() if _is_tpaf_image(f))
    if not tpaf_files:
        print(f"  [WARN] No TPAF images found in {avg16_dir} — skipped")
        return 0, 0, 0

    print(f"\n  avg_16  ({len(tpaf_files)} TPAF image(s))")

    for img16_path in tpaf_files:
        mask_path = _find_mask(img16_path, mask_suffix)
        if mask_path is None:
            print(f"    [WARN] No mask for {img16_path.name} — skipped")
            continue

        # Locate the matching avg_1 file (same stem, may differ in extension).
        img1_path: Optional[Path] = None
        for ext in ('.tif', '.tiff'):
            candidate = avg1_dir / (img16_path.stem + ext)
            if candidate.exists():
                img1_path = candidate
                break
        if img1_path is None:
            print(f"    [WARN] No avg_1 counterpart for {img16_path.name} — skipped")
            continue

        img16 = _to_2d(_load_image(img16_path))
        img1  = _to_2d(_load_image(img1_path))
        mask  = _to_2d(_load_mask(mask_path))
        H, W  = img16.shape

        if H < window_size or W < window_size:
            print(f"    [WARN] {img16_path.name} ({H}×{W}) is smaller than "
                  f"the {window_size}×{window_size} window — skipped")
            continue

        if img1.shape != img16.shape:
            print(f"    [WARN] Shape mismatch: avg_16 {img16.shape} vs "
                  f"avg_1 {img1.shape} for {img16_path.name} — skipped")
            continue

        n_images += 1
        windows   = list(_sliding_windows(H, W, window_size, stride))
        img_saved = 0

        for row, col in windows:
            n_checked += 1
            patch16    = img16[row:row + window_size, col:col + window_size]
            patch1     = img1 [row:row + window_size, col:col + window_size]
            mask_patch = mask [row:row + window_size, col:col + window_size]
            n_nuc = _count_nuclei(mask_patch)

            if verbose:
                kept = "  kept" if n_nuc >= min_nuclei else ""
                print(f"      x={col:04d} y={row:04d}  nuclei={n_nuc}{kept}")

            if n_nuc < min_nuclei:
                continue

            stem = f"avg16_{img16_path.stem}_x{col:04d}_y{row:04d}"
            if not dry_run:
                _save_image(patch16,    clean_dir / f"{stem}.tif")
                _save_image(patch1,     noisy_dir / f"{stem}.tif")
                _save_mask (mask_patch, masks_dir / f"{stem}.png")

            n_saved   += 1
            img_saved += 1

        print(f"    {img16_path.name}: {img_saved}/{len(windows)} "
              f"windows kept (≥{min_nuclei} nuclei)")

    return n_images, n_checked, n_saved


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='step1b_select_patches',
        description=(
            'Slide a 512×512 window over TPAF images and save only patches '
            'whose Cellpose mask contains ≥ min_nuclei distinct nuclei '
            '(Step 1b of data prep).'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        '--wsi_dirs', nargs='+', required=True, metavar='DIR',
        help='One or more WSI directories, each containing clean TPAF .tif '
             'images and co-located Cellpose masks (*_cp_masks.png).  '
             'Example: /data/260227_WSI_00 /data/260227_WSI_01',
    )
    p.add_argument(
        '--avg16_dir', required=True, metavar='DIR',
        help='avg_16 calibration directory.  Contains clean TPAF images '
             'and co-located Cellpose masks.  Used for window qualification '
             'and as the clean reference for avg_1 noisy patches.',
    )
    p.add_argument(
        '--avg1_dir', required=True, metavar='DIR',
        help='avg_1 calibration directory.  Contains single-frame (noisiest) '
             'TPAF images with the same stems as avg_16.  Qualifying windows '
             'are extracted and saved to noisy/.',
    )
    p.add_argument(
        '--output_dir', default='outputs/patches', metavar='DIR',
        help='Root output directory.  Subdirectories clean/, masks/, and '
             'noisy/ are created automatically.',
    )
    p.add_argument(
        '--mask_suffix', default='_cp_masks',
        help='Suffix that Cellpose appends to the image stem when saving '
             'instance-mask files (e.g. fov001_cp_masks.png).',
    )
    p.add_argument(
        '--window_size', type=int, default=512, metavar='PX',
        help='Side length of the square sliding window in pixels.',
    )
    p.add_argument(
        '--stride', type=int, default=256, metavar='PX',
        help='Sliding window step in pixels.  256 (50 %% overlap with a '
             '512 px window) ensures every cell-dense region appears intact '
             'in at least one window.  Use 384 for less redundancy on very '
             'large images.',
    )
    p.add_argument(
        '--min_nuclei', type=int, default=50, metavar='N',
        help='Minimum number of distinct nuclei (non-zero Cellpose label '
             'values) required to keep a patch.',
    )
    p.add_argument(
        '--dry_run', action='store_true',
        help='Evaluate all windows and report counts without writing files.  '
             'Use this to estimate dataset size before committing disk space.',
    )
    p.add_argument(
        '--verbose', action='store_true',
        help='Print nucleus count for every individual window.',
    )

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    print(f"\n{BANNER}")
    print("STEP 1b — TPAF patch selection")
    print(f"  window_size : {args.window_size} px")
    print(f"  stride      : {args.stride} px  "
          f"({100 * (args.window_size - args.stride) // args.window_size} % overlap)")
    print(f"  min_nuclei  : {args.min_nuclei}")
    print(f"  dry_run     : {args.dry_run}")
    print(BANNER)

    if not _preflight(args):
        return 1

    # ---- output directories ------------------------------------------------
    out_root  = Path(args.output_dir)
    clean_dir = out_root / 'clean'
    masks_dir = out_root / 'masks'
    noisy_dir = out_root / 'noisy'

    if not args.dry_run:
        for d in (clean_dir, masks_dir, noisy_dir):
            d.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput root : {out_root.resolve()}")
        print(f"  clean/    : {clean_dir}")
        print(f"  masks/    : {masks_dir}")
        print(f"  noisy/    : {noisy_dir}")
    else:
        print("\n[DRY RUN] No files will be written.")

    # ---- WSI directories ---------------------------------------------------
    print(f"\n{BANNER}")
    print("PROCESSING WSI DIRECTORIES")
    print(BANNER)

    wsi_dirs = [Path(d) for d in args.wsi_dirs]
    wsi_images, wsi_checked, wsi_saved = _process_wsi_dirs(
        wsi_dirs    = wsi_dirs,
        clean_dir   = clean_dir,
        masks_dir   = masks_dir,
        window_size = args.window_size,
        stride      = args.stride,
        min_nuclei  = args.min_nuclei,
        mask_suffix = args.mask_suffix,
        dry_run     = args.dry_run,
        verbose     = args.verbose,
    )

    # ---- avg_16 / avg_1 directories ----------------------------------------
    print(f"\n{BANNER}")
    print("PROCESSING avg_16 / avg_1 DIRECTORIES")
    print(BANNER)

    avg_images, avg_checked, avg_saved = _process_avg_dirs(
        avg16_dir   = Path(args.avg16_dir),
        avg1_dir    = Path(args.avg1_dir),
        clean_dir   = clean_dir,
        masks_dir   = masks_dir,
        noisy_dir   = noisy_dir,
        window_size = args.window_size,
        stride      = args.stride,
        min_nuclei  = args.min_nuclei,
        mask_suffix = args.mask_suffix,
        dry_run     = args.dry_run,
        verbose     = args.verbose,
    )

    # ---- summary -----------------------------------------------------------
    total_clean = wsi_saved + avg_saved
    total_masks = total_clean       # one mask per clean patch
    total_noisy = avg_saved         # only avg_1 contributes to noisy/

    dry_note = "  (dry run — not written)" if args.dry_run else ""

    print(f"\n{BANNER}")
    print("SUMMARY")
    print(f"  Sources processed         : "
          f"{wsi_images} WSI image(s)  +  {avg_images} avg_16 image(s)")
    print(f"  Windows evaluated (total) : {wsi_checked + avg_checked}")
    print(f"  Patches → clean/          : {total_clean}{dry_note}")
    print(f"  Patches → masks/          : {total_masks}{dry_note}")
    print(f"  Patches → noisy/          : {total_noisy}{dry_note}")
    print(BANNER)

    if not args.dry_run and total_clean > 0:
        print("\n*** MANUAL CHECKS REQUIRED ***\n")
        print("1. Verify file counts are as expected:")
        print(f"       clean/ : {total_clean} files")
        print(f"       masks/ : {total_masks} files  (should equal clean count)")
        print(f"       noisy/ : {total_noisy} files  (should equal avg_16 contribution)")
        print()
        print("2. Open 3–5 patch triplets in Fiji or Python and confirm:")
        print("     - Images are 512×512 uint8 grayscale TIFF.")
        print("     - Masks are 512×512 uint16 PNG (Cellpose instance labels).")
        print("     - avg_1 (noisy/) and avg_16 (clean/) patches are spatially aligned.")
        print()
        print("3. Run Step 2 to generate synthetic noisy images from clean/ patches:")
        print(f"       python data_prep/step2_generate_noisy.py \\")
        print(f"           --clean_dir   {out_root}/clean \\")
        print(f"           --params_file outputs/noise_estimation/noise_params.json \\")
        print(f"           --output_dir  outputs/patches_noisy \\")
        print(f"           --n_frames 1")
        print()
        print("4. Run Step 3 (twice, once per noisy source) to build train/val splits:")
        print(f"       # synthetic noisy (covers WSI + avg_16 clean patches)")
        print(f"       python data_prep/step3_build_splits.py \\")
        print(f"           --noisy_dir  outputs/patches_noisy/n_frames_1 \\")
        print(f"           --clean_dir  {out_root}/clean \\")
        print(f"           --mask_dir   {out_root}/masks \\")
        print(f"           --output_dir outputs/stage2_data --seed 42")
        print()
        print(f"       # real avg_1 noisy (covers avg_16 subset only)")
        print(f"       python data_prep/step3_build_splits.py \\")
        print(f"           --noisy_dir  {out_root}/noisy \\")
        print(f"           --clean_dir  {out_root}/clean \\")
        print(f"           --mask_dir   {out_root}/masks \\")
        print(f"           --output_dir outputs/stage2_data --seed 42")

    return 0


if __name__ == '__main__':
    sys.exit(main())
