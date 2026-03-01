#!/usr/bin/env python3
"""
Step 3: Assemble (noisy, clean, mask) triplets into the train/val split
expected by ``training/train_stage2_colab.py``.

Expected inputs
---------------
The noisy images (from Step 2) are always in their own directory.
Clean images and masks can be provided in two ways:

**Option A — separate directories** (``--clean_dir`` + ``--mask_dir``)::

    noisy_dir/       # synthetic noisy images from Step 2
        img_001.tif
        …
    clean_dir/
        img_001.tif
        …
    mask_dir/
        img_001.tif
        …

**Option B — co-located** (``--clean_mask_dir`` + ``--mask_suffix``)  ← actual data::

    clean_mask_dir/
        20240829 ... _0000.tif           # clean image
        20240829 ... _0000_cp_masks.png  # mask (stem = image stem + "_cp_masks")
        20240829 ... _0001.tif
        20240829 ... _0001_cp_masks.png
        …

    Use ``--mask_suffix _cp_masks`` (the default).  The script strips the suffix
    from the mask filename to recover the matching clean image stem.

Output layout
-------------
::

    output_dir/
        train/
            noisy/    # ~(1 - val_fraction) × N images
            clean/
            masks/
        val/
            noisy/    # ~val_fraction × N images
            clean/
            masks/

This layout is directly consumable by ``training/train_stage2_colab.py``.

Using multiple noise levels
---------------------------
To include images from more than one noise level (e.g., n_frames_1 and
n_frames_4 for augmentation), run this script twice with different
``--noisy_dir`` values pointing to the same ``--output_dir``.  The second
run will add files into the existing train/ and val/ directories **using the
same train/val split** if you pass the same ``--seed`` and ``--val_fraction``.
This guarantees that the same clean image does not appear in both train and
val across noise levels.

Usage
-----
::

    # Option B — co-located images and masks (actual data):
    python data_prep/step3_build_splits.py \\
        --noisy_dir      outputs/noisy_images/n_frames_1 \\
        --clean_mask_dir /path/to/260227_WSI_00 \\
        --mask_suffix    _cp_masks \\
        --output_dir     outputs/stage2_data

    # Option A — separate directories:
    python data_prep/step3_build_splits.py \\
        --noisy_dir  outputs/noisy_images/n_frames_1 \\
        --clean_dir  /path/to/clean_images \\
        --mask_dir   /path/to/masks \\
        --output_dir outputs/stage2_data

    # Add n_frames_4 augmentation to the same split (either option):
    python data_prep/step3_build_splits.py \\
        --noisy_dir      outputs/noisy_images/n_frames_4 \\
        --clean_mask_dir /path/to/260227_WSI_00 \\
        --mask_suffix    _cp_masks \\
        --output_dir     outputs/stage2_data \\
        --noisy_suffix   _n4

After running, MANUALLY check
------------------------------
1. **Filename matching** — The script prints any files that exist in one
   directory but not the others.  Mismatches must be resolved (rename or
   remove orphan files) before the dataset can be used.

2. **Train/val counts** — Confirm the split sizes look reasonable for your
   dataset.  For < 50 images total, consider ``--val_fraction 0.20`` to keep
   at least 10 validation samples.

3. **Triplet alignment** — Pick 3–5 random stems.  Open the corresponding
   noisy, clean, and mask files side-by-side in Fiji or napari.  Verify that
   all three images show the **same field of view** (no systematic offset,
   no accidental filename collision between different FOVs).

4. **Mask quality** — Masks should be binary (pixel values 0 or 1 / 0 or 255).
   Nucleus boundaries should be drawn tightly.  Holes inside nuclei or ragged
   boundaries will impair segmentation learning.

5. **Symlinks** — By default the output contains symlinks.  If you intend to
   move the source directories, re-run with ``--copy`` to make physical copies.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

_SUPPORTED_EXT = {'.npy', '.png', '.tif', '.tiff'}

BANNER = "=" * 64


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='step3_build_splits',
        description='Organise noisy/clean/mask triplets into train/val splits (Step 3).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--noisy_dir', required=True,
        help='Directory with noisy images for one n_frames level.',
    )

    # ---- clean + mask source: two mutually exclusive modes ----
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        '--clean_mask_dir',
        metavar='DIR',
        help='Directory containing clean images AND masks interleaved '
             '(use together with --mask_suffix).  '
             'Example: /path/to/260227_WSI_00',
    )
    src.add_argument(
        '--clean_dir',
        metavar='DIR',
        help='Directory with clean reference images only '
             '(use together with --mask_dir).',
    )

    p.add_argument(
        '--mask_dir',
        metavar='DIR',
        default=None,
        help='Directory with mask files (required when --clean_dir is used).',
    )
    p.add_argument(
        '--mask_suffix', default='_cp_masks',
        help='Suffix that identifies mask files inside --clean_mask_dir.  '
             'Example: "_cp_masks" means img_001_cp_masks.png is the mask '
             'for img_001.tif.',
    )

    p.add_argument(
        '--output_dir', default='outputs/stage2_data',
        help='Root output directory; train/ and val/ subdirectories are created here.',
    )
    p.add_argument(
        '--val_fraction', type=float, default=0.15,
        help='Fraction of complete triplets held out as validation set.',
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducible train/val split.',
    )
    p.add_argument(
        '--noisy_suffix', type=str, default='',
        help='Optional suffix appended to noisy filenames in the output '
             '(e.g. "_n4" → img_001_n4.tif).  Use when combining multiple '
             'noise levels to avoid filename collisions.',
    )
    p.add_argument(
        '--copy', action='store_true', default=False,
        help='Copy files instead of creating symlinks.  Required if you plan '
             'to move the source directories after running this script.',
    )
    p.add_argument(
        '--dry_run', action='store_true', default=False,
        help='Print what would be done without creating any files.',
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_stems(directory: Path) -> dict:
    """Return {stem: path} for all supported image files in directory."""
    return {
        f.stem: f
        for f in sorted(directory.iterdir())
        if f.suffix.lower() in _SUPPORTED_EXT
    }


def _transfer(src: Path, dst: Path, do_copy: bool, dry_run: bool) -> None:
    """Copy or symlink src → dst, creating parent directories as needed."""
    if dry_run:
        action = "COPY" if do_copy else "LINK"
        print(f"    [{action}] {src.name}  →  {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return  # already written (e.g. second noise level run)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _split_clean_mask_dir(directory: Path, mask_suffix: str):
    """
    Separate clean images from mask files in a co-located directory.

    A file is a mask if its stem ends with ``mask_suffix``.
    Its clean-image stem is ``file.stem[:-len(mask_suffix)]``.

    Returns:
        clean_map : {clean_stem: Path}  — clean image files
        mask_map  : {clean_stem: Path}  — mask files, keyed by matching clean stem
    """
    clean_map: dict = {}
    mask_map: dict  = {}

    for f in sorted(directory.iterdir()):
        if f.suffix.lower() not in _SUPPORTED_EXT:
            continue
        if f.stem.endswith(mask_suffix):
            clean_stem = f.stem[: -len(mask_suffix)]
            mask_map[clean_stem] = f
        else:
            clean_map[f.stem] = f

    return clean_map, mask_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    noisy_dir  = Path(args.noisy_dir)
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Validate input directories
    # ------------------------------------------------------------------
    if not noisy_dir.exists():
        print(f"[ERROR] --noisy_dir does not exist: {noisy_dir}")
        return 1

    # ---- Resolve clean_map and mask_map depending on mode ----
    if args.clean_mask_dir:
        # Option B: co-located images and masks
        clean_mask_dir = Path(args.clean_mask_dir)
        if not clean_mask_dir.exists():
            print(f"[ERROR] --clean_mask_dir does not exist: {clean_mask_dir}")
            return 1
        print(f"\n{BANNER}")
        print(f"SPLITTING co-located directory by mask suffix '{args.mask_suffix}'")
        print(f"  Directory : {clean_mask_dir}")
        clean_map, mask_map = _split_clean_mask_dir(clean_mask_dir, args.mask_suffix)
        print(f"  Clean images found : {len(clean_map)}")
        print(f"  Mask files found   : {len(mask_map)}")
        if not clean_map:
            print(f"[ERROR] No clean image files found (files whose stem does NOT end "
                  f"with '{args.mask_suffix}').")
            return 1
        if not mask_map:
            print(f"[ERROR] No mask files found (files whose stem ends with "
                  f"'{args.mask_suffix}').")
            print(f"        Check --mask_suffix or inspect the directory contents.")
            return 1
        # Warn about unmatched masks
        unmatched_masks = set(mask_map) - set(clean_map)
        if unmatched_masks:
            print(f"  [WARN] {len(unmatched_masks)} mask(s) have no matching clean image:")
            print(f"         {sorted(unmatched_masks)[:3]}"
                  + ("  …" if len(unmatched_masks) > 3 else ""))
        print(BANNER)
    else:
        # Option A: separate directories
        if args.mask_dir is None:
            print("[ERROR] --mask_dir is required when using --clean_dir.")
            return 1
        clean_dir = Path(args.clean_dir)
        mask_dir  = Path(args.mask_dir)
        for d in (clean_dir, mask_dir):
            if not d.exists():
                print(f"[ERROR] Directory does not exist: {d}")
                return 1
        clean_map = _get_stems(clean_dir)
        mask_map  = _get_stems(mask_dir)

    # ------------------------------------------------------------------
    # Collect stems from each directory
    # ------------------------------------------------------------------
    noisy_map = _get_stems(noisy_dir)

    all_stems = set(noisy_map) | set(clean_map) | set(mask_map)

    clean_src = args.clean_mask_dir if args.clean_mask_dir else args.clean_dir
    mask_src  = args.clean_mask_dir if args.clean_mask_dir else args.mask_dir

    print(f"\n{BANNER}")
    print("STEM INVENTORY")
    print(f"  noisy/ : {len(noisy_map)} files  ({noisy_dir})")
    print(f"  clean/ : {len(clean_map)} files  ({clean_src})")
    print(f"  masks/ : {len(mask_map)} files  ({mask_src})")
    print(BANNER)

    # ------------------------------------------------------------------
    # Categorise stems
    # ------------------------------------------------------------------
    complete  = set(noisy_map) & set(clean_map) & set(mask_map)
    no_noisy  = (set(clean_map) & set(mask_map)) - set(noisy_map)
    no_clean  = (set(noisy_map) & set(mask_map)) - set(clean_map)
    no_mask   = (set(noisy_map) & set(clean_map)) - set(mask_map)
    only_one  = all_stems - complete - no_noisy - no_clean - no_mask

    print(f"\n{BANNER}")
    print("TRIPLET MATCHING REPORT")
    print(f"  Complete triplets (noisy + clean + mask) : {len(complete)}")

    has_warnings = False

    if no_mask:
        has_warnings = True
        print(f"\n  [WARN] {len(no_mask)} stem(s) have noisy + clean but NO mask")
        print("         These will be EXCLUDED from the dataset.")
        print("         Examples: " + ", ".join(sorted(no_mask)[:5])
              + ("  …" if len(no_mask) > 5 else ""))

    if no_clean:
        has_warnings = True
        print(f"\n  [WARN] {len(no_clean)} stem(s) have noisy + mask but NO clean")
        print("         These will be EXCLUDED from the dataset.")
        print("         Examples: " + ", ".join(sorted(no_clean)[:5])
              + ("  …" if len(no_clean) > 5 else ""))

    if no_noisy:
        has_warnings = True
        print(f"\n  [WARN] {len(no_noisy)} stem(s) have clean + mask but NO noisy image")
        print("         These will be EXCLUDED from the dataset.")
        print("         Did you run Step 2 with all required clean images?")
        print("         Examples: " + ", ".join(sorted(no_noisy)[:5])
              + ("  …" if len(no_noisy) > 5 else ""))

    if only_one:
        print(f"\n  [INFO] {len(only_one)} stem(s) appear in only one directory — orphan files.")

    print(BANNER)

    if not complete:
        print("\n[ERROR] No complete triplets found.")
        print("        Common causes:")
        print("          - Filenames differ between noisy/, clean/, masks/")
        print("            (e.g. 'sample_001' vs 'sample001').")
        print("          - Step 2 was run with a different set of clean images.")
        return 1

    # Prompt user if there are warnings in a non-dry-run interactive session
    if has_warnings and not args.dry_run and sys.stdin.isatty():
        print(f"\n*** MANUAL CHECK REQUIRED ***")
        print(f"Only {len(complete)} of {len(all_stems)} stems form complete triplets.")
        print("Resolve the missing-file warnings above, or continue with complete triplets only.")
        ans = input("\nContinue with complete triplets only? [y/N] ").strip().lower()
        if ans != 'y':
            print("Aborted.  Fix filename mismatches and re-run.")
            return 1
    elif has_warnings:
        print(f"\n[INFO] Continuing with {len(complete)} complete triplets "
              f"(non-interactive mode — skipping confirmation prompt).")

    if len(complete) < 10:
        print(f"\n[WARN] Only {len(complete)} triplets — dataset is very small.")
        print("       Consider acquiring more labeled data before training.")

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    stems_sorted = sorted(complete)
    rng = random.Random(args.seed)
    rng.shuffle(stems_sorted)

    n_val   = max(1, round(len(stems_sorted) * args.val_fraction))
    n_train = len(stems_sorted) - n_val

    val_stems   = set(stems_sorted[:n_val])
    train_stems = set(stems_sorted[n_val:])

    print(f"\n{BANNER}")
    print("TRAIN / VAL SPLIT")
    print(f"  Total    : {len(stems_sorted)}")
    print(f"  Train    : {n_train}  ({n_train/len(stems_sorted):.1%})")
    print(f"  Val      : {n_val}   ({n_val/len(stems_sorted):.1%})")
    print(f"  Seed     : {args.seed}")
    print(BANNER)

    if args.dry_run:
        print("\n[DRY RUN] No files will be created.")

    # ------------------------------------------------------------------
    # Write files
    # ------------------------------------------------------------------
    suffix = args.noisy_suffix
    action_word = "Copying" if args.copy else "Symlinking"
    print(f"\n{action_word} files to {output_dir} …")

    subsets = {'train': train_stems, 'val': val_stems}

    for split, stems in subsets.items():
        for stem in stems:
            # noisy
            src_noisy = noisy_map[stem]
            dst_name  = f"{stem}{suffix}{src_noisy.suffix}"
            dst_noisy = output_dir / split / 'noisy' / dst_name
            _transfer(src_noisy, dst_noisy, args.copy, args.dry_run)

            # clean
            src_clean = clean_map[stem]
            dst_clean = output_dir / split / 'clean' / src_clean.name
            _transfer(src_clean, dst_clean, args.copy, args.dry_run)

            # mask
            src_mask = mask_map[stem]
            dst_mask = output_dir / split / 'masks' / src_mask.name
            _transfer(src_mask, dst_mask, args.copy, args.dry_run)

    # ------------------------------------------------------------------
    # Output verification
    # ------------------------------------------------------------------
    if not args.dry_run:
        print(f"\n{BANNER}")
        print("OUTPUT VERIFICATION")
        all_ok = True
        for split, stems in subsets.items():
            for role in ('noisy', 'clean', 'masks'):
                d = output_dir / split / role
                if d.exists():
                    found    = len([f for f in d.iterdir()
                                    if f.suffix.lower() in _SUPPORTED_EXT])
                    expected = len(stems)
                    status   = "[OK]  " if found == expected else "[WARN]"
                    if found != expected:
                        all_ok = False
                else:
                    found, expected, status = 0, len(stems), "[WARN]"
                    all_ok = False
                print(f"  {status} {split}/{role}: {found}/{expected}")
        print(BANNER)
    else:
        all_ok = True  # dry run always exits cleanly

    # ------------------------------------------------------------------
    # Manual check reminder
    # ------------------------------------------------------------------
    mode = "copies" if args.copy else "symlinks"
    print(f"\n*** MANUAL CHECKS REQUIRED ***")
    print()
    print("1. Confirm train/val counts are sensible:")
    print(f"   Train={n_train}, Val={n_val} "
          f"(adjust --val_fraction if needed; current={args.val_fraction}).")
    print()
    print("2. Spot-check triplet alignment:")
    sample_stems = sorted(train_stems)[:3]
    print("   Pick a few stems and open noisy/, clean/, masks/ side-by-side:")
    for s in sample_stems:
        print(f"     stem: {s}")
    print("   All three images must show the same field of view.")
    print("   A systematic mismatch (e.g. every mask is off by one image)")
    print("   means the filenames were assigned incorrectly.")
    print()
    print("3. Mask sanity check:")
    print("   - Pixel values should be 0 (background) or 1/255 (nucleus).")
    print("   - Run a quick histogram check in Fiji: Analyze → Histogram.")
    print("     Only two bars should appear for a binary mask.")
    print()
    print(f"Note: output files are {mode}.")
    if not args.copy:
        print("      If you move source directories, re-run with --copy.")
    print()
    print(f"Next step → training/train_stage2_colab.py --data_dir {output_dir.resolve()}")
    print(BANNER)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
