#!/usr/bin/env python3
"""
Step 1: Estimate Poisson-Gaussian noise parameters from calibration images.

Wraps noise_scripts/estimate_noise.py with automatic quality checks so that
problematic fits are flagged before downstream data generation.

Two calibration directory layouts are supported via ``--calibration_layout``:

per_fov (default)
    Each subdirectory is one field of view, containing all four averaging levels::

        calibration_dir/
            fov_001/
                avg1.tif    # 1-frame average  (highest noise)
                avg4.tif    # 4-frame average
                avg8.tif    # 8-frame average
                avg16.tif   # 16-frame average (clean reference)
            fov_002/
                ...

per_level  ← use this for the actual data
    Each subdirectory is one averaging level, containing all FOVs as files::

        calibration_dir/
            avg_1/
                fov_001.tif
                fov_002.tif
                ...
            avg_4/
                fov_001.tif
                fov_002.tif
                ...
            avg_8/   ...
            avg_16/  ...

    The script automatically reorganises this into the per_fov layout (using
    symlinks in a temporary directory) before running estimation.

Usage
-----
::

    python data_prep/step1_estimate_noise.py \\
        --calibration_dir /path/to/calibration \\
        --calibration_layout per_level \\
        --output_dir      outputs/noise_estimation

Outputs
-------
``output_dir/noise_params.json``
    Estimated parameters (a, b) consumed by Step 2.
``output_dir/validation_report.json``
    Per-N and per-group consistency statistics.
``output_dir/plots/``
    Diagnostic plots: mean_variance_curve.png, per_group_consistency.png,
    residual_distribution.png.

After running, MANUALLY check
------------------------------
1. ``R² ≥ 0.95`` is printed at the end of the quality report.
2. Poisson gain ``a`` is physically plausible for your detector
   (typical TPAF range: 0.01 – 100).
3. Readout noise ``b`` is physically plausible (typical range: 0 – 500 ADU).
4. ``plots/mean_variance_curve.png`` — data points follow a clean straight line.
5. ``plots/per_group_consistency.png`` — per-group a/b values cluster tightly
   (large spread indicates inconsistent calibration FOVs).
6. ``plots/residual_distribution.png`` — residuals roughly Gaussian; heavy
   tails indicate outlier bins or image artefacts.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# per_level layout: map directory name patterns → N value
# Accepts avg_1 / avg1 / avg_01 / avg01 etc.
# ---------------------------------------------------------------------------
_LEVEL_NAMES: dict = {
    1:  {'avg_1',  'avg1',  'avg_01',  'avg01'},
    4:  {'avg_4',  'avg4',  'avg_04',  'avg04'},
    8:  {'avg_8',  'avg8',  'avg_08',  'avg08'},
    16: {'avg_16', 'avg16', 'avg_016', 'avg016'},
}

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
R2_WARN = 0.90    # warn if R² is below this
R2_FAIL = 0.80    # hard failure if R² is below this

# Typical TPAF detector ranges — adjust if your data uses different units
A_PLAUSIBLE = (1e-3, 1e3)   # Poisson gain (ADU / photon)
B_PLAUSIBLE = (0.0, 500.0)  # Readout noise std (ADU)

BANNER = "=" * 64


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='step1_estimate_noise',
        description='Estimate Poisson-Gaussian noise model parameters (Step 1 of data prep).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--calibration_dir', required=True,
        help='Root directory containing calibration images.',
    )
    p.add_argument(
        '--calibration_layout', choices=['per_fov', 'per_level'], default='per_level',
        help=(
            'Directory layout. '
            '"per_level": subdirs are avg_1/, avg_4/, avg_8/, avg_16/ (actual data). '
            '"per_fov": subdirs are one per FOV, each containing avg1/4/8/16.tif.'
        ),
    )
    p.add_argument(
        '--output_dir', default='outputs/noise_estimation',
        help='Directory for noise_params.json, validation_report.json, and plots/.',
    )
    p.add_argument('--n_bins', type=int, default=200,
                   help='Number of intensity bins for regression.')
    p.add_argument('--min_samples_per_bin', type=int, default=30,
                   help='Discard bins with fewer than this many pixels.')
    p.add_argument('--background_percentile', type=float, default=5.0,
                   help='Background exclusion percentile (pixels below this are '
                        'excluded from the mean-variance regression).')
    p.add_argument('--no_robust', action='store_true', default=False,
                   help='Disable IRLS robust fitting (not recommended).')
    p.add_argument('--verbose', action='store_true', default=False,
                   help='Pass --verbose to the underlying estimation script.')
    return p


# ---------------------------------------------------------------------------
# Calibration directory structure validation and layout reorganisation
# ---------------------------------------------------------------------------

def _validate_per_fov(root: Path) -> bool:
    """Validate per_fov layout: each subdir must contain avg1/4/8/16.tif."""
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"[ERROR] calibration_dir has no subdirectories: {root}")
        return False

    required = ['avg1.tif', 'avg4.tif', 'avg8.tif', 'avg16.tif']
    n_complete = 0
    for d in sorted(subdirs):
        missing = [f for f in required if not (d / f).exists()]
        if missing:
            print(f"  [WARN] Group '{d.name}' missing: {missing} — will be skipped.")
        else:
            n_complete += 1

    print(f"\n{BANNER}")
    print("CALIBRATION DIRECTORY CHECK  (layout: per_fov)")
    print(f"  Subdirectories (FOVs)   : {len(subdirs)}")
    print(f"  Complete groups         : {n_complete}  (all 4 avg files present)")
    print(f"  Incomplete / skipped    : {len(subdirs) - n_complete}")
    print(BANNER)

    if n_complete == 0:
        print("[ERROR] No complete calibration groups found.")
        return False
    if n_complete < 3:
        print("[WARN]  Fewer than 3 complete groups — estimate may be unreliable.")
        print("        Recommended: ≥ 5 groups.")
    return True


def _validate_per_level(root: Path) -> bool:
    """Validate per_level layout: avg_1/, avg_4/, avg_8/, avg_16/ must exist."""
    found_n: dict = {}
    for n, names in _LEVEL_NAMES.items():
        for d in root.iterdir():
            if d.is_dir() and d.name.lower() in names:
                found_n[n] = d
                break

    missing = sorted({1, 4, 8, 16} - set(found_n.keys()))
    if missing:
        print(f"[ERROR] Cannot find level directories for N={missing}.")
        print(f"        Expected subdirectory names (case-insensitive):")
        for n in missing:
            print(f"          N={n}: one of {_LEVEL_NAMES[n]}")
        print(f"        Found subdirectories: {[d.name for d in root.iterdir() if d.is_dir()]}")
        return False

    # Count FOVs per level and check they match
    fov_counts = {}
    for n, d in found_n.items():
        exts = {'.tif', '.tiff', '.npy'}
        stems = {f.stem for f in d.iterdir() if f.suffix.lower() in exts}
        fov_counts[n] = stems

    complete_fovs = set.intersection(*fov_counts.values())
    total_fovs = set.union(*fov_counts.values())

    print(f"\n{BANNER}")
    print("CALIBRATION DIRECTORY CHECK  (layout: per_level)")
    for n in sorted(found_n):
        d = found_n[n]
        cnt = len(fov_counts[n])
        print(f"  N={n:2d}  ({d.name}/) : {cnt} FOV files")
    print(f"  Complete FOVs (present in all 4 levels) : {len(complete_fovs)}")
    if len(complete_fovs) < len(total_fovs):
        missing_fovs = total_fovs - complete_fovs
        print(f"  Partial FOVs (missing from ≥1 level)    : {len(missing_fovs)}"
              f"  (will be skipped)")
        print(f"  Examples: {sorted(missing_fovs)[:3]}")
    print(BANNER)

    if len(complete_fovs) == 0:
        print("[ERROR] No FOV appears in all four level directories.")
        print("        Check that filenames match across avg_1/, avg_4/, avg_8/, avg_16/.")
        return False
    if len(complete_fovs) < 3:
        print("[WARN]  Only", len(complete_fovs), "complete FOVs — estimate may be unreliable.")
        print("        Recommended: ≥ 5 FOVs.")
    return True


def _reorganize_per_level_to_per_fov(calibration_dir: str, tmp_dir: Path) -> str:
    """
    Build a temporary per_fov directory from a per_level calibration layout.

    Creates symlinks::

        tmp_dir/
            fov_001/
                avg1.tif  →  calibration_dir/avg_1/fov_001.tif
                avg4.tif  →  calibration_dir/avg_4/fov_001.tif
                avg8.tif  →  calibration_dir/avg_8/fov_001.tif
                avg16.tif →  calibration_dir/avg_16/fov_001.tif
            fov_002/
                ...

    Returns the path to tmp_dir (the per_fov root).
    """
    root = Path(calibration_dir)
    exts = {'.tif', '.tiff', '.npy'}

    # Locate the four level directories
    level_dirs: dict = {}
    for n, names in _LEVEL_NAMES.items():
        for d in root.iterdir():
            if d.is_dir() and d.name.lower() in names:
                level_dirs[n] = d
                break

    # Collect FOV stems per level
    fov_stems_per_level = {
        n: {f.stem: f for f in d.iterdir() if f.suffix.lower() in exts}
        for n, d in level_dirs.items()
    }

    # Intersect across all levels
    complete_fovs = set.intersection(
        *[set(s.keys()) for s in fov_stems_per_level.values()]
    )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for stem in sorted(complete_fovs):
        fov_dir = tmp_dir / stem
        fov_dir.mkdir(exist_ok=True)
        for n, stem_map in fov_stems_per_level.items():
            src = stem_map[stem]
            # Destination name must match what noise_model.py expects: avg1.tif etc.
            dst = fov_dir / f"avg{n}.tif"
            if not dst.exists():
                dst.symlink_to(src.resolve())

    print(f"  Reorganised {len(complete_fovs)} FOVs into temporary per_fov layout:")
    print(f"  {tmp_dir}")
    return str(tmp_dir)


def _validate_calibration_dir(calibration_dir: str, layout: str) -> bool:
    """Dispatch to the appropriate layout validator."""
    root = Path(calibration_dir)
    if not root.exists():
        print(f"[ERROR] calibration_dir does not exist: {root}")
        return False
    if layout == 'per_level':
        return _validate_per_level(root)
    return _validate_per_fov(root)


# ---------------------------------------------------------------------------
# Run underlying estimation script
# ---------------------------------------------------------------------------

def _run_estimation(args, data_dir: str) -> int:
    script = _PROJECT_ROOT / 'noise_scripts' / 'estimate_noise.py'
    cmd = [
        sys.executable, str(script),
        '--data_dir', data_dir,
        '--output_dir', args.output_dir,
        '--n_bins', str(args.n_bins),
        '--min_samples_per_bin', str(args.min_samples_per_bin),
        '--exclude_background_percentile', str(args.background_percentile),
        '--save_plots',
        '--validate',
    ]
    if args.no_robust:
        cmd.append('--no_robust')
    if args.verbose:
        cmd.append('--verbose')

    print(f"\n{BANNER}")
    print("RUNNING: noise_scripts/estimate_noise.py")
    print(f"  data_dir   : {data_dir}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  n_bins          : {args.n_bins}")
    print(f"  min_samples     : {args.min_samples_per_bin}")
    print(f"  bg_percentile   : {args.background_percentile}")
    print(f"  robust IRLS     : {not args.no_robust}")
    print(BANNER)

    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Quality check on estimated parameters
# ---------------------------------------------------------------------------

def _quality_check(output_dir: str) -> bool:
    """Read noise_params.json, print a quality report, return True if OK."""
    params_path = Path(output_dir) / 'noise_params.json'
    if not params_path.exists():
        print(f"\n[ERROR] noise_params.json not found at {params_path}")
        print("        The estimation script may have failed — check output above.")
        return False

    with open(params_path) as fh:
        p = json.load(fh)

    a       = p['a']
    b       = p['b']
    r2      = p['r_squared']
    n_grp   = p['n_groups']
    n_pts   = p['n_points']
    n_pix   = p['n_pixels_used']

    print(f"\n{BANNER}")
    print("ESTIMATED PARAMETERS")
    print(f"  Poisson gain  a  =  {a:.4f}   (1/a = {p['slope']:.6f})")
    print(f"  Readout noise b  =  {b:.4f}   (b²  = {p['intercept']:.4f})")
    print(f"  R²               =  {r2:.4f}")
    print(f"  Groups used      =  {n_grp}")
    print(f"  Regression pts   =  {n_pts}")
    print(f"  Pixels used      =  {n_pix:,}")
    print(BANNER)

    ok = True

    # ---- R² check ----
    if r2 < R2_FAIL:
        print(f"  [FAIL] R² = {r2:.4f} < {R2_FAIL}  — fit is unreliable.")
        print("         Likely causes: misaligned images, too few calibration groups,")
        print("         or heavy clipping/saturation in the raw data.")
        ok = False
    elif r2 < R2_WARN:
        print(f"  [WARN] R² = {r2:.4f} < {R2_WARN}  — fit quality is marginal.")
        print("         Proceed with caution; inspect plots carefully.")
    else:
        print(f"  [OK]   R² = {r2:.4f}")

    # ---- Poisson gain check ----
    if not (A_PLAUSIBLE[0] <= a <= A_PLAUSIBLE[1]):
        print(f"  [WARN] a = {a:.4f} is outside the expected range {A_PLAUSIBLE}.")
        print("         Check image units (photon counts vs. ADU) and pixel intensity range.")
    else:
        print(f"  [OK]   a = {a:.4f} (plausible Poisson gain)")

    # ---- Readout noise check ----
    if not (B_PLAUSIBLE[0] <= b <= B_PLAUSIBLE[1]):
        print(f"  [WARN] b = {b:.4f} is outside the expected range {B_PLAUSIBLE}.")
        print("         Unusually large b may indicate the images are already pre-processed.")
    else:
        print(f"  [OK]   b = {b:.4f} (plausible readout noise)")

    # ---- Validation report summary ----
    val_path = Path(output_dir) / 'validation_report.json'
    if val_path.exists():
        with open(val_path) as fh:
            report = json.load(fh)
        print()
        print("  Per-N consistency (a should be stable across N=1, 4, 8):")
        for label, v in report.get('per_N_params', {}).items():
            flag = "[OK]  " if v['a_rel_error_pct'] < 10.0 else "[WARN]"
            print(f"    {flag} {label}:  a={v['a']:.4f} (±{v['a_rel_error_pct']:.1f} %)"
                  f"   b={v['b']:.4f}   R²={v['r_squared']:.4f}")
        nt = report.get('normality_test')
        if nt:
            flag = "[OK]  " if nt['normal_at_5pct'] else "[WARN]"
            print(f"\n  {flag} Shapiro-Wilk normality:  W={nt['statistic']:.4f}"
                  f"   p={nt['p_value']:.4f}"
                  f"  ({'PASS' if nt['normal_at_5pct'] else 'FAIL — residuals non-Gaussian'})")

    plots_dir = Path(output_dir) / 'plots'
    print(f"\n{BANNER}")
    print("*** MANUAL CHECKS REQUIRED ***")
    print()
    print("1. Inspect diagnostic plots:")
    print(f"   {plots_dir}/mean_variance_curve.png")
    print("       -> Data points should follow a tight straight line.")
    print("       -> Outlier clouds at high intensity may indicate sensor saturation.")
    print(f"   {plots_dir}/per_group_consistency.png")
    print("       -> Each FOV's a and b estimates should cluster within ~10 %.")
    print("       -> Outlier FOVs may have imaging artefacts — consider removing them.")
    print(f"   {plots_dir}/residual_distribution.png")
    print("       -> Residuals should be roughly bell-shaped.")
    print("       -> Heavy tails indicate the Poisson-Gaussian model is a poor fit.")
    print()
    print("2. If R² < 0.90 or a/b are implausible:")
    print("   - Add more calibration FOVs (currently only unique positions/depths).")
    print("   - Adjust --background_percentile (try values between 1 and 15).")
    print("   - Verify that avg16 images are actually less noisy than avg1 images.")
    print()
    print("3. Note the a and b values for your records — they characterise your")
    print("   specific microscope/detector combination and should be stable across")
    print("   sessions if imaging conditions do not change.")
    print(BANNER)

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Pre-flight check
    if not _validate_calibration_dir(args.calibration_dir, args.calibration_layout):
        return 1

    # For per_level layout, reorganise into a temporary per_fov directory
    tmp_dir_obj = None
    data_dir = args.calibration_dir
    if args.calibration_layout == 'per_level':
        print(f"\n{BANNER}")
        print("REORGANISING per_level → per_fov (symlinks in temp directory) …")
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix='afn_calib_')
        tmp_path = Path(tmp_dir_obj.name)
        data_dir = _reorganize_per_level_to_per_fov(args.calibration_dir, tmp_path)
        print(BANNER)

    try:
        # Run estimation
        rc = _run_estimation(args, data_dir)
        if rc != 0:
            print(f"\n[ERROR] estimate_noise.py exited with code {rc}.")
            print("        Check the output above for error messages.")
            return rc

        # Quality check
        ok = _quality_check(args.output_dir)
        return 0 if ok else 1
    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


if __name__ == '__main__':
    sys.exit(main())
