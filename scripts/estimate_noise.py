#!/usr/bin/env python3
"""
CLI entry point for Poisson-Gaussian noise parameter estimation.

Scans a directory of multi-frame TPAF image groups, estimates the noise
model parameters (a, b) via Multi-Level Mean-Variance linear regression,
saves the result as JSON, and optionally produces diagnostic plots.

Usage
-----
::

    python scripts/estimate_noise.py \\
        --data_dir  data/ \\
        --output_dir  outputs/ \\
        --n_bins  200 \\
        --min_samples_per_bin  30 \\
        --exclude_background_percentile  5 \\
        --save_plots

Arguments
---------
See ``python scripts/estimate_noise.py --help`` for the full list.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root so the script can be run from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.noise_model import PoissonGaussianEstimator
from src.visualization import (
    plot_mean_variance_curve,
    plot_per_group_consistency,
    plot_residual_distribution,
)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
        datefmt='%H:%M:%S',
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger('estimate_noise')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='estimate_noise',
        description='Estimate Poisson-Gaussian noise parameters from multi-frame TPAF images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- I/O ----
    p.add_argument(
        '--data_dir', required=True,
        help='Root directory containing image group subdirectories '
             '(each with avg1.tif, avg4.tif, avg8.tif, avg16.tif).',
    )
    p.add_argument(
        '--output_dir', default='outputs',
        help='Directory where noise_params.json and plots are saved.',
    )
    p.add_argument(
        '--params_filename', default='noise_params.json',
        help='Filename for the estimated parameters JSON.',
    )

    # ---- Estimation hyper-parameters ----
    p.add_argument(
        '--n_bins', type=int, default=200,
        help='Number of equal-width intensity bins for the regression.',
    )
    p.add_argument(
        '--min_samples_per_bin', type=int, default=30,
        help='Discard bins with fewer than this many pixels (statistical reliability).',
    )
    p.add_argument(
        '--exclude_background_percentile', type=float, default=5.0,
        dest='background_percentile',
        help='Exclude pixels whose avg16 intensity is below this global percentile '
             '(dark background does not follow the Poisson model).',
    )
    p.add_argument(
        '--no_robust', action='store_false', dest='robust', default=True,
        help='Disable IRLS robust fitting (use ordinary WLS instead).',
    )

    # ---- Outputs ----
    p.add_argument(
        '--save_plots', action='store_true', default=False,
        help='Save diagnostic plots (mean-variance, per-group, residuals) to output_dir.',
    )
    p.add_argument(
        '--validate', action='store_true', default=False,
        help='Run per-N and per-group validation after estimation.',
    )

    # ---- Misc ----
    p.add_argument('--verbose', action='store_true', default=False,
                   help='Enable DEBUG-level logging.')

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Run parameter estimation.

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
    logger.info("Poisson-Gaussian Noise Parameter Estimation")
    logger.info("=" * 60)
    logger.info("data_dir            : %s", args.data_dir)
    logger.info("output_dir          : %s", output_dir)
    logger.info("n_bins              : %d", args.n_bins)
    logger.info("min_samples_per_bin : %d", args.min_samples_per_bin)
    logger.info("background_pct      : %.1f %%", args.background_percentile)
    logger.info("robust IRLS         : %s", args.robust)

    # ------------------------------------------------------------------
    # Load image groups
    # ------------------------------------------------------------------
    estimator = PoissonGaussianEstimator()

    try:
        logger.info("Loading image groups …")
        groups = estimator.load_image_groups(args.data_dir)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Failed to load image groups: %s", exc)
        return 1

    logger.info("Loaded %d groups.", len(groups))

    # ------------------------------------------------------------------
    # Estimate parameters
    # ------------------------------------------------------------------
    logger.info("Estimating noise parameters …")
    try:
        params = estimator.estimate_params(
            groups,
            n_bins=args.n_bins,
            min_samples_per_bin=args.min_samples_per_bin,
            background_percentile=args.background_percentile,
            robust=args.robust,
        )
    except ValueError as exc:
        logger.error("Estimation failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("RESULTS")
    logger.info("  Poisson gain  a  = %.4f   (1/a = %.4f)", params.a, params.slope)
    logger.info("  Gaussian std  b  = %.4f   (b² = %.4f)", params.b, params.intercept)
    logger.info("  R²               = %.4f", params.r_squared)
    logger.info("  Groups used      = %d", params.n_groups)
    logger.info("  Data points      = %d", params.n_points)
    logger.info("  Pixels used      = %d", params.n_pixels_used)
    logger.info("-" * 60)

    # ------------------------------------------------------------------
    # Save parameters
    # ------------------------------------------------------------------
    params_path = output_dir / args.params_filename
    estimator.save_params(params, str(params_path))
    logger.info("Parameters saved to '%s'", params_path)

    # ------------------------------------------------------------------
    # Optional validation
    # ------------------------------------------------------------------
    report = None
    if args.validate:
        logger.info("Running validation …")
        report = estimator.validate_params(
            groups, params,
            n_bins=args.n_bins,
            min_samples_per_bin=args.min_samples_per_bin,
            background_percentile=args.background_percentile,
        )
        validation_path = output_dir / 'validation_report.json'
        # Convert numpy types for JSON serialisation
        def _jsonify(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(type(obj))

        with open(validation_path, 'w') as fh:
            json.dump(report, fh, indent=2, default=_jsonify)
        logger.info("Validation report saved to '%s'", validation_path)

        # Print per-N table
        logger.info("Per-N parameter estimates:")
        for label, v in report['per_N_params'].items():
            logger.info(
                "  %s :  a=%.4f (±%.1f %%)   b=%.4f (|Δb|=%.4f)   R²=%.4f",
                label, v['a'], v['a_rel_error_pct'], v['b'], v['b_abs_error'], v['r_squared'],
            )

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if args.save_plots:
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        logger.info("Generating diagnostic plots …")

        # ---- 1. Mean-variance curve ----
        if estimator.last_fit_data is not None:
            fig = plot_mean_variance_curve(
                estimator_results=estimator.last_fit_data,
                a=params.a,
                b=params.b,
                r_squared=params.r_squared,
                output_path=str(plots_dir / 'mean_variance_curve.png'),
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

        # ---- 2. Per-group consistency ----
        logger.info("Computing per-group parameter estimates for consistency plot …")
        per_group = estimator.estimate_per_group_params(
            groups,
            n_bins=args.n_bins,
            min_samples_per_bin=args.min_samples_per_bin,
            background_percentile=args.background_percentile,
        )
        if per_group:
            import matplotlib.pyplot as plt
            fig = plot_per_group_consistency(
                per_group_params=per_group,
                output_path=str(plots_dir / 'per_group_consistency.png'),
            )
            plt.close(fig)

        # ---- 3. Residual distribution ----
        if report is not None and report.get('residuals'):
            import matplotlib.pyplot as plt
            residuals = np.array(report['residuals'])
            fig = plot_residual_distribution(
                residuals=residuals,
                output_path=str(plots_dir / 'residual_distribution.png'),
            )
            plt.close(fig)
        elif args.validate:
            logger.warning(
                "No residuals available for residual plot "
                "(need --validate to collect them)."
            )
        else:
            logger.info(
                "Skipping residual plot (add --validate to enable)."
            )

        logger.info("All plots saved to '%s'", plots_dir)

    logger.info("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
