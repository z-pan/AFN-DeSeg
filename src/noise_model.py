"""
Poisson-Gaussian noise parameter estimation for two-photon fluorescence microscopy.

Noise model
-----------
Single-frame observation:
    y = (1/a) · Poisson(a · x) + N(0, b²)

N-frame average:
    y_N = (1/a) · Poisson(a · x · N) / N + N(0, b²/N)
    Var[y_N] = (x/a + b²) / N

Core estimation identity
------------------------
Using avg16 as the clean reference and computing residuals residual_N = avg_N − avg16:

    Var[residual_N] = Var[avg_N] + Var[avg16]       (frames assumed independent)
                    = (x/a + b²) / N + (x/a + b²) / 16
                    = (x/a + b²) · (1/N + 1/16)

Therefore:
    Var_corrected ≡ Var[residual_N] / (1/N + 1/16) = x/a + b²

A single weighted linear regression across all groups and all N values
simultaneously estimates 1/a (slope) and b² (intercept).
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class NoiseModelParams:
    """Estimated Poisson-Gaussian noise model parameters.

    Attributes:
        a: Poisson gain (detector efficiency, inverse of shot-noise slope).
        b: Gaussian readout noise standard deviation.
        r_squared: Coefficient of determination of the weighted linear fit.
        n_groups: Number of image groups used in estimation.
        n_points: Number of (bin, N) pairs entering the regression.
        n_pixels_used: Total pixel count summed across all bins.
        slope: Raw regression slope == 1/a.
        intercept: Raw regression intercept == b².
        fit_details: Dictionary of auxiliary fit diagnostics.
    """

    a: float
    b: float
    r_squared: float
    n_groups: int
    n_points: int
    n_pixels_used: int
    slope: float
    intercept: float
    fit_details: Dict


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class PoissonGaussianEstimator:
    """Estimates Poisson-Gaussian noise parameters from multi-frame TPAF images.

    Each image group must contain four files:
        avg1.tif   — 1-frame average  (highest noise)
        avg4.tif   — 4-frame average
        avg8.tif   — 8-frame average
        avg16.tif  — 16-frame average (used as clean reference)

    After calling :meth:`estimate_params`, the raw regression data are
    available via :attr:`last_fit_data` for downstream visualisation.

    Example::

        estimator = PoissonGaussianEstimator()
        groups = estimator.load_image_groups('data/')
        params = estimator.estimate_params(groups)
        estimator.save_params(params, 'outputs/noise_params.json')
    """

    #: Frame counts for each expected key.
    FRAME_COUNTS: Dict[str, int] = {
        'avg1': 1, 'avg4': 4, 'avg8': 8, 'avg16': 16
    }
    #: The key used as clean reference.
    REFERENCE_KEY: str = 'avg16'
    REFERENCE_N: int = 16
    #: Keys (N values) used as noisy observations.
    NOISY_KEYS: List[str] = ['avg1', 'avg4', 'avg8']

    def __init__(self) -> None:
        #: Raw data from the most recent :meth:`estimate_params` call.
        #: Keys: 'x', 'var', 'n_values', 'weights' (all np.ndarray).
        self.last_fit_data: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image_groups(self, data_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Scan *data_dir* and load every group's four averaging-level images.

        Args:
            data_dir: Root directory whose immediate subdirectories are
                      image groups, each containing avg1.tif … avg16.tif.

        Returns:
            Mapping ``{group_name: {key: array, ...}}`` where *array* is
            ``float64`` and *key* is one of ``'avg1'``, ``'avg4'``,
            ``'avg8'``, ``'avg16'``.

        Raises:
            ValueError: If no valid groups are found.
        """
        data_path = Path(data_dir)
        groups: Dict[str, Dict[str, np.ndarray]] = {}
        group_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())

        logger.info("Scanning %d subdirectories in '%s'", len(group_dirs), data_dir)

        for group_dir in group_dirs:
            name = group_dir.name
            images: Dict[str, np.ndarray] = {}
            skip = False

            for key in self.FRAME_COUNTS:
                tif_path = group_dir / f"{key}.tif"
                if not tif_path.exists():
                    logger.warning("Missing '%s' in group '%s' — skipping group", tif_path, name)
                    skip = True
                    break
                raw = tifffile.imread(str(tif_path))
                images[key] = raw.astype(np.float64)
                logger.debug("Loaded %s: shape=%s dtype=%s", tif_path, raw.shape, raw.dtype)

            if not skip:
                groups[name] = images
                logger.info("Loaded group '%s': shape=%s", name, images[self.REFERENCE_KEY].shape)

        if not groups:
            raise ValueError(f"No valid image groups found in '{data_dir}'")

        logger.info("Loaded %d groups successfully", len(groups))
        return groups

    def estimate_params(
        self,
        image_groups: Dict[str, Dict[str, np.ndarray]],
        n_bins: int = 200,
        min_samples_per_bin: int = 30,
        background_percentile: float = 5.0,
        robust: bool = True,
    ) -> NoiseModelParams:
        """Estimate *a* and *b* using Multi-Level Mean-Variance linear regression.

        Algorithm:

        1. Compute a background-exclusion threshold from the global 1st–99.5th
           percentile range of all avg16 reference images.
        2. For each group and each N ∈ {1, 4, 8}:

           a. Compute residuals: ``residual_N = avg_N − avg16``.
           b. Bin foreground pixels of avg16 into *n_bins* equal-width bins.
           c. For each bin with ≥ *min_samples_per_bin* pixels compute:

              - ``x_mean``: mean clean intensity in the bin.
              - ``Var_corrected = Var_emp / (1/N + 1/16)``.

        3. Pool all ``(x_mean, Var_corrected, pixel_count)`` triples and fit::

               Var_corrected = slope · x + intercept
               a = 1/slope,  b = sqrt(intercept)

        4. If *robust* is True, use Iteratively Reweighted Least Squares (IRLS)
           with Huber loss to down-weight outlier bins.

        After this call, :attr:`last_fit_data` holds ``{'x', 'var',
        'n_values', 'weights'}`` arrays for plotting.

        Args:
            image_groups: Mapping returned by :meth:`load_image_groups`.
            n_bins: Number of equal-width intensity bins for the regression.
            min_samples_per_bin: Minimum pixels required per bin; bins with
                fewer samples are discarded as statistically unreliable.
            background_percentile: Pixels whose avg16 intensity is below this
                percentile of the global intensity distribution are excluded
                (dark background does not follow the Poisson model).
            robust: Whether to use IRLS robust fitting (recommended).

        Returns:
            :class:`NoiseModelParams` with ``a``, ``b``, and diagnostics.

        Raises:
            ValueError: If fewer than 10 valid (bin, N) data points are found.
        """
        # ----------------------------------------------------------
        # Global intensity thresholds (computed across all groups)
        # ----------------------------------------------------------
        all_ref_pixels = np.concatenate(
            [g[self.REFERENCE_KEY].ravel() for g in image_groups.values()]
        )
        bg_threshold = float(np.percentile(all_ref_pixels, background_percentile))
        global_max = float(np.percentile(all_ref_pixels, 99.5))

        logger.info(
            "Background threshold (%.1f-pct): %.1f  |  Global max (99.5-pct): %.1f",
            background_percentile, bg_threshold, global_max,
        )

        bin_edges = np.linspace(bg_threshold, global_max, n_bins + 1)

        # ----------------------------------------------------------
        # Collect (x, var_corrected, weight, N) tuples
        # ----------------------------------------------------------
        all_x: List[float] = []
        all_var: List[float] = []
        all_weights: List[float] = []
        all_n_vals: List[int] = []
        n_groups_used = 0

        for group_name, images in image_groups.items():
            clean: np.ndarray = images[self.REFERENCE_KEY]
            clean_flat = clean.ravel()
            fg_mask = clean_flat > bg_threshold
            bin_indices = np.digitize(clean_flat, bin_edges) - 1   # 0-based bin index
            group_pts = 0

            for key in self.NOISY_KEYS:
                N = self.FRAME_COUNTS[key]
                avg_N: np.ndarray = images[key]

                # Residual and correction factor
                residual_flat = (avg_N - clean).ravel()
                correction = 1.0 / N + 1.0 / self.REFERENCE_N   # Var divides by this

                for b_idx in range(n_bins):
                    in_bin = fg_mask & (bin_indices == b_idx)
                    count = int(in_bin.sum())

                    if count < min_samples_per_bin:
                        continue

                    x_mean = float(clean_flat[in_bin].mean())
                    if x_mean <= 0.0:
                        continue

                    # Unbiased sample variance of residuals (ddof=1)
                    var_emp = float(np.var(residual_flat[in_bin], ddof=1))
                    var_corrected = var_emp / correction

                    if var_corrected < 0.0:
                        continue

                    all_x.append(x_mean)
                    all_var.append(var_corrected)
                    all_weights.append(float(count))
                    all_n_vals.append(N)
                    group_pts += 1

            if group_pts > 0:
                n_groups_used += 1
                logger.debug("Group '%s': %d (bin, N) pairs", group_name, group_pts)

        if len(all_x) < 10:
            raise ValueError(
                f"Only {len(all_x)} valid (bin, N) data points found — not enough "
                "for reliable regression.  Consider reducing min_samples_per_bin "
                "or background_percentile."
            )

        x_arr = np.array(all_x, dtype=np.float64)
        var_arr = np.array(all_var, dtype=np.float64)
        w_arr = np.array(all_weights, dtype=np.float64)
        n_arr = np.array(all_n_vals, dtype=np.int32)

        logger.info(
            "Regression input: %d points from %d groups  (total %d pixels)",
            len(x_arr), n_groups_used, int(w_arr.sum()),
        )

        # ----------------------------------------------------------
        # Weighted linear regression
        # ----------------------------------------------------------
        if robust:
            slope, intercept, r_sq = self._irls_fit(x_arr, var_arr, w_arr)
        else:
            slope, intercept, r_sq = self._wls_fit(x_arr, var_arr, w_arr)

        # ----------------------------------------------------------
        # Physical constraints
        # ----------------------------------------------------------
        if intercept < 0.0:
            logger.warning(
                "Negative intercept (%.4f) → b² < 0.  "
                "Setting b = 0 and refitting slope only.",
                intercept,
            )
            # Fit origin-constrained model: Var = slope * x
            slope = float(np.average(var_arr / x_arr, weights=w_arr))
            intercept = 0.0
            y_pred = slope * x_arr
            ss_res = np.sum(w_arr * (var_arr - y_pred) ** 2)
            ss_tot = np.sum(w_arr * (var_arr - np.average(var_arr, weights=w_arr)) ** 2)
            r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        a = 1.0 / max(slope, 1e-12)
        b = float(np.sqrt(max(intercept, 0.0)))

        # ----------------------------------------------------------
        # Store raw data for visualisation
        # ----------------------------------------------------------
        self.last_fit_data = {
            'x': x_arr,
            'var': var_arr,
            'n_values': n_arr,
            'weights': w_arr,
        }

        params = NoiseModelParams(
            a=float(a),
            b=float(b),
            r_squared=float(r_sq),
            n_groups=n_groups_used,
            n_points=len(x_arr),
            n_pixels_used=int(w_arr.sum()),
            slope=float(slope),
            intercept=float(intercept),
            fit_details={
                'x_range': [float(x_arr.min()), float(x_arr.max())],
                'var_range': [float(var_arr.min()), float(var_arr.max())],
                'n_bins': n_bins,
                'min_samples_per_bin': min_samples_per_bin,
                'background_threshold': float(bg_threshold),
                'global_max': float(global_max),
                'robust': robust,
            },
        )

        logger.info(
            "Estimated: a=%.4f, b=%.4f, R²=%.4f  (slope=%.6f, intercept=%.4f)",
            a, b, r_sq, slope, intercept,
        )
        return params

    def validate_params(
        self,
        image_groups: Dict[str, Dict[str, np.ndarray]],
        params: NoiseModelParams,
        n_bins: int = 200,
        min_samples_per_bin: int = 30,
        background_percentile: float = 5.0,
    ) -> Dict:
        """Validate estimated parameters by checking per-N and per-group consistency.

        For each N ∈ {1, 4, 8} a separate weighted linear regression is fitted;
        the resulting a_N and b_N should be close to the jointly estimated values.

        A Shapiro-Wilk normality test is run on the standardised residuals of the
        joint fit (truncated to 5,000 points for speed).

        Args:
            image_groups: Mapping returned by :meth:`load_image_groups`.
            params: :class:`NoiseModelParams` from :meth:`estimate_params`.
            n_bins: Number of intensity bins.
            min_samples_per_bin: Minimum pixel count per bin.
            background_percentile: Background exclusion percentile.

        Returns:
            Validation report dictionary with keys:

            * ``'joint_params'``: dict with a, b, r_squared.
            * ``'per_N_params'``: dict mapping ``'N=<k>'`` to per-N fit stats.
            * ``'per_group_params'``: dict mapping group name to fit stats.
            * ``'residuals'``: list of standardised residuals (may be None).
            * ``'normality_test'``: Shapiro-Wilk result dict (may be None).
        """
        all_ref = np.concatenate(
            [g[self.REFERENCE_KEY].ravel() for g in image_groups.values()]
        )
        bg_threshold = float(np.percentile(all_ref, background_percentile))
        global_max = float(np.percentile(all_ref, 99.5))
        bin_edges = np.linspace(bg_threshold, global_max, n_bins + 1)

        report: Dict = {
            'joint_params': {'a': params.a, 'b': params.b, 'r_squared': params.r_squared},
            'per_N_params': {},
            'per_group_params': {},
            'residuals': None,
            'normality_test': None,
        }

        all_std_residuals: List[float] = []

        # ----------------------------------------------------------
        # Per-N fits
        # ----------------------------------------------------------
        for key in self.NOISY_KEYS:
            N = self.FRAME_COUNTS[key]
            correction = 1.0 / N + 1.0 / self.REFERENCE_N
            x_pts, var_pts, w_pts = [], [], []

            for images in image_groups.values():
                clean = images[self.REFERENCE_KEY]
                avg_N = images[key]
                clean_flat = clean.ravel()
                res_flat = (avg_N - clean).ravel()
                fg_mask = clean_flat > bg_threshold
                bin_idx = np.digitize(clean_flat, bin_edges) - 1

                for b_idx in range(n_bins):
                    in_bin = fg_mask & (bin_idx == b_idx)
                    count = int(in_bin.sum())
                    if count < min_samples_per_bin:
                        continue
                    x_mean = float(clean_flat[in_bin].mean())
                    if x_mean <= 0.0:
                        continue
                    var_emp = float(np.var(res_flat[in_bin], ddof=1))
                    var_c = var_emp / correction
                    if var_c < 0.0:
                        continue
                    x_pts.append(x_mean)
                    var_pts.append(var_c)
                    w_pts.append(float(count))

            if len(x_pts) < 5:
                continue

            x_a = np.array(x_pts)
            v_a = np.array(var_pts)
            w_a = np.array(w_pts)
            sl, ic, rsq = self._wls_fit(x_a, v_a, w_a)
            a_N = 1.0 / max(sl, 1e-12)
            b_N = float(np.sqrt(max(ic, 0.0)))

            report['per_N_params'][f'N={N}'] = {
                'a': float(a_N),
                'b': float(b_N),
                'r_squared': float(rsq),
                'n_points': len(x_pts),
                'a_rel_error_pct': abs(a_N - params.a) / params.a * 100.0 if params.a > 0 else float('nan'),
                'b_abs_error': abs(b_N - params.b),
            }

            # Standardised residuals against joint fit
            y_joint = (1.0 / params.a) * x_a + params.b ** 2
            sigma_approx = np.sqrt(2.0) * y_joint + 1e-10
            all_std_residuals.extend(((v_a - y_joint) / sigma_approx).tolist())

        # ----------------------------------------------------------
        # Per-group fits (joint over N)
        # ----------------------------------------------------------
        for group_name, images in image_groups.items():
            x_pts, var_pts, w_pts = [], [], []
            clean = images[self.REFERENCE_KEY]
            clean_flat = clean.ravel()
            fg_mask = clean_flat > bg_threshold
            bin_idx = np.digitize(clean_flat, bin_edges) - 1

            for key in self.NOISY_KEYS:
                N = self.FRAME_COUNTS[key]
                correction = 1.0 / N + 1.0 / self.REFERENCE_N
                res_flat = (images[key] - clean).ravel()

                for b_idx in range(n_bins):
                    in_bin = fg_mask & (bin_idx == b_idx)
                    count = int(in_bin.sum())
                    if count < min_samples_per_bin:
                        continue
                    x_mean = float(clean_flat[in_bin].mean())
                    if x_mean <= 0.0:
                        continue
                    var_emp = float(np.var(res_flat[in_bin], ddof=1))
                    var_c = var_emp / correction
                    if var_c < 0.0:
                        continue
                    x_pts.append(x_mean)
                    var_pts.append(var_c)
                    w_pts.append(float(count))

            if len(x_pts) < 5:
                continue

            sl, ic, rsq = self._wls_fit(np.array(x_pts), np.array(var_pts), np.array(w_pts))
            a_g = 1.0 / max(sl, 1e-12)
            b_g = float(np.sqrt(max(ic, 0.0)))

            report['per_group_params'][group_name] = {
                'a': float(a_g),
                'b': float(b_g),
                'r_squared': float(rsq),
            }

        # ----------------------------------------------------------
        # Normality test on standardised residuals
        # ----------------------------------------------------------
        if len(all_std_residuals) >= 8:
            res_arr = np.array(all_std_residuals)
            sample = res_arr[:min(len(res_arr), 5000)]
            stat, pval = stats.shapiro(sample)
            report['residuals'] = res_arr.tolist()
            report['normality_test'] = {
                'test': 'shapiro-wilk',
                'statistic': float(stat),
                'p_value': float(pval),
                'normal_at_5pct': bool(pval > 0.05),
            }
            logger.info(
                "Shapiro-Wilk normality test: W=%.4f, p=%.4f  (%s)",
                stat, pval, "PASS" if pval > 0.05 else "FAIL",
            )

        logger.info("Validation complete. Per-N results:")
        for label, v in report['per_N_params'].items():
            logger.info(
                "  %s: a=%.4f (±%.1f%%), b=%.4f (|err|=%.4f), R²=%.4f",
                label, v['a'], v['a_rel_error_pct'], v['b'], v['b_abs_error'], v['r_squared'],
            )

        return report

    def estimate_per_group_params(
        self,
        image_groups: Dict[str, Dict[str, np.ndarray]],
        n_bins: int = 200,
        min_samples_per_bin: int = 30,
        background_percentile: float = 5.0,
    ) -> Dict[str, Dict]:
        """Estimate separate (a, b) for each group independently.

        Convenience wrapper around :meth:`validate_params` that returns only the
        per-group parameter dictionary (useful for consistency box plots).

        Args:
            image_groups: Mapping from :meth:`load_image_groups`.
            n_bins: Number of intensity bins.
            min_samples_per_bin: Minimum pixels per bin.
            background_percentile: Background exclusion percentile.

        Returns:
            Dict mapping group_name → ``{'a', 'b', 'r_squared'}``.
        """
        # Fit a dummy joint params object (values not used here)
        dummy = NoiseModelParams(
            a=1.0, b=0.0, r_squared=0.0, n_groups=0, n_points=0,
            n_pixels_used=0, slope=1.0, intercept=0.0, fit_details={},
        )
        report = self.validate_params(
            image_groups, dummy, n_bins, min_samples_per_bin, background_percentile
        )
        return report['per_group_params']

    def save_params(self, params: NoiseModelParams, path: str) -> None:
        """Serialise *params* to a JSON file.

        Args:
            params: :class:`NoiseModelParams` to save.
            path: Output file path (will create parent directories).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as fh:
            json.dump(asdict(params), fh, indent=2)
        logger.info("Parameters saved to '%s'", out)

    def load_params(self, path: str) -> NoiseModelParams:
        """Deserialise :class:`NoiseModelParams` from a JSON file.

        Args:
            path: Path to JSON file produced by :meth:`save_params`.

        Returns:
            Reconstructed :class:`NoiseModelParams`.
        """
        with open(path) as fh:
            data = json.load(fh)
        params = NoiseModelParams(**data)
        logger.info("Loaded params from '%s': a=%.4f, b=%.4f", path, params.a, params.b)
        return params

    # ------------------------------------------------------------------
    # Private regression helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wls_fit(
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Weighted least squares: y = slope·x + intercept.

        Args:
            x: Predictor (bin mean intensities).
            y: Response (corrected variances).
            w: Sample weights (pixel counts).

        Returns:
            Tuple ``(slope, intercept, r_squared)``.
        """
        sw = w.sum()
        sw_x = (w * x).sum()
        sw_y = (w * y).sum()
        sw_xx = (w * x * x).sum()
        sw_xy = (w * x * y).sum()

        denom = sw * sw_xx - sw_x ** 2
        if abs(denom) < 1e-30:
            # Degenerate case (all x identical)
            slope = 0.0
            intercept = float(sw_y / sw) if sw > 0 else 0.0
        else:
            slope = (sw * sw_xy - sw_x * sw_y) / denom
            intercept = (sw_y - slope * sw_x) / sw

        y_pred = slope * x + intercept
        ss_res = float(np.sum(w * (y - y_pred) ** 2))
        ss_tot = float(np.sum(w * (y - sw_y / sw) ** 2)) if sw > 0 else 0.0
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

        return slope, intercept, r_sq

    def _irls_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-7,
        huber_c: float = 1.345,
    ) -> Tuple[float, float, float]:
        """Iteratively Reweighted Least Squares with Huber loss.

        Robust to outlier bins by iteratively down-weighting points whose
        residuals exceed *huber_c* × MAD-based scale estimate.

        Args:
            x: Predictor values.
            y: Response values.
            w: Initial pixel-count weights (normalised internally).
            max_iter: Maximum IRLS iterations.
            tol: Convergence tolerance on the L1 change of (slope, intercept).
            huber_c: Huber threshold in units of the MAD-derived σ.
                     1.345 gives 95 % asymptotic efficiency under normality.

        Returns:
            Tuple ``(slope, intercept, r_squared)``.
        """
        # Normalise initial weights so they don't swamp Huber weights
        irls_w = w * (len(w) / w.sum())

        slope, intercept, r_sq = self._wls_fit(x, y, irls_w)

        for i in range(max_iter):
            slope_prev, intercept_prev = slope, intercept
            residuals = y - (slope * x + intercept)

            # MAD-based robust scale (1.4826 → consistent σ under normality)
            mad = float(np.median(np.abs(residuals - np.median(residuals))))
            sigma = 1.4826 * mad

            if sigma < 1e-12:
                break

            # Huber weights: 1 for |r| ≤ c·σ, else c·σ/|r|
            r_scaled = np.abs(residuals) / (huber_c * sigma)
            huber_w = np.where(r_scaled <= 1.0, 1.0, 1.0 / r_scaled)

            combined_w = np.maximum(w * huber_w, 1e-12)
            slope, intercept, r_sq = self._wls_fit(x, y, combined_w)

            delta = abs(slope - slope_prev) + abs(intercept - intercept_prev)
            logger.debug("IRLS iter %d: slope=%.6f intercept=%.4f delta=%.2e", i + 1, slope, intercept, delta)

            if delta < tol:
                logger.info("IRLS converged at iteration %d", i + 1)
                break

        return slope, intercept, r_sq
