"""
Visualisation utilities for Poisson-Gaussian noise model analysis.

Functions
---------
plot_mean_variance_curve     — Mean-variance scatter + fitted regression line.
plot_noise_level_comparison  — Side-by-side clean vs. synthetic noisy images.
plot_per_group_consistency   — Box plots of per-group a / b estimates.
plot_residual_distribution   — Residual histogram + Gaussian overlay + QQ-plot.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use('Agg')   # Non-interactive backend; works in scripts and notebooks
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Colour palette keyed by N value
_N_COLOURS: Dict[int, str] = {1: '#d62728', 4: '#2ca02c', 8: '#1f77b4'}
_N_LABELS:  Dict[int, str] = {1: 'N=1 (single frame)', 4: 'N=4 frames', 8: 'N=8 frames'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, path: Optional[str], label: str) -> None:
    """Save *fig* to *path* (if given) and log the result."""
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches='tight')
        logger.info("%s plot saved to '%s'", label, out)


# ---------------------------------------------------------------------------
# 1. Mean-variance curve
# ---------------------------------------------------------------------------

def plot_mean_variance_curve(
    estimator_results: Dict[str, np.ndarray],
    a: float,
    b: float,
    r_squared: float,
    output_path: Optional[str] = None,
    title: str = 'Mean-Variance Relationship (Poisson-Gaussian Model)',
) -> plt.Figure:
    """Plot mean-intensity vs. corrected noise variance with the fitted line.

    Args:
        estimator_results: Dict returned alongside the fit; must contain:

            * ``'x'``       – 1-D array of bin-mean clean intensities (ADU).
            * ``'var'``     – 1-D array of corrected variance values (ADU²).
            * ``'n_values'``– 1-D int array of N values per data point.
            * ``'weights'`` – 1-D array of pixel counts (for marker sizing).

        a: Estimated Poisson gain.
        b: Estimated Gaussian noise std.
        r_squared: Coefficient of determination from the fit.
        output_path: If given, save the figure to this file path (PNG/PDF).
        title: Figure title string.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    x_arr = np.asarray(estimator_results['x'])
    var_arr = np.asarray(estimator_results['var'])
    n_arr = np.asarray(estimator_results['n_values'])
    w_arr = np.asarray(estimator_results['weights'])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter points coloured by N
    w_norm = w_arr / w_arr.max()   # normalise to [0, 1]
    for N, colour in _N_COLOURS.items():
        mask = n_arr == N
        if not mask.any():
            continue
        ax.scatter(
            x_arr[mask], var_arr[mask],
            c=colour,
            alpha=0.55,
            s=np.clip(w_norm[mask] * 60, 3, 60),
            label=_N_LABELS.get(N, f'N={N}'),
            edgecolors='none',
            rasterized=True,
        )

    # Fitted line: Var = (1/a)·x + b²
    x_fit = np.linspace(x_arr.min(), x_arr.max(), 300)
    y_fit = (1.0 / a) * x_fit + b ** 2
    ax.plot(x_fit, y_fit, 'k-', lw=2.0,
            label=f'Fit: Var = x / {a:.3f} + {b**2:.2f}',
            zorder=5)

    # ±5 % shaded band (visual guide only)
    ax.fill_between(x_fit, y_fit * 0.95, y_fit * 1.05,
                    color='gray', alpha=0.15, label='±5 % band')

    ax.set_xlabel('Mean intensity — clean reference (ADU)', fontsize=12)
    ax.set_ylabel('Corrected noise variance (ADU²)', fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Parameter annotation box
    text_lines = [
        f'$a$ = {a:.3f}',
        f'$b$ = {b:.3f} ADU',
        f'$R^2$ = {r_squared:.4f}',
        f'$n_{{points}}$ = {len(x_arr):,}',
    ]
    ax.text(
        0.97, 0.06, '\n'.join(text_lines),
        transform=ax.transAxes,
        fontsize=10.5, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7),
    )

    plt.tight_layout()
    _save_figure(fig, output_path, 'Mean-variance')
    return fig


# ---------------------------------------------------------------------------
# 2. Noise-level comparison
# ---------------------------------------------------------------------------

def plot_noise_level_comparison(
    clean_image: np.ndarray,
    noisy_images: Dict[float, np.ndarray],
    output_path: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'gray',
    crop: Optional[int] = None,
) -> plt.Figure:
    """Side-by-side display of the clean image and synthetic noisy images.

    Args:
        clean_image: Reference clean image, shape ``(H, W)``.
        noisy_images: Dict mapping *n_frames* (float) → noisy image array.
                      The dict will be displayed in ascending frame-count order.
        output_path: Optional save path.
        vmin: Lower display intensity limit (defaults to 1st percentile of clean).
        vmax: Upper display intensity limit (defaults to 99th percentile of clean).
        cmap: Matplotlib colourmap name.
        crop: If given, display only the central ``crop × crop`` pixel region.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    clean_f = clean_image.astype(np.float64)

    if vmin is None:
        vmin = float(np.percentile(clean_f, 1))
    if vmax is None:
        vmax = float(np.percentile(clean_f, 99))

    # Sort by ascending n_frames
    sorted_items = sorted(noisy_images.items(), key=lambda kv: kv[0])
    n_panels = len(sorted_items) + 1     # +1 for clean

    fig, axes = plt.subplots(1, n_panels, figsize=(3.8 * n_panels, 4.2))
    if n_panels == 1:
        axes = [axes]

    def _crop(arr: np.ndarray) -> np.ndarray:
        if crop is None:
            return arr
        h, w = arr.shape[-2], arr.shape[-1]
        cy, cx = h // 2, w // 2
        half = crop // 2
        return arr[..., max(0, cy - half): cy + half, max(0, cx - half): cx + half]

    def _snr(noisy: np.ndarray) -> float:
        noise_std = float((noisy.astype(np.float64) - clean_f).std())
        return float(clean_f.mean() / max(noise_std, 1e-12))

    # Clean panel
    axes[0].imshow(_crop(clean_image), cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title('Clean reference\n(avg16)', fontsize=10)
    axes[0].axis('off')

    # Noisy panels
    for i, (nf, img) in enumerate(sorted_items, start=1):
        axes[i].imshow(_crop(img), cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        snr = _snr(img)
        axes[i].set_title(f'N = {nf:.0f} frames\nSNR ≈ {snr:.1f}', fontsize=10)
        axes[i].axis('off')

    fig.suptitle('Synthetic Poisson-Gaussian Noise Levels', fontsize=12, y=1.01)
    plt.tight_layout()
    _save_figure(fig, output_path, 'Noise-level comparison')
    return fig


# ---------------------------------------------------------------------------
# 3. Per-group consistency
# ---------------------------------------------------------------------------

def plot_per_group_consistency(
    per_group_params: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Box plot of per-group *a* and *b* estimates to assess cross-group stability.

    Args:
        per_group_params: Mapping ``{group_name: {'a': float, 'b': float, ...}}``
                          as returned by
                          :meth:`~noise_model.PoissonGaussianEstimator.estimate_per_group_params`.
        output_path: Optional save path.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    groups = sorted(per_group_params.keys())
    if not groups:
        logger.warning("per_group_params is empty — nothing to plot")
        fig, _ = plt.subplots()
        return fig

    a_vals = [per_group_params[g]['a'] for g in groups]
    b_vals = [per_group_params[g]['b'] for g in groups]
    n_groups = len(groups)

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, n_groups)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, vals, colour, label in [
        (ax_a, a_vals, 'steelblue', 'Poisson gain  $a$  (per group)'),
        (ax_b, b_vals, 'salmon',    'Gaussian noise std  $b$  (per group)'),
    ]:
        bp = ax.boxplot(vals, vert=True, widths=0.45,
                        patch_artist=True,
                        boxprops=dict(facecolor='lightyellow', linewidth=1.2),
                        medianprops=dict(color='red', linewidth=2),
                        flierprops=dict(marker='x', markeredgecolor='gray'))
        ax.scatter(
            np.ones(n_groups) + jitter, vals,
            c=colour, alpha=0.7, s=35, zorder=5, edgecolors='white', linewidths=0.5,
        )

        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if n_groups > 1 else 0.0
        cv_v = std_v / mean_v * 100 if mean_v > 0 else 0.0

        ax.set_title(label, fontsize=12, pad=6)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_xticks([1])
        ax.set_xticklabels([f'n={n_groups} groups'])
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.text(
            0.97, 0.97,
            f'mean = {mean_v:.3f}\nstd = {std_v:.3f}\nCV = {cv_v:.1f} %',
            transform=ax.transAxes, va='top', ha='right', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.6),
        )

    fig.suptitle('Per-Group Parameter Consistency', fontsize=13, y=1.02)
    plt.tight_layout()
    _save_figure(fig, output_path, 'Per-group consistency')
    return fig


# ---------------------------------------------------------------------------
# 4. Residual distribution (histogram + QQ-plot)
# ---------------------------------------------------------------------------

def plot_residual_distribution(
    residuals: Union[np.ndarray, List[float]],
    output_path: Optional[str] = None,
    title: str = 'Fit Residual Distribution',
) -> plt.Figure:
    """Histogram of standardised residuals with a Gaussian overlay and a QQ-plot.

    A Shapiro-Wilk normality test result is annotated on the histogram panel.

    Args:
        residuals: 1-D array (or list) of standardised fit residuals.
        output_path: Optional save path.
        title: Figure title.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    res = np.asarray(residuals, dtype=np.float64)

    if res.size == 0:
        logger.warning("residuals array is empty — returning blank figure")
        fig, _ = plt.subplots()
        return fig

    # ---- Shapiro-Wilk (subsample if > 5000 points for speed) ----
    sample = res[:5000] if res.size > 5000 else res
    sw_stat, sw_pval = stats.shapiro(sample)
    sw_label = 'PASS' if sw_pval > 0.05 else 'FAIL'

    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Histogram ----
    n_bins = min(60, max(10, res.size // 20))
    ax_hist.hist(
        res, bins=n_bins, density=True,
        color='steelblue', alpha=0.7, edgecolor='white',
        label='Standardised residuals',
    )
    mu, sigma = float(res.mean()), float(res.std(ddof=1))
    x_pdf = np.linspace(res.min(), res.max(), 300)
    ax_hist.plot(x_pdf, stats.norm.pdf(x_pdf, mu, sigma),
                 'r-', lw=2, label=f'N({mu:.2f}, {sigma:.2f}²)')

    ax_hist.set_xlabel('Standardised residual', fontsize=11)
    ax_hist.set_ylabel('Density', fontsize=11)
    ax_hist.set_title(title, fontsize=12)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3, linestyle='--')

    # Shapiro-Wilk annotation
    ax_hist.text(
        0.97, 0.97,
        f'Shapiro-Wilk\nW = {sw_stat:.4f}\np = {sw_pval:.4f}\n[{sw_label} @ α=5%]',
        transform=ax_hist.transAxes, va='top', ha='right', fontsize=9.5,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='wheat', alpha=0.7),
    )

    # ---- QQ-plot ----
    (osm, osr), (slope, intercept, r) = stats.probplot(res, dist='norm')
    ax_qq.scatter(osm, osr, s=6, alpha=0.45, color='steelblue', rasterized=True,
                  label='Data quantiles')
    line_x = np.array([osm[0], osm[-1]])
    ax_qq.plot(line_x, slope * line_x + intercept, 'r-', lw=2, label='Normal reference')

    ax_qq.set_xlabel('Theoretical quantiles  (Normal)', fontsize=11)
    ax_qq.set_ylabel('Sample quantiles', fontsize=11)
    ax_qq.set_title(f'Normal QQ-Plot  (r = {r:.4f})', fontsize=12)
    ax_qq.legend(fontsize=9)
    ax_qq.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    _save_figure(fig, output_path, 'Residual distribution')
    return fig
