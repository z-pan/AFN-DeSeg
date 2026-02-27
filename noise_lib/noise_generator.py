"""
Poisson-Gaussian noise synthesis for two-photon fluorescence microscopy images.

Noise model
-----------
For N-frame average with clean signal x:

    y_N = (1/a) · Poisson(a · x · N) / N + N(0, b²/N)

    Var[y_N] = (x/a + b²) / N

Synthesis modes
---------------
*Exact mode* — used when the Poisson rate λ = a·x·N ≤ ``_APPROX_THRESHOLD``:

    poisson_part   = Poisson(a · x · N) / (a · N)
    gaussian_part  = N(0, b² / N)
    y              = poisson_part + gaussian_part

*Gaussian approximation* — used when λ > ``_APPROX_THRESHOLD`` to avoid
numpy overflow (numpy.random.Generator.poisson clips lam to int64 range ≈ 9.2e18,
but very large values produce numerical artefacts):

    σ²  = (x/a + b²) / N
    y   = x + N(0, σ²)

The threshold defaults to 1 × 10⁶ (λ ≳ 1000 is already Gaussian to < 0.1 %
relative error).
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Poisson → Gaussian approximation threshold (on the per-pixel rate λ = a*x*N)
_APPROX_THRESHOLD: float = 1_000_000.0


class PoissonGaussianGenerator:
    """Synthesises Poisson-Gaussian noise from clean two-photon microscopy images.

    Attributes:
        a: Poisson gain (detector efficiency).  Must be > 0.
        b: Gaussian readout noise standard deviation.  Must be ≥ 0.

    Example::

        gen = PoissonGaussianGenerator(a=2.5, b=12.3)
        noisy = gen.add_noise(clean_img, n_frames=1, seed=0)

        # Or load from an estimated params file:
        gen = PoissonGaussianGenerator.from_params_file('outputs/noise_params.json')
        levels = gen.generate_noise_levels(clean_img, n_frames_list=[1, 4, 8, 16])
    """

    def __init__(self, a: float, b: float) -> None:
        """Initialise the generator.

        Args:
            a: Poisson gain (must be positive).
            b: Gaussian noise std (must be non-negative).

        Raises:
            ValueError: If *a* ≤ 0 or *b* < 0.
        """
        if a <= 0:
            raise ValueError(f"Poisson gain 'a' must be positive, got {a}")
        if b < 0:
            raise ValueError(f"Gaussian noise std 'b' must be non-negative, got {b}")
        self.a = float(a)
        self.b = float(b)
        logger.info("PoissonGaussianGenerator: a=%.4f, b=%.4f", self.a, self.b)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_params_file(cls, path: str) -> "PoissonGaussianGenerator":
        """Construct a generator from a JSON params file.

        The file must contain at minimum the keys ``'a'`` and ``'b'``
        (as produced by :meth:`~noise_model.PoissonGaussianEstimator.save_params`).

        Args:
            path: Path to the JSON params file.

        Returns:
            Initialised :class:`PoissonGaussianGenerator`.

        Raises:
            KeyError: If ``'a'`` or ``'b'`` is absent from the JSON.
            FileNotFoundError: If *path* does not exist.
        """
        with open(path) as fh:
            data = json.load(fh)
        a = float(data['a'])
        b = float(data['b'])
        logger.info("Loaded generator params from '%s': a=%.4f, b=%.4f", path, a, b)
        return cls(a=a, b=b)

    # ------------------------------------------------------------------
    # Core noise synthesis
    # ------------------------------------------------------------------

    def add_noise(
        self,
        clean_image: np.ndarray,
        n_frames: float = 1.0,
        seed: Optional[int] = None,
        approx_threshold: float = _APPROX_THRESHOLD,
    ) -> np.ndarray:
        """Add Poisson-Gaussian noise equivalent to averaging *n_frames* frames.

        The method automatically selects exact Poisson sampling for pixels where
        λ = a·x·N ≤ *approx_threshold* and a Gaussian approximation for
        larger rates, preventing integer-overflow in numpy's Poisson sampler.

        Input images of integer dtype (e.g. ``uint16``) are returned as the
        same dtype after rounding and clipping to ``[0, dtype.max]``.
        Float inputs are clipped to ``[0, 1]`` (assumed normalised) or to
        ``[0, max(clean_image)]`` if the image range exceeds 1.

        Args:
            clean_image: Clean reference image, shape ``(H, W)`` or ``(C, H, W)``.
                         Must contain non-negative values in the same unit as the
                         calibration data (e.g. raw 16-bit ADU counts).
            n_frames: Equivalent number of averaged frames (any positive float).
                      *n_frames = 1* → single-frame noise (highest);
                      *n_frames = 16* → 16-frame-average noise (lowest).
            seed: Optional integer seed for the numpy random number generator.
                  Using a fixed seed gives reproducible outputs.
            approx_threshold: Per-pixel Poisson rate above which the Gaussian
                              approximation is used (default 1e6).

        Returns:
            Noisy image with the same shape and dtype as *clean_image*,
            clipped to the valid intensity range.

        Raises:
            ValueError: If *n_frames* ≤ 0.
        """
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")

        orig_dtype = clean_image.dtype
        orig_shape = clean_image.shape

        # Determine output clip range from dtype
        if np.issubdtype(orig_dtype, np.integer):
            clip_max = float(np.iinfo(orig_dtype).max)
            clip_min = 0.0
        else:
            # Float image: preserve original range
            clip_min = 0.0
            clip_max = float(max(1.0, clean_image.max()))

        rng = np.random.default_rng(seed)
        x = clean_image.astype(np.float64)
        x_nn = np.maximum(x, 0.0)   # non-negative for Poisson

        # Per-pixel Poisson rate: λ = a · x · N
        lam = self.a * x_nn * n_frames

        noisy = np.empty_like(x)

        # ---- Exact Poisson regime ----
        mask_exact = lam <= approx_threshold
        if mask_exact.any():
            lam_ex = lam[mask_exact]
            # numpy default_rng.poisson handles float arrays correctly
            pois = rng.poisson(lam_ex).astype(np.float64)
            poisson_part = pois / (self.a * n_frames)

            sigma_g = self.b / math.sqrt(n_frames)
            gauss_part = rng.normal(0.0, sigma_g, size=lam_ex.shape) if sigma_g > 0 else np.zeros_like(lam_ex)

            noisy[mask_exact] = poisson_part + gauss_part

        # ---- Gaussian approximation regime ----
        mask_approx = ~mask_exact
        if mask_approx.any():
            x_ap = x_nn[mask_approx]
            var_ap = (x_ap / self.a + self.b ** 2) / n_frames
            sigma_ap = np.sqrt(np.maximum(var_ap, 0.0))
            noisy[mask_approx] = x_ap + rng.normal(0.0, 1.0, size=x_ap.shape) * sigma_ap

        n_approx = int(mask_approx.sum())
        if n_approx > 0:
            logger.debug(
                "add_noise: %d / %d pixels used Gaussian approximation (λ > %.0e)",
                n_approx, x.size, approx_threshold,
            )

        # Clip and cast back to original dtype
        noisy = np.clip(noisy, clip_min, clip_max)
        if np.issubdtype(orig_dtype, np.integer):
            noisy = np.round(noisy).astype(orig_dtype)
        else:
            noisy = noisy.astype(orig_dtype)

        return noisy

    # ------------------------------------------------------------------
    # Batch and multi-level helpers
    # ------------------------------------------------------------------

    def add_noise_batch(
        self,
        clean_images: List[np.ndarray],
        n_frames_list: List[float],
        seed: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Add noise to a list of images, each possibly with a different *n_frames*.

        Per-image random seeds are derived from *seed* + image index so that
        results are reproducible but statistically independent.

        Args:
            clean_images: List of clean input arrays.
            n_frames_list: Per-image frame counts.  If the list has length 1,
                           the single value is broadcast to all images.
            seed: Optional base random seed.

        Returns:
            List of noisy arrays in the same order as *clean_images*.

        Raises:
            ValueError: If ``len(n_frames_list) not in (1, len(clean_images))``.
        """
        n_imgs = len(clean_images)

        if len(n_frames_list) == 1:
            n_frames_list = list(n_frames_list) * n_imgs
        elif len(n_frames_list) != n_imgs:
            raise ValueError(
                f"n_frames_list length ({len(n_frames_list)}) must be 1 or "
                f"equal to the number of images ({n_imgs})"
            )

        results: List[np.ndarray] = []
        for i, (img, nf) in enumerate(zip(clean_images, n_frames_list)):
            per_seed = None if seed is None else seed + i
            results.append(self.add_noise(img, n_frames=float(nf), seed=per_seed))

        logger.info(
            "add_noise_batch: generated %d noisy images (n_frames=%s)",
            n_imgs, n_frames_list,
        )
        return results

    def generate_noise_levels(
        self,
        clean_image: np.ndarray,
        n_frames_list: Union[List[float], List[int]] = (1, 2, 4, 8, 16, 32),
        seed: Optional[int] = None,
    ) -> Dict[float, np.ndarray]:
        """Generate one noisy image per entry in *n_frames_list* from a single clean image.

        Useful for creating training data at multiple SNR levels or for
        visual comparison across noise levels.

        Args:
            clean_image: Single clean reference image, shape ``(H, W)`` or ``(C, H, W)``.
            n_frames_list: Sequence of equivalent frame counts to simulate.
                           Defaults to ``(1, 2, 4, 8, 16, 32)``.
            seed: Optional base random seed; per-level seeds are derived
                  as *seed* + level_index.

        Returns:
            Ordered dict mapping each frame count (float) → noisy image array.
            The dict is sorted in ascending order of frame count.
        """
        results: Dict[float, np.ndarray] = {}

        for i, nf in enumerate(sorted(set(float(v) for v in n_frames_list))):
            per_seed = None if seed is None else seed + i
            noisy = self.add_noise(clean_image, n_frames=nf, seed=per_seed)
            results[nf] = noisy

            if logger.isEnabledFor(logging.DEBUG):
                diff = noisy.astype(np.float64) - clean_image.astype(np.float64)
                logger.debug(
                    "n_frames=%.0f: noise_std=%.2f  (theoretical=%.2f)",
                    nf,
                    diff.std(),
                    math.sqrt((clean_image.mean() / self.a + self.b ** 2) / nf),
                )

        logger.info(
            "generate_noise_levels: produced %d images for n_frames=%s",
            len(results), sorted(results.keys()),
        )
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def theoretical_snr(self, mean_intensity: float, n_frames: float = 1.0) -> float:
        """Compute theoretical signal-to-noise ratio for a given mean intensity.

        SNR = x / sqrt(Var[y_N]) = x / sqrt((x/a + b²) / N)

        Args:
            mean_intensity: Mean clean signal level (in same ADU units as calibration).
            n_frames: Number of averaged frames.

        Returns:
            Predicted SNR (linear, not dB).
        """
        var = (mean_intensity / self.a + self.b ** 2) / n_frames
        return mean_intensity / math.sqrt(max(var, 1e-12))

    def __repr__(self) -> str:
        return f"PoissonGaussianGenerator(a={self.a:.4f}, b={self.b:.4f})"
