"""Reference denoising baselines used in Section 9."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle

from .haar import AdaptiveHaarTree, full_coefficients, reconstruct


def heat_diffusion(noisy: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter(np.asarray(noisy, float), sigma=sigma, mode="reflect")


def perona_malik(
    image: np.ndarray,
    iterations: int = 20,
    kappa: float = 0.08,
    dt: float = 0.18,
) -> np.ndarray:
    """Four-neighbour Perona-Malik anisotropic diffusion."""
    u = np.asarray(image, float).copy()
    for _ in range(iterations):
        north = np.roll(u, -1, axis=0) - u
        south = np.roll(u, 1, axis=0) - u
        east = np.roll(u, -1, axis=1) - u
        west = np.roll(u, 1, axis=1) - u
        # Remove periodic coupling introduced by roll.
        north[-1, :] = 0.0
        south[0, :] = 0.0
        east[:, -1] = 0.0
        west[:, 0] = 0.0
        flux = 0.0
        for d in (north, south, east, west):
            g = np.exp(-(d / max(kappa, 1.0e-15)) ** 2)
            flux = flux + g * d
        u = u + dt * flux
    return np.clip(u, 0.0, 1.0)


def haar_soft_threshold(noisy: np.ndarray, threshold: float) -> np.ndarray:
    arr = np.asarray(noisy, float)
    level = min(int(np.log2(n)) for n in arr.shape)
    tree = AdaptiveHaarTree.uniform(arr.ndim, level)
    coeffs = full_coefficients(arr, level)
    for idx in list(coeffs):
        if idx.kind == "wavelet":
            coeffs[idx] = float(np.sign(coeffs[idx]) * max(abs(coeffs[idx]) - threshold, 0.0))
    return np.clip(reconstruct(coeffs, tree, arr.shape), 0.0, 1.0)


def total_variation(noisy: np.ndarray, weight: float = 0.08) -> np.ndarray:
    return np.asarray(denoise_tv_chambolle(np.asarray(noisy, float), weight=weight, channel_axis=None), float)
