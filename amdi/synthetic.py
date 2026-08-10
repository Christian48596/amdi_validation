"""Synthetic 1D and 2D validation targets."""

from __future__ import annotations

import numpy as np


def reference_1d(n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n) + 0.5) / n
    u = 0.25 * np.sin(2.0 * np.pi * x)
    u += 0.55 * np.exp(-140.0 * (x - 0.32) ** 2)
    u += 0.45 * (x >= 0.68)
    return x, u


def smooth_reference_1d(n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n) + 0.5) / n
    u = np.sin(2.0 * np.pi * x) + 0.25 * np.cos(6.0 * np.pi * x)
    return x, u


def four_region_image(n: int = 128) -> np.ndarray:
    if n % 2:
        raise ValueError("n must be even")
    y, x = np.mgrid[0:n, 0:n]
    X = (x + 0.5) / n
    Y = (y + 0.5) / n
    img = np.zeros((n, n), float)

    # Upper-left: constant region.
    mask = (X < 0.5) & (Y < 0.5)
    img[mask] = 0.25

    # Upper-right: smooth gradient.
    mask = (X >= 0.5) & (Y < 0.5)
    img[mask] = 0.15 + 0.7 * (X[mask] - 0.5) / 0.5

    # Lower-left: sharp disk/discontinuity.
    mask_ll = (X < 0.5) & (Y >= 0.5)
    disk = ((X - 0.25) ** 2 + (Y - 0.75) ** 2) <= 0.12**2
    img[mask_ll] = 0.15
    img[disk] = 0.9

    # Lower-right: oscillatory texture.
    mask = (X >= 0.5) & (Y >= 0.5)
    img[mask] = 0.5 + 0.28 * np.sin(18 * np.pi * X[mask]) * np.sin(18 * np.pi * Y[mask])
    return np.clip(img, 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(np.asarray(image, float) + rng.normal(0.0, sigma, image.shape), 0.0, 1.0)
