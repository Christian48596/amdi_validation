"""Image-quality and compression metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def rmse(u: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(u) - np.asarray(truth)) ** 2)))


def mse(u: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((np.asarray(u) - np.asarray(truth)) ** 2))


def psnr(u: np.ndarray, truth: np.ndarray, data_range: float = 1.0) -> float:
    m = mse(u, truth)
    if m == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / m))


def ssim(u: np.ndarray, truth: np.ndarray, data_range: float = 1.0) -> float:
    return float(structural_similarity(np.asarray(truth), np.asarray(u), data_range=data_range))


def all_metrics(u: np.ndarray, truth: np.ndarray, active: int | None = None, full: int | None = None) -> dict:
    out = {"RMSE": rmse(u, truth), "PSNR": psnr(u, truth), "SSIM": ssim(u, truth)}
    if active is not None and full is not None:
        out["C_rel"] = float(active) / float(full)
    return out
