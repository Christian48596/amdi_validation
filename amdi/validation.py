"""Reusable publication-benchmark helpers for AMDI validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from skimage import data
from skimage.transform import resize

from .benchmarks import heat_diffusion, perona_malik, haar_soft_threshold, total_variation
from .energy import EnergyParameters
from .graph import GraphParameters
from .haar import adaptive_tree_from_detail_threshold, full_coefficients
from .metrics import all_metrics
from .solver import adaptive_frozen_denoise
from .synthetic import four_region_image


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric(row: dict, key: str) -> float:
    return float(row[key])


def build_truth(kind: str, n: int) -> np.ndarray:
    if kind == "synthetic":
        return four_region_image(n)
    if kind == "camera":
        arr = np.asarray(data.camera(), float) / 255.0
        return resize(arr, (n, n), anti_aliasing=True, preserve_range=True)
    raise ValueError(f"Unknown image kind: {kind}")


def run_amdi_from_row(
    noisy: np.ndarray,
    row: dict,
    outer_iterations: int,
    *,
    initial_threshold_scale: float = 1.0,
    h_scale: float = 1.0,
    alpha_scale: float = 1.0,
    beta_scale: float = 1.0,
    tau_scale: float = 1.0,
    stop_on_convergence: bool = False,
    update_tol: float = 1.0e-5,
    energy_tol: float = 1.0e-8,
    patience: int = 2,
):
    n = int(noisy.shape[0])
    if noisy.shape != (n, n) or n <= 0 or (n & (n - 1)):
        raise ValueError("AMDI validation expects a square power-of-two image")
    max_level = int(np.log2(n))
    full = full_coefficients(noisy, max_level=max_level)
    initial = adaptive_tree_from_detail_threshold(
        full,
        2,
        max_level,
        threshold=float(row["initial_threshold"]) * initial_threshold_scale,
        min_level=2,
    )
    ep = EnergyParameters(
        alpha=float(row["alpha"]) * alpha_scale,
        beta=float(row["beta"]) * beta_scale,
        tau=float(row["tau"]) * tau_scale,
        smooth_l1=False,
    )
    gp = GraphParameters(
        sigma_c=float(row["sigma_c"]),
        state_dependent=True,
        refinement_decay=float(row["refinement_decay"]),
    )
    image, tree, coeffs, history = adaptive_frozen_denoise(
        noisy,
        initial,
        full,
        ep,
        gp,
        h=float(row["h"]) * h_scale,
        outer_iterations=outer_iterations,
        adapt=True,
        zeta=float(row["zeta"]),
        refine_fraction=float(row["refine_fraction"]),
        coarsen_fraction=float(row["coarsen_fraction"]),
        stop_on_convergence=stop_on_convergence,
        update_tol=update_tol,
        energy_tol=energy_tol,
        patience=patience,
    )
    return np.clip(image, 0.0, 1.0), tree, coeffs, history, full, gp


def baseline_parameter_sets(n: int, sigma: float) -> list[tuple[str, list[dict]]]:
    return [
        ("Heat", [{"heat_sigma": float(v)} for v in np.linspace(0.25, 1.80, 17)]),
        (
            "Anisotropic",
            [
                {"iterations": int(it), "kappa": float(k), "dt": float(dt)}
                for it in (10, 20, 30)
                for k in (0.04, 0.06, 0.08, 0.10, 0.13, 0.16)
                for dt in (0.12, 0.18)
            ],
        ),
        (
            "Wavelet soft",
            [
                {
                    "threshold": float(mult * sigma * np.sqrt(2.0 * np.log(n * n)) / n),
                    "threshold_multiplier": float(mult),
                }
                for mult in np.linspace(0.15, 2.00, 20)
            ],
        ),
        ("TV", [{"weight": float(v)} for v in np.geomspace(0.008, 0.25, 22)]),
    ]


def apply_baseline(name: str, noisy: np.ndarray, params: dict) -> np.ndarray:
    if name == "Heat":
        return heat_diffusion(noisy, sigma=float(params["heat_sigma"]))
    if name == "Anisotropic":
        return perona_malik(
            noisy,
            iterations=int(float(params["iterations"])),
            kappa=float(params["kappa"]),
            dt=float(params["dt"]),
        )
    if name == "Wavelet soft":
        return haar_soft_threshold(noisy, threshold=float(params["threshold"]))
    if name == "TV":
        return total_variation(noisy, weight=float(params["weight"]))
    raise ValueError(name)


def tune_baselines(noisy: np.ndarray, truth: np.ndarray, sigma: float) -> tuple[list[dict], list[dict]]:
    """Return all baseline trials and best rows separately for RMSE and SSIM."""
    n = truth.shape[0]
    trials: list[dict] = []
    best: list[dict] = []
    for name, parameter_sets in baseline_parameter_sets(n, sigma):
        method_rows = []
        for params in parameter_sets:
            image = np.clip(apply_baseline(name, noisy, params), 0.0, 1.0)
            row = {"method": name, **params, **all_metrics(image, truth, active=truth.size, full=truth.size)}
            method_rows.append(row)
            trials.append(row)
        rmse_row = min(method_rows, key=lambda r: float(r["RMSE"]))
        ssim_row = max(method_rows, key=lambda r: float(r["SSIM"]))
        best.append({"selection_metric": "RMSE", **rmse_row})
        best.append({"selection_metric": "SSIM", **ssim_row})
    return trials, best


def params_only(row: dict) -> dict:
    ignore = {
        "method", "selection_metric", "RMSE", "PSNR", "SSIM", "C_rel",
        "family", "status", "runtime_s", "basis_size", "n_leaves",
        "max_level_used", "final_energy", "safeguard_accept_rate", "run_id",
        "error",
    }
    return {k: v for k, v in row.items() if k not in ignore and v not in (None, "")}


def mean_std(rows: Iterable[dict], key: str) -> tuple[float, float]:
    vals = np.asarray([float(r[key]) for r in rows], float)
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
