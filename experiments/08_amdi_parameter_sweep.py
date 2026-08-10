#!/usr/bin/env python3
"""Systematic AMDI parameter sweep for the application benchmark.

The purpose of this experiment is not to force AMDI to outperform a baseline.
It maps reconstruction quality against adaptive complexity and identifies the
nondominated AMDI parameter sets.  The sweep is deterministic for a given
seed and writes every tested parameter combination to CSV.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np
from skimage import data
from skimage.transform import resize

from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from amdi.haar import adaptive_tree_from_detail_threshold, full_coefficients
from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.metrics import all_metrics
from amdi.pareto import best_under_budgets, pareto_front
from amdi.solver import adaptive_frozen_denoise
from amdi.synthetic import add_gaussian_noise, four_region_image


def build_truth(kind: str, n: int) -> np.ndarray:
    if kind == "synthetic":
        return four_region_image(n)
    if kind == "camera":
        arr = np.asarray(data.camera(), float) / 255.0
        return resize(arr, (n, n), anti_aliasing=True, preserve_range=True)
    raise ValueError(kind)


def parameter_pool() -> list[dict]:
    grid = {
        "alpha": [0.002, 0.005, 0.010, 0.020, 0.040],
        "beta": [0.0, 5.0e-5, 2.0e-4, 5.0e-4],
        "tau": [1.0e-8, 1.0e-7, 5.0e-7, 2.0e-6],
        "sigma_c": [0.05, 0.08, 0.12, 0.20, 0.35],
        "refinement_decay": [0.20, 0.35, 0.60],
        "h": [0.25, 0.50, 0.80, 1.20],
        "initial_threshold": [0.004, 0.008, 0.012, 0.020],
        "refine_fraction": [0.05, 0.10, 0.20],
        "coarsen_fraction": [0.05, 0.10, 0.20],
        "zeta": [0.0, 1.0e-7, 1.0e-6],
    }
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def choose_candidates(budget: int, seed: int) -> list[dict]:
    pool = parameter_pool()
    rng = np.random.default_rng(seed)

    # Always include the original manuscript-development setting first.
    anchor = {
        "alpha": 0.010,
        "beta": 2.0e-4,
        "tau": 1.0e-7,
        "sigma_c": 0.12,
        "refinement_decay": 0.35,
        "h": 0.60,
        "initial_threshold": 0.012,
        "refine_fraction": 0.10,
        "coarsen_fraction": 0.10,
        "zeta": 1.0e-6,
    }
    if budget <= 1:
        return [anchor]

    idx = rng.choice(len(pool), size=min(budget - 1, len(pool)), replace=False)
    chosen = [anchor]
    for i in idx:
        p = pool[int(i)]
        # Avoid exact duplication of the anchor if it happens to be sampled.
        if p != anchor:
            chosen.append(p)
        if len(chosen) >= budget:
            break
    return chosen


def rerun_image(noisy, full, params, max_level, outer_iterations):
    initial = adaptive_tree_from_detail_threshold(
        full, 2, max_level,
        threshold=float(params["initial_threshold"]),
        min_level=2,
    )
    ep = EnergyParameters(
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        tau=float(params["tau"]),
        smooth_l1=False,
    )
    gp = GraphParameters(
        sigma_c=float(params["sigma_c"]),
        state_dependent=True,
        refinement_decay=float(params["refinement_decay"]),
    )
    return adaptive_frozen_denoise(
        noisy, initial, full, ep, gp,
        h=float(params["h"]),
        outer_iterations=outer_iterations,
        adapt=True,
        zeta=float(params["zeta"]),
        refine_fraction=float(params["refine_fraction"]),
        coarsen_fraction=float(params["coarsen_fraction"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=48, help="number of AMDI parameter sets to test")
    parser.add_argument("--full", action="store_true", help="use a larger 192-point sweep")
    parser.add_argument("--image", choices=["synthetic", "camera"], default="synthetic")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--noise-seed", type=int, default=11)
    parser.add_argument("--outer-iterations", type=int, default=6)
    args = parser.parse_args()

    if args.full:
        args.budget = max(args.budget, 192)
    if args.n <= 0 or args.n & (args.n - 1):
        raise ValueError("--n must be a positive power of two")

    out = ensure_dir(ROOT / "results" / "08_amdi_parameter_sweep")
    truth = build_truth(args.image, args.n)
    noisy = add_gaussian_noise(truth, args.sigma, seed=args.noise_seed)
    max_level = int(np.log2(args.n))
    full = full_coefficients(noisy, max_level=max_level)

    candidates = choose_candidates(args.budget, args.seed)
    rows = []
    print(f"AMDI sweep: {len(candidates)} parameter sets, image={args.image}, n={args.n}, sigma={args.sigma}")

    for run_id, p in enumerate(candidates):
        t0 = time.perf_counter()
        row = {"run_id": run_id, "status": "ok", **p}
        try:
            image, tree, _, history = rerun_image(noisy, full, p, max_level, args.outer_iterations)
            image = np.clip(image, 0.0, 1.0)
            row.update(all_metrics(image, truth, active=tree.basis_size(), full=args.n * args.n))
            row.update({
                "basis_size": tree.basis_size(),
                "n_leaves": len(tree.leaves()),
                "max_level_used": max(level for level, _ in tree.leaves()),
                "final_energy": float(history[-1]["energy"]),
                "safeguard_accept_rate": float(np.mean([h.get("safeguard_accepted", True) for h in history[1:]])) if len(history) > 1 else 1.0,
            })
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        row["runtime_s"] = time.perf_counter() - t0
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"[{run_id+1:3d}/{len(candidates)}] "
                f"RMSE={row['RMSE']:.5f} SSIM={row['SSIM']:.4f} "
                f"C_rel={row['C_rel']:.4f} time={row['runtime_s']:.2f}s"
            )
        else:
            print(f"[{run_id+1:3d}/{len(candidates)}] FAILED: {row['error']}")

    write_csv(out / "sweep_results.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("All parameter sets failed; inspect sweep_results.csv")

    front_rmse = pareto_front(ok, "C_rel", "RMSE", minimize_y=True)
    front_ssim = pareto_front(ok, "C_rel", "SSIM", minimize_y=False)
    best_budgets = best_under_budgets(ok, metric="RMSE", minimize=True)
    write_csv(out / "pareto_rmse.csv", front_rmse)
    write_csv(out / "pareto_ssim.csv", front_ssim)
    write_csv(out / "best_by_complexity.csv", best_budgets)

    best_rmse = min(ok, key=lambda r: float(r["RMSE"]))
    best_ssim = max(ok, key=lambda r: float(r["SSIM"]))
    feasible_10 = [r for r in ok if float(r["C_rel"]) <= 0.10]
    best_10 = min(feasible_10, key=lambda r: float(r["RMSE"])) if feasible_10 else min(ok, key=lambda r: float(r["C_rel"]))

    write_json(out / "best_parameters.json", {
        "best_rmse": best_rmse,
        "best_ssim": best_ssim,
        "best_under_10pct_complexity": best_10,
    })
    write_json(out / "sweep_metadata.json", {
        "image": args.image,
        "n": args.n,
        "noise_sigma": args.sigma,
        "noise_seed": args.noise_seed,
        "sampling_seed": args.seed,
        "budget": len(candidates),
        "outer_iterations": args.outer_iterations,
        "parameter_selection": "deterministic random subset of a declared Cartesian grid plus the original anchor setting",
    })

    # Quality--complexity scatter with nondominated fronts.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    c = np.asarray([float(r["C_rel"]) for r in ok])
    rm = np.asarray([float(r["RMSE"]) for r in ok])
    ss = np.asarray([float(r["SSIM"]) for r in ok])
    axes[0].scatter(c, rm, s=24, alpha=0.55)
    axes[0].plot([float(r["C_rel"]) for r in front_rmse], [float(r["RMSE"]) for r in front_rmse], marker="o", linewidth=1.5)
    axes[0].set_xlabel(r"Relative complexity $\mathcal{C}_{\rm rel}$ [-]")
    axes[0].set_ylabel("RMSE [-]")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(c, ss, s=24, alpha=0.55)
    axes[1].plot([float(r["C_rel"]) for r in front_ssim], [float(r["SSIM"]) for r in front_ssim], marker="o", linewidth=1.5)
    axes[1].set_xlabel(r"Relative complexity $\mathcal{C}_{\rm rel}$ [-]")
    axes[1].set_ylabel("SSIM [-]")
    axes[1].grid(alpha=0.2)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "amdi_parameter_sweep")

    # Reconstruct representative operating points for direct visual inspection.
    representatives = [
        ("Truth", truth),
        ("Noisy", noisy),
    ]
    for label, row in (("Best RMSE", best_rmse), ("Best <=10% DOF", best_10)):
        image, _, _, _ = rerun_image(noisy, full, row, max_level, args.outer_iterations)
        representatives.append((label, np.clip(image, 0.0, 1.0)))

    fig, axes = plt.subplots(1, len(representatives), figsize=(11.0, 3.4))
    for ax, (_, image) in zip(axes, representatives):
        show_normalized_image(ax, image, cmap="gray", vmin=0, vmax=1)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "amdi_selected_operating_points")

    print("\nBest AMDI RMSE:", {k: best_rmse[k] for k in ("RMSE", "PSNR", "SSIM", "C_rel", "alpha", "beta", "tau", "sigma_c")})
    print("Best AMDI under 10% complexity:", {k: best_10[k] for k in ("RMSE", "PSNR", "SSIM", "C_rel", "alpha", "beta", "tau", "sigma_c")})


if __name__ == "__main__":
    main()
