#!/usr/bin/env python3
"""Holdout multi-seed benchmark with parameters frozen after calibration.

Experiment 08/09 are calibration studies.  This script fixes those selected
parameters and evaluates them on unseen Gaussian-noise realizations.  It is the
preferred source for publication mean +/- standard-deviation performance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.haar import reconstruct
from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.metrics import all_metrics
from amdi.solver import frozen_tree_refit
from amdi.synthetic import add_gaussian_noise
from amdi.validation import (
    apply_baseline,
    build_truth,
    mean_std,
    params_only,
    read_csv,
    run_amdi_from_row,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default=str(ROOT / "results" / "08_amdi_parameter_sweep"))
    parser.add_argument("--benchmark-dir", default=str(ROOT / "results" / "09_quality_complexity_pareto"))
    parser.add_argument("--refit-dir", default=str(ROOT / "results" / "11_amdi_debias_refit"))
    parser.add_argument("--seeds", default="101,103,107,109,113,127,131,137")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    bench_dir = Path(args.benchmark_dir)
    meta = json.loads((sweep_dir / "sweep_metadata.json").read_text(encoding="utf-8"))
    best = json.loads((sweep_dir / "best_parameters.json").read_text(encoding="utf-8"))
    baseline_file = bench_dir / "baseline_best_by_metric.csv"
    if not baseline_file.exists():
        raise FileNotFoundError("Run the final experiment 09 first (baseline_best_by_metric.csv is missing).")
    baseline_rows = read_csv(baseline_file)

    refit_file = Path(args.refit_dir) / "best_refits.json"
    refit = json.loads(refit_file.read_text(encoding="utf-8")) if refit_file.exists() else {}

    n = int(meta["n"])
    sigma = float(meta["noise_sigma"])
    outer_iterations = int(meta["outer_iterations"])
    truth = build_truth(meta["image"], n)
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    calibration_seed = int(meta["noise_seed"])
    if calibration_seed in seeds:
        raise ValueError("Holdout seeds must not include the calibration noise seed")

    out = ensure_dir(ROOT / "results" / "13_holdout_multiseed_benchmark")
    rows = []

    for seed in seeds:
        noisy = add_gaussian_noise(truth, sigma, seed=seed)

        for b in baseline_rows:
            t0 = time.perf_counter()
            image = np.clip(apply_baseline(b["method"], noisy, params_only(b)), 0.0, 1.0)
            rows.append({
                "seed": seed,
                "family": "baseline",
                "method": f"{b['method']} [{b['selection_metric']}-tuned]",
                "selection_metric": b["selection_metric"],
                **all_metrics(image, truth, active=n*n, full=n*n),
                "runtime_s": time.perf_counter() - t0,
            })

        amdi_variants = [
            ("AMDI best RMSE", best["best_rmse"]),
            ("AMDI compressed", best["best_under_10pct_complexity"]),
        ]
        for name, row in amdi_variants:
            t0 = time.perf_counter()
            image, tree, coeffs, history, full, gp = run_amdi_from_row(noisy, row, outer_iterations)
            metrics = all_metrics(image, truth, active=tree.basis_size(), full=n*n)
            rows.append({
                "seed": seed,
                "family": "AMDI",
                "method": name,
                "selection_metric": "calibration",
                **metrics,
                "iterations": int(history[-1]["iteration"]),
                "safeguard_accept_rate": float(np.mean([h.get("safeguard_accepted", True) for h in history[1:]])) if len(history) > 1 else 1.0,
                "runtime_s": time.perf_counter() - t0,
            })

            if name == "AMDI compressed" and "compressed" in refit:
                t1 = time.perf_counter()
                scale = float(refit["compressed"]["refit_alpha_scale"])
                refit_coeffs = frozen_tree_refit(tree, coeffs, full, float(row["alpha"]) * scale, gp)
                refit_image = np.clip(reconstruct(refit_coeffs, tree, noisy.shape), 0.0, 1.0)
                rows.append({
                    "seed": seed,
                    "family": "AMDI",
                    "method": "AMDI compressed + refit",
                    "selection_metric": "calibration",
                    **all_metrics(refit_image, truth, active=tree.basis_size(), full=n*n),
                    "refit_alpha_scale": scale,
                    "iterations": int(history[-1]["iteration"]),
                    "safeguard_accept_rate": float(np.mean([h.get("safeguard_accepted", True) for h in history[1:]])) if len(history) > 1 else 1.0,
                    "runtime_s": (time.perf_counter() - t1),
                })

        print(f"completed holdout seed {seed}")

    write_csv(out / "holdout_runs.csv", rows)

    methods = sorted({r["method"] for r in rows})
    summary = []
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        rm, rs = mean_std(rr, "RMSE")
        sm, ss = mean_std(rr, "SSIM")
        cm, cs = mean_std(rr, "C_rel")
        pm, ps = mean_std(rr, "PSNR")
        summary.append({
            "method": method,
            "n_seeds": len(rr),
            "RMSE_mean": rm, "RMSE_std": rs,
            "PSNR_mean": pm, "PSNR_std": ps,
            "SSIM_mean": sm, "SSIM_std": ss,
            "C_rel_mean": cm, "C_rel_std": cs,
        })
    summary.sort(key=lambda r: float(r["RMSE_mean"]))
    write_csv(out / "holdout_summary.csv", summary)

    best_rmse = min(summary, key=lambda r: float(r["RMSE_mean"]))
    best_ssim = max(summary, key=lambda r: float(r["SSIM_mean"]))
    assessment = {
        "calibration_seed": calibration_seed,
        "holdout_seeds": seeds,
        "best_mean_RMSE_method": best_rmse["method"],
        "best_mean_RMSE": best_rmse["RMSE_mean"],
        "best_mean_SSIM_method": best_ssim["method"],
        "best_mean_SSIM": best_ssim["SSIM_mean"],
        "protocol": "all method parameters frozen before holdout-noise evaluation",
    }
    write_json(out / "holdout_assessment.json", assessment)

    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))
    axes[0].errorbar(x, [r["RMSE_mean"] for r in summary], yerr=[r["RMSE_std"] for r in summary], fmt="o", capsize=3)
    axes[0].set_ylabel("RMSE, mean ± std [-]")
    axes[1].errorbar(x, [r["SSIM_mean"] for r in summary], yerr=[r["SSIM_std"] for r in summary], fmt="o", capsize=3)
    axes[1].set_ylabel("SSIM, mean ± std [-]")
    axes[2].errorbar(x, [r["C_rel_mean"] for r in summary], yerr=[r["C_rel_std"] for r in summary], fmt="o", capsize=3)
    axes[2].set_ylabel(r"$\mathcal{C}_{\rm rel}$, mean ± std [-]")
    for ax in axes:
        ax.set_xlabel("Method [-]")
        ax.set_xticks(x); ax.set_xticklabels([r["method"] for r in summary], rotation=55, ha="right", fontsize=7); ax.grid(alpha=0.2)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "holdout_multiseed_summary")

    print("\nHoldout summary:")
    for r in summary: print(r)


if __name__ == "__main__":
    main()
