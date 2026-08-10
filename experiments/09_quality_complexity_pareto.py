#!/usr/bin/env python3
"""Fair multi-metric quality--complexity benchmark.

Classical baselines are tuned independently for RMSE and SSIM on the same
calibration image/noise realization used for the AMDI parameter sweep.  The
AMDI sweep is represented by its full quality--complexity Pareto fronts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.metrics import all_metrics
from amdi.pareto import pareto_front
from amdi.synthetic import add_gaussian_noise
from amdi.validation import (
    apply_baseline,
    build_truth,
    numeric,
    params_only,
    read_csv,
    run_amdi_from_row,
    tune_baselines,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default=str(ROOT / "results" / "08_amdi_parameter_sweep"))
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    results_file = sweep_dir / "sweep_results.csv"
    metadata_file = sweep_dir / "sweep_metadata.json"
    if not results_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Run experiments/08_amdi_parameter_sweep.py first.")

    meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    rows = [r for r in read_csv(results_file) if r.get("status") == "ok"]
    if not rows:
        raise RuntimeError("No successful AMDI runs found in sweep_results.csv")

    n = int(meta["n"])
    sigma = float(meta["noise_sigma"])
    noise_seed = int(meta["noise_seed"])
    outer_iterations = int(meta["outer_iterations"])
    truth = build_truth(meta["image"], n)
    noisy = add_gaussian_noise(truth, sigma, seed=noise_seed)
    out = ensure_dir(ROOT / "results" / "09_quality_complexity_pareto")

    trials, best_by_metric = tune_baselines(noisy, truth, sigma)
    write_csv(out / "baseline_tuning_trials.csv", trials)
    write_csv(out / "baseline_best_by_metric.csv", best_by_metric)
    baseline_rmse = [r for r in best_by_metric if r["selection_metric"] == "RMSE"]
    baseline_ssim = [r for r in best_by_metric if r["selection_metric"] == "SSIM"]
    # Backward-compatible file used by older analysis scripts.
    write_csv(out / "baseline_best.csv", baseline_rmse)

    for row in best_by_metric:
        print(
            f"{row['method']:14s} [{row['selection_metric']:4s}] "
            f"RMSE={float(row['RMSE']):.5f} SSIM={float(row['SSIM']):.4f} "
            f"params={params_only(row)}"
        )

    front_rmse = pareto_front(rows, "C_rel", "RMSE", minimize_y=True)
    front_ssim = pareto_front(rows, "C_rel", "SSIM", minimize_y=False)
    best_amdi_rmse = min(rows, key=lambda r: numeric(r, "RMSE"))
    best_amdi_ssim = max(rows, key=lambda r: numeric(r, "SSIM"))
    feasible_10 = [r for r in rows if numeric(r, "C_rel") <= 0.10]
    compressed_amdi = min(feasible_10, key=lambda r: numeric(r, "RMSE")) if feasible_10 else min(rows, key=lambda r: numeric(r, "C_rel"))

    best_baseline_rmse = min(baseline_rmse, key=lambda r: float(r["RMSE"]))
    best_baseline_ssim = max(baseline_ssim, key=lambda r: float(r["SSIM"]))
    assessment = {
        "best_baseline_RMSE_method": best_baseline_rmse["method"],
        "best_baseline_RMSE": float(best_baseline_rmse["RMSE"]),
        "best_baseline_SSIM_method": best_baseline_ssim["method"],
        "best_baseline_SSIM": float(best_baseline_ssim["SSIM"]),
        "best_AMDI_RMSE": numeric(best_amdi_rmse, "RMSE"),
        "best_AMDI_RMSE_C_rel": numeric(best_amdi_rmse, "C_rel"),
        "best_AMDI_SSIM": numeric(best_amdi_ssim, "SSIM"),
        "best_AMDI_SSIM_C_rel": numeric(best_amdi_ssim, "C_rel"),
        "AMDI_matches_or_beats_best_baseline_RMSE": numeric(best_amdi_rmse, "RMSE") <= float(best_baseline_rmse["RMSE"]),
        "AMDI_matches_or_beats_best_baseline_SSIM": numeric(best_amdi_ssim, "SSIM") >= float(best_baseline_ssim["SSIM"]),
        "compressed_AMDI_RMSE": numeric(compressed_amdi, "RMSE"),
        "compressed_AMDI_C_rel": numeric(compressed_amdi, "C_rel"),
        "compressed_AMDI_SSIM": numeric(compressed_amdi, "SSIM"),
    }
    write_json(out / "benchmark_assessment.json", assessment)

    summary = []
    for b in best_by_metric:
        summary.append({"family": "baseline", **b})
    summary.append({"family": "AMDI", "method": "AMDI best RMSE", **best_amdi_rmse})
    summary.append({"family": "AMDI", "method": "AMDI best SSIM", **best_amdi_ssim})
    summary.append({"family": "AMDI", "method": "AMDI <=10% DOF", **compressed_amdi})
    write_csv(out / "benchmark_summary.csv", summary)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    x = np.asarray([numeric(r, "C_rel") for r in rows])
    rm = np.asarray([numeric(r, "RMSE") for r in rows])
    ss = np.asarray([numeric(r, "SSIM") for r in rows])

    axes[0].scatter(x, rm, s=20, alpha=0.35, label="AMDI sweep")
    axes[0].plot([numeric(r, "C_rel") for r in front_rmse], [numeric(r, "RMSE") for r in front_rmse], marker="o", linewidth=1.8, label="AMDI Pareto front")
    for b in baseline_rmse:
        axes[0].scatter([1.0], [float(b["RMSE"])], marker="X", s=80)
        axes[0].annotate(b["method"], (1.0, float(b["RMSE"])), xytext=(-6, 5), textcoords="offset points", ha="right", fontsize=8)
    axes[0].set_xlabel(r"Relative complexity $\mathcal{C}_{\rm rel}$ [-]")
    axes[0].set_ylabel("RMSE [-]")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    axes[1].scatter(x, ss, s=20, alpha=0.35, label="AMDI sweep")
    axes[1].plot([numeric(r, "C_rel") for r in front_ssim], [numeric(r, "SSIM") for r in front_ssim], marker="o", linewidth=1.8, label="AMDI Pareto front")
    for b in baseline_ssim:
        axes[1].scatter([1.0], [float(b["SSIM"])], marker="X", s=80)
        axes[1].annotate(b["method"], (1.0, float(b["SSIM"])), xytext=(-6, 5), textcoords="offset points", ha="right", fontsize=8)
    axes[1].set_xlabel(r"Relative complexity $\mathcal{C}_{\rm rel}$ [-]")
    axes[1].set_ylabel("SSIM [-]")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "quality_complexity_pareto")

    amdi_best_img, _, _, _, _, _ = run_amdi_from_row(noisy, best_amdi_rmse, outer_iterations)
    amdi_comp_img, _, _, _, _, _ = run_amdi_from_row(noisy, compressed_amdi, outer_iterations)
    b_rmse_img = np.clip(apply_baseline(best_baseline_rmse["method"], noisy, params_only(best_baseline_rmse)), 0.0, 1.0)
    b_ssim_img = np.clip(apply_baseline(best_baseline_ssim["method"], noisy, params_only(best_baseline_ssim)), 0.0, 1.0)
    show = [
        ("Truth", truth, None),
        ("Noisy", noisy, all_metrics(noisy, truth)),
        (f"RMSE baseline\n{best_baseline_rmse['method']}", b_rmse_img, best_baseline_rmse),
        (f"SSIM baseline\n{best_baseline_ssim['method']}", b_ssim_img, best_baseline_ssim),
        ("AMDI best RMSE", amdi_best_img, best_amdi_rmse),
        ("AMDI <=10% DOF", amdi_comp_img, compressed_amdi),
    ]
    fig, axes = plt.subplots(1, len(show), figsize=(15.2, 3.5))
    for ax, (_, image, _) in zip(axes, show):
        show_normalized_image(ax, image, cmap="gray", vmin=0, vmax=1)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "selected_reconstructions")

    print("\nBenchmark assessment:")
    for k, v in assessment.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
