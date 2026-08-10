#!/usr/bin/env python3
"""Fixed-tree coefficient debias/refit after AMDI tree selection.

This experiment tests whether part of AMDI's RMSE cost is coefficient
shrinkage rather than structural under-resolution.  The adaptive tree is kept
strictly fixed, so the relative complexity does not change during refitting.
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

from amdi.haar import reconstruct
from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.metrics import all_metrics
from amdi.solver import frozen_tree_refit
from amdi.synthetic import add_gaussian_noise
from amdi.validation import build_truth, run_amdi_from_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default=str(ROOT / "results" / "08_amdi_parameter_sweep"))
    parser.add_argument("--alpha-scales", default="0,0.1,0.25,0.5,1,2")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    best_file = sweep_dir / "best_parameters.json"
    meta_file = sweep_dir / "sweep_metadata.json"
    if not best_file.exists() or not meta_file.exists():
        raise FileNotFoundError("Run experiment 08 first.")

    best = json.loads(best_file.read_text(encoding="utf-8"))
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    n = int(meta["n"])
    truth = build_truth(meta["image"], n)
    noisy = add_gaussian_noise(truth, float(meta["noise_sigma"]), seed=int(meta["noise_seed"]))
    outer_iterations = int(meta["outer_iterations"])
    alpha_scales = [float(v) for v in args.alpha_scales.split(",") if v.strip()]

    operating_points = {
        "best_rmse": best["best_rmse"],
        "compressed": best["best_under_10pct_complexity"],
    }
    out = ensure_dir(ROOT / "results" / "11_amdi_debias_refit")
    rows = []
    images = []
    best_refits = {}

    for name, row in operating_points.items():
        image, tree, coeffs, history, full, gp = run_amdi_from_row(noisy, row, outer_iterations)
        base_metrics = all_metrics(image, truth, active=tree.basis_size(), full=n*n)
        rows.append({"operating_point": name, "refit_alpha_scale": "original", **base_metrics})
        images.append((f"{name}\noriginal", image, base_metrics))

        local_rows = []
        for scale in alpha_scales:
            alpha_refit = float(row["alpha"]) * scale
            refit_coeffs = frozen_tree_refit(tree, coeffs, full, alpha_refit, gp)
            refit_image = np.clip(reconstruct(refit_coeffs, tree, noisy.shape), 0.0, 1.0)
            metrics = all_metrics(refit_image, truth, active=tree.basis_size(), full=n*n)
            entry = {
                "operating_point": name,
                "refit_alpha_scale": scale,
                "refit_alpha": alpha_refit,
                **metrics,
            }
            rows.append(entry)
            local_rows.append(entry)
        best_local = min(local_rows, key=lambda r: float(r["RMSE"]))
        best_refits[name] = best_local
        best_scale = float(best_local["refit_alpha_scale"])
        best_coeffs = frozen_tree_refit(tree, coeffs, full, float(row["alpha"]) * best_scale, gp)
        best_image = np.clip(reconstruct(best_coeffs, tree, noisy.shape), 0.0, 1.0)
        images.append((f"{name}\nrefit", best_image, best_local))

    write_csv(out / "debias_refit_results.csv", rows)
    write_json(out / "best_refits.json", best_refits)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for name in operating_points:
        rr = [r for r in rows if r["operating_point"] == name and r["refit_alpha_scale"] != "original"]
        x = np.asarray([float(r["refit_alpha_scale"]) for r in rr])
        axes[0].plot(x, [float(r["RMSE"]) for r in rr], marker="o", label=name)
        axes[1].plot(x, [float(r["SSIM"]) for r in rr], marker="o", label=name)
    axes[0].set_xlabel(r"Refit diffusion scale $\alpha_{\rm refit}/\alpha$ [-]")
    axes[0].set_ylabel("RMSE [-]")
    axes[1].set_xlabel(r"Refit diffusion scale $\alpha_{\rm refit}/\alpha$ [-]")
    axes[1].set_ylabel("SSIM [-]")
    for ax in axes:
        ax.grid(alpha=0.2); ax.legend(fontsize=8)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "debias_refit_metrics")

    fig, axes = plt.subplots(1, len(images) + 2, figsize=(3.0*(len(images)+2), 3.5))
    show_normalized_image(axes[0], truth, cmap="gray", vmin=0, vmax=1)
    show_normalized_image(axes[1], noisy, cmap="gray", vmin=0, vmax=1)
    for ax, (_, image, _) in zip(axes[2:], images):
        show_normalized_image(ax, image, cmap="gray", vmin=0, vmax=1)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "debias_refit_reconstructions")

    print("Best fixed-tree refits:")
    for k, v in best_refits.items():
        print(k, v)


if __name__ == "__main__":
    main()
