#!/usr/bin/env python3
"""Matched-complexity ablation on common AMDI-selected adaptive spaces.

For each noise realization, the calibrated compressed AMDI operating point is
rerun and its final adaptive tree is retained. All coefficient/operator
ablations are then solved on exactly that same tree. Thus, within each noise
realization, every variant has identical C_rel and the comparison is not
confounded by different representation sizes.
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

from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.metrics import all_metrics
from amdi.solver import adaptive_frozen_denoise
from amdi.synthetic import add_gaussian_noise
from amdi.validation import build_truth, run_amdi_from_row


def _run_fixed(noisy, common_tree, full, row, *, alpha_scale=1.0, state_dependent=True,
               outer_iterations=12, update_tol=2.0e-5, energy_tol=2.0e-7):
    ep = EnergyParameters(
        alpha=float(row["alpha"]) * float(alpha_scale),
        beta=float(row["beta"]),
        tau=float(row["tau"]),
        smooth_l1=False,
    )
    gp = GraphParameters(
        sigma_c=float(row["sigma_c"]),
        state_dependent=bool(state_dependent),
        refinement_decay=float(row["refinement_decay"]),
    )
    return adaptive_frozen_denoise(
        noisy, common_tree, full, ep, gp,
        h=float(row["h"]),
        outer_iterations=outer_iterations,
        adapt=False,
        zeta=float(row["zeta"]),
        refine_fraction=float(row["refine_fraction"]),
        coarsen_fraction=float(row["coarsen_fraction"]),
        stop_on_convergence=True,
        update_tol=update_tol,
        energy_tol=energy_tol,
        patience=2,
    )


def _history_summary(history):
    steps = history[1:]
    return {
        "iterations": int(history[-1]["iteration"]),
        "energy_monotone_all_steps": bool(all(bool(h.get("energy_monotone", True)) for h in steps)),
        "safeguard_accept_rate": float(np.mean([bool(h.get("safeguard_accepted", True)) for h in steps])) if steps else 1.0,
        "final_relative_update": float(history[-1].get("relative_update", np.nan)),
        "final_relative_energy_change": float(history[-1].get("relative_energy_change", np.nan)),
        "final_energy": float(history[-1]["energy"]),
    }


def _mean_std(rows, key):
    vals = np.asarray([float(r[key]) for r in rows], float)
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default=str(ROOT / "results" / "08_amdi_parameter_sweep"))
    parser.add_argument("--operating-point", choices=["best_rmse", "best_ssim", "best_under_10pct_complexity"],
                        default="best_under_10pct_complexity")
    parser.add_argument("--noise-seeds", default="11,17,23,31,43")
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--update-tol", type=float, default=2.0e-5)
    parser.add_argument("--energy-tol", type=float, default=2.0e-7)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    best = json.loads((sweep_dir / "best_parameters.json").read_text(encoding="utf-8"))
    meta = json.loads((sweep_dir / "sweep_metadata.json").read_text(encoding="utf-8"))
    row = best[args.operating_point]

    n = int(meta["n"])
    sigma = float(meta["noise_sigma"])
    truth = build_truth(meta["image"], n)
    seeds = [int(v) for v in args.noise_seeds.split(",") if v.strip()]
    out = ensure_dir(ROOT / "results" / "16_matched_complexity_ablation")

    variant_specs = [
        ("Adaptive AMDI reference", "adaptive"),
        ("Fixed selected tree", "fixed_state_dependent"),
        ("No intrinsic diffusion", "no_diffusion"),
        ("State-independent weights", "linear_weights"),
    ]

    rows = []
    representative_images = None
    for seed in seeds:
        noisy = add_gaussian_noise(truth, sigma, seed=seed)
        ref_image, common_tree, _, ref_history, full, _ = run_amdi_from_row(
            noisy, row, args.outer_iterations,
            stop_on_convergence=True,
            update_tol=args.update_tol,
            energy_tol=args.energy_tol,
            patience=2,
        )
        target_complexity = common_tree.basis_size() / float(n*n)

        local = [("Adaptive AMDI reference", np.clip(ref_image, 0, 1), common_tree, ref_history,
                  "adaptive tree evolution + state-dependent intrinsic diffusion")]

        image, tree, _, history = _run_fixed(
            noisy, common_tree.copy(), full, row,
            alpha_scale=1.0, state_dependent=True,
            outer_iterations=args.outer_iterations,
            update_tol=args.update_tol, energy_tol=args.energy_tol,
        )
        local.append(("Fixed selected tree", np.clip(image,0,1), tree, history,
                      "same AMDI-selected tree; state-dependent intrinsic diffusion retained"))

        image, tree, _, history = _run_fixed(
            noisy, common_tree.copy(), full, row,
            alpha_scale=0.0, state_dependent=True,
            outer_iterations=args.outer_iterations,
            update_tol=args.update_tol, energy_tol=args.energy_tol,
        )
        local.append(("No intrinsic diffusion", np.clip(image,0,1), tree, history,
                      "same tree and sparsity term; alpha=0"))

        image, tree, _, history = _run_fixed(
            noisy, common_tree.copy(), full, row,
            alpha_scale=1.0, state_dependent=False,
            outer_iterations=args.outer_iterations,
            update_tol=args.update_tol, energy_tol=args.energy_tol,
        )
        local.append(("State-independent weights", np.clip(image,0,1), tree, history,
                      "same tree and alpha; coefficient dependence removed from graph weights"))

        if representative_images is None:
            representative_images = [("Truth", truth), ("Noisy", noisy)] + [(a,b) for a,b,_,_,_ in local]

        for name, image, tree, history, description in local:
            metrics = all_metrics(image, truth, active=tree.basis_size(), full=n*n)
            result = {
                "noise_seed": seed,
                "variant": name,
                "description": description,
                **metrics,
                "basis_size": tree.basis_size(),
                "common_tree_C_rel": target_complexity,
                "complexity_match_error": abs(float(metrics["C_rel"]) - target_complexity),
                **_history_summary(history),
            }
            rows.append(result)
            print(seed, name, {k: result[k] for k in ("RMSE", "SSIM", "C_rel", "iterations")})

    write_csv(out / "matched_complexity_ablation_runs.csv", rows)

    summary_rows = []
    for name, _ in variant_specs:
        rr = [r for r in rows if r["variant"] == name]
        rm, rm_sd = _mean_std(rr, "RMSE")
        ss, ss_sd = _mean_std(rr, "SSIM")
        cc, cc_sd = _mean_std(rr, "C_rel")
        summary_rows.append({
            "variant": name,
            "n_noise_realizations": len(rr),
            "RMSE_mean": rm, "RMSE_std": rm_sd,
            "SSIM_mean": ss, "SSIM_std": ss_sd,
            "C_rel_mean": cc, "C_rel_std": cc_sd,
            "max_within_seed_complexity_match_error": float(max(r["complexity_match_error"] for r in rr)),
            "all_energy_monotone": bool(all(r["energy_monotone_all_steps"] for r in rr)),
            "minimum_safeguard_accept_rate": float(min(r["safeguard_accept_rate"] for r in rr)),
        })
    write_csv(out / "matched_complexity_ablation_summary.csv", summary_rows)
    write_json(out / "matched_complexity_ablation_assessment.json", {
        "operating_point": args.operating_point,
        "noise_seeds": seeds,
        "all_comparisons_matched_within_seed": bool(max(r["complexity_match_error"] for r in rows) < 1.0e-15),
        "all_runs_energy_monotone": bool(all(r["energy_monotone_all_steps"] for r in rows)),
        "interpretation": "Operator/coefficient ablations use the identical AMDI-selected tree within each noise realization; report what the data show rather than assuming every term improves RMSE/SSIM.",
    })

    if representative_images is not None:
        fig, axes = plt.subplots(1, len(representative_images), figsize=(17.0, 3.4))
        for ax, (_, image) in zip(axes, representative_images):
            show_normalized_image(ax, image, cmap="gray", vmin=0, vmax=1)
        add_panel_labels(axes)
        fig.tight_layout()
        save_publication_figure(fig, out, "matched_complexity_ablation")

    labels = [r["variant"] for r in summary_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].bar(x, [r["RMSE_mean"] for r in summary_rows], yerr=[r["RMSE_std"] for r in summary_rows], capsize=3)
    axes[0].set_ylabel("RMSE [-]")
    axes[1].bar(x, [r["SSIM_mean"] for r in summary_rows], yerr=[r["SSIM_std"] for r in summary_rows], capsize=3)
    axes[1].set_ylabel("SSIM [-]")
    for ax in axes:
        ax.set_xlabel("Variant [-]")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8); ax.grid(axis="y", alpha=0.2)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "matched_complexity_metrics")

    print("\nMatched-complexity summary:")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    main()
