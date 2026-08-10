#!/usr/bin/env python3
"""Robustness and convergence diagnostics for AMDI.

The experiment perturbs one numerical/regularization parameter at a time,
changes the initial-tree threshold, and repeats the calculation across several
noise realizations.  No claim of robustness is hard-coded: the generated CSV
and JSON report the observed variation, monotonicity, safeguard behaviour and
convergence indicators.
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
from amdi.synthetic import add_gaussian_noise
from amdi.validation import build_truth, run_amdi_from_row


def _final_diagnostics(history: list[dict], max_iterations: int, update_tol: float, energy_tol: float) -> dict:
    tail = history[-1]
    steps = history[1:]
    return {
        "iterations": int(tail["iteration"]),
        "stopped_before_max": int(tail["iteration"]) < max_iterations,
        "final_relative_update": float(tail.get("relative_update", np.nan)),
        "final_relative_energy_change": float(tail.get("relative_energy_change", np.nan)),
        "convergence_tolerances_met_at_final_step": bool(
            float(tail.get("relative_update", np.inf)) <= update_tol
            and float(tail.get("relative_energy_change", np.inf)) <= energy_tol
        ),
        "energy_monotone_all_steps": bool(all(bool(h.get("energy_monotone", True)) for h in steps)),
        "safeguard_accept_rate": float(np.mean([bool(h.get("safeguard_accepted", True)) for h in steps])) if steps else 1.0,
        "mean_safeguard_factor": float(np.mean([float(h.get("safeguard_factor", 1.0)) for h in steps])) if steps else 1.0,
        "total_tree_distance": int(sum(int(h.get("tree_distance", 0)) for h in steps)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default=str(ROOT / "results" / "08_amdi_parameter_sweep"))
    parser.add_argument("--operating-point", choices=["best_rmse", "best_ssim", "best_under_10pct_complexity"], default="best_under_10pct_complexity")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--update-tol", type=float, default=2.0e-5)
    parser.add_argument("--energy-tol", type=float, default=2.0e-7)
    parser.add_argument("--noise-seeds", default="11,17,23,31,43")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    best = json.loads((sweep_dir / "best_parameters.json").read_text(encoding="utf-8"))
    meta = json.loads((sweep_dir / "sweep_metadata.json").read_text(encoding="utf-8"))
    row = best[args.operating_point]
    n = int(meta["n"])
    sigma = float(meta["noise_sigma"])
    calibration_seed = int(meta["noise_seed"])
    truth = build_truth(meta["image"], n)
    out = ensure_dir(ROOT / "results" / "12_robustness_convergence")

    # One-at-a-time perturbations around the selected operating point.
    cases = [{"case": "base", "parameter": "base", "scale": 1.0}]
    for parameter in ("initial_threshold", "h", "alpha", "beta", "tau"):
        for scale in (0.5, 2.0):
            cases.append({"case": f"{parameter}_x{scale:g}", "parameter": parameter, "scale": scale})

    rows = []
    histories = {}
    for case in cases:
        kwargs = dict(
            initial_threshold_scale=case["scale"] if case["parameter"] == "initial_threshold" else 1.0,
            h_scale=case["scale"] if case["parameter"] == "h" else 1.0,
            alpha_scale=case["scale"] if case["parameter"] == "alpha" else 1.0,
            beta_scale=case["scale"] if case["parameter"] == "beta" else 1.0,
            tau_scale=case["scale"] if case["parameter"] == "tau" else 1.0,
        )
        noisy = add_gaussian_noise(truth, sigma, seed=calibration_seed)
        image, tree, _, history, _, _ = run_amdi_from_row(
            noisy, row, args.max_iterations,
            stop_on_convergence=True,
            update_tol=args.update_tol,
            energy_tol=args.energy_tol,
            patience=2,
            **kwargs,
        )
        result = {
            "group": "parameter_perturbation",
            **case,
            "noise_seed": calibration_seed,
            **all_metrics(image, truth, active=tree.basis_size(), full=n*n),
            **_final_diagnostics(history, args.max_iterations, args.update_tol, args.energy_tol),
        }
        rows.append(result)
        histories[case["case"]] = history
        print(case["case"], {k: result[k] for k in ("RMSE", "SSIM", "C_rel", "iterations", "energy_monotone_all_steps")})

    # Independent noise realizations with all AMDI parameters fixed.
    noise_seeds = [int(v) for v in args.noise_seeds.split(",") if v.strip()]
    for seed in noise_seeds:
        noisy = add_gaussian_noise(truth, sigma, seed=seed)
        image, tree, _, history, _, _ = run_amdi_from_row(
            noisy, row, args.max_iterations,
            stop_on_convergence=True,
            update_tol=args.update_tol,
            energy_tol=args.energy_tol,
            patience=2,
        )
        result = {
            "group": "noise_realization",
            "case": f"noise_seed_{seed}",
            "parameter": "noise_seed",
            "scale": 1.0,
            "noise_seed": seed,
            **all_metrics(image, truth, active=tree.basis_size(), full=n*n),
            **_final_diagnostics(history, args.max_iterations, args.update_tol, args.energy_tol),
        }
        rows.append(result)

    write_csv(out / "robustness_runs.csv", rows)

    param_rows = [r for r in rows if r["group"] == "parameter_perturbation"]
    seed_rows = [r for r in rows if r["group"] == "noise_realization"]
    assessment = {
        "operating_point": args.operating_point,
        "n_parameter_runs": len(param_rows),
        "n_noise_realizations": len(seed_rows),
        "all_runs_energy_monotone": bool(all(r["energy_monotone_all_steps"] for r in rows)),
        "minimum_safeguard_accept_rate": float(min(r["safeguard_accept_rate"] for r in rows)),
        "mean_safeguard_accept_rate": float(np.mean([r["safeguard_accept_rate"] for r in rows])),
        "parameter_RMSE_range": [float(min(r["RMSE"] for r in param_rows)), float(max(r["RMSE"] for r in param_rows))],
        "parameter_SSIM_range": [float(min(r["SSIM"] for r in param_rows)), float(max(r["SSIM"] for r in param_rows))],
        "parameter_C_rel_range": [float(min(r["C_rel"] for r in param_rows)), float(max(r["C_rel"] for r in param_rows))],
        "noise_RMSE_mean": float(np.mean([r["RMSE"] for r in seed_rows])),
        "noise_RMSE_std": float(np.std([r["RMSE"] for r in seed_rows], ddof=1)) if len(seed_rows) > 1 else 0.0,
        "noise_SSIM_mean": float(np.mean([r["SSIM"] for r in seed_rows])),
        "noise_SSIM_std": float(np.std([r["SSIM"] for r in seed_rows], ddof=1)) if len(seed_rows) > 1 else 0.0,
        "noise_C_rel_mean": float(np.mean([r["C_rel"] for r in seed_rows])),
        "noise_C_rel_std": float(np.std([r["C_rel"] for r in seed_rows], ddof=1)) if len(seed_rows) > 1 else 0.0,
        "fraction_stopped_before_max": float(np.mean([r["stopped_before_max"] for r in rows])),
        "fraction_final_tolerances_met": float(np.mean([r["convergence_tolerances_met_at_final_step"] for r in rows])),
    }
    write_json(out / "robustness_assessment.json", assessment)

    # Convergence trajectories for the central parameter-perturbation cases.
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for label, history in histories.items():
        it = np.asarray([h["iteration"] for h in history])
        e = np.asarray([h["energy"] for h in history], float)
        rel = np.asarray([h.get("relative_update", 0.0) for h in history], float)
        comp = np.asarray([h["basis_size"]/(n*n) for h in history], float)
        axes[0].plot(it, e / max(abs(e[0]), 1e-15), alpha=0.72, label=label)
        axes[1].semilogy(it[1:], np.maximum(rel[1:], 1e-16), alpha=0.72)
        axes[2].plot(it, comp, alpha=0.72)
    axes[0].set_xlabel("Iteration [-]"); axes[0].set_ylabel(r"$\mathcal{E}^n/|\mathcal{E}^0|$ [-]")
    axes[1].set_xlabel("Iteration [-]"); axes[1].set_ylabel("Relative update [-]")
    axes[2].set_xlabel("Iteration [-]"); axes[2].set_ylabel(r"$\mathcal{C}_{\rm rel}$ [-]")
    for ax in axes: ax.grid(alpha=0.2)
    axes[0].legend(fontsize=6, ncol=2)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "robustness_convergence")

    # Final metric sensitivity.
    labels = [r["case"] for r in param_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    axes[0].plot(x, [r["RMSE"] for r in param_rows], marker="o")
    axes[0].set_ylabel("RMSE [-]")
    axes[1].plot(x, [r["SSIM"] for r in param_rows], marker="o")
    axes[1].set_ylabel("SSIM [-]")
    axes[2].plot(x, [r["C_rel"] for r in param_rows], marker="o")
    axes[2].set_ylabel(r"$\mathcal{C}_{\rm rel}$ [-]")
    for ax in axes:
        ax.set_xlabel("Perturbation case [-]")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7); ax.grid(alpha=0.2)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "parameter_sensitivity")

    print("\nRobustness assessment:")
    for k, v in assessment.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
