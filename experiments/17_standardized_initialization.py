#!/usr/bin/env python3
"""Standardize AMDI initialization by an explicit initial complexity budget.

The original robustness test perturbed a raw coefficient threshold and showed
that the final accuracy--complexity tradeoff can depend appreciably on that
threshold.  A raw threshold is scale dependent and is therefore not an ideal
user-facing initialization parameter.

This experiment replaces it by a deterministic, interpretable budget:
    C_rel^0 = N_{Lambda^0} / N_full.
For each requested budget, a bisection in the Haar detail threshold constructs
an admissible initial tree whose basis size is as close as possible to the
budget without exceeding it (unless the mandatory min-level tree already does).
No ground-truth image metric is used to choose the threshold.
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
from amdi.haar import adaptive_tree_from_detail_threshold, full_coefficients
from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.metrics import all_metrics
from amdi.solver import adaptive_frozen_denoise
from amdi.synthetic import add_gaussian_noise
from amdi.validation import build_truth


def _subtree_detail_norms(full, dim, max_level):
    """Accumulate unresolved wavelet-detail energy for every dyadic cell."""
    energy = {}
    for idx, value in full.items():
        if idx.kind != "wavelet":
            continue
        lev = int(idx.level)
        tr = tuple(int(v) for v in idx.translation)
        vv = float(value) ** 2
        for anc_level in range(0, lev + 1):
            factor = 2 ** (lev - anc_level)
            anc_tr = tuple(v // factor for v in tr)
            cell = (anc_level, anc_tr)
            energy[cell] = energy.get(cell, 0.0) + vv
    return {cell: float(np.sqrt(val)) for cell, val in energy.items()}


def _tree_from_cached_threshold(norms, dim, max_level, threshold, min_level=2):
    from amdi.haar import AdaptiveHaarTree
    tree = AdaptiveHaarTree(dim=dim, max_level=max_level)
    queue = [tree.root(dim)]
    while queue:
        cell = queue.pop(0)
        level, _ = cell
        if level >= max_level:
            continue
        must_refine = level < min_level
        if must_refine or norms.get(cell, 0.0) >= float(threshold):
            tree.refined.add(cell)
            queue.extend(tree.children(cell))
    tree._enforce_ancestry()
    return tree


def _tree_for_complexity_budget(full, dim, max_level, target_c_rel, full_size, min_level=2):
    """Return a deterministic threshold/tree pair near a requested complexity budget.

    The search is over the finite set of subtree-detail norms at which the
    admissible Haar tree can actually change.  This is both exact for this
    threshold rule and much faster than repeated floating-point bisection.
    """
    target_basis = max(1, int(np.floor(float(target_c_rel) * full_size)))
    norms = _subtree_detail_norms(full, dim, max_level)
    candidates = sorted(set([0.0, *norms.values()]))
    if candidates:
        candidates.append(float(candidates[-1]) * (1.0 + 1.0e-12) + 1.0e-15)

    cache = {}
    def evaluate(i):
        if i not in cache:
            th = candidates[i]
            tr = _tree_from_cached_threshold(norms, dim, max_level, th, min_level=min_level)
            cache[i] = (th, tr, tr.basis_size())
        return cache[i]

    # Tree size is nonincreasing with threshold. Binary-search the first
    # threshold whose tree does not exceed the requested basis budget.
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        _, _, size = evaluate(mid)
        if size > target_basis:
            lo = mid + 1
        else:
            hi = mid

    # Inspect the crossing and immediate neighbours because tree size is discrete.
    trial_indices = sorted(set(i for i in (lo - 2, lo - 1, lo, lo + 1, lo + 2) if 0 <= i < len(candidates)))
    trials = [evaluate(i) for i in trial_indices]
    feasible = [z for z in trials if z[2] <= target_basis]
    if feasible:
        # Maximize utilized basis under the budget; for ties prefer the larger
        # threshold (the more conservative deterministic initialization).
        th, tree, _ = sorted(feasible, key=lambda z: (-z[2], -z[0]))[0]
    else:
        th, tree, _ = min(trials, key=lambda z: abs(z[2] - target_basis))
    return th, tree, target_basis


def _run(noisy, full, tree, row, max_iterations, update_tol, energy_tol):
    ep = EnergyParameters(
        alpha=float(row["alpha"]),
        beta=float(row["beta"]),
        tau=float(row["tau"]),
        smooth_l1=False,
    )
    gp = GraphParameters(
        sigma_c=float(row["sigma_c"]),
        state_dependent=True,
        refinement_decay=float(row["refinement_decay"]),
    )
    return adaptive_frozen_denoise(
        noisy, tree, full, ep, gp,
        h=float(row["h"]),
        outer_iterations=max_iterations,
        adapt=True,
        zeta=float(row["zeta"]),
        refine_fraction=float(row["refine_fraction"]),
        coarsen_fraction=float(row["coarsen_fraction"]),
        stop_on_convergence=True,
        update_tol=update_tol,
        energy_tol=energy_tol,
        patience=2,
    )


def _diagnostics(history):
    steps = history[1:]
    return {
        "iterations": int(history[-1]["iteration"]),
        "energy_monotone_all_steps": bool(all(bool(h.get("energy_monotone", True)) for h in steps)),
        "safeguard_accept_rate": float(np.mean([bool(h.get("safeguard_accepted", True)) for h in steps])) if steps else 1.0,
        "total_tree_distance": int(sum(int(h.get("tree_distance", 0)) for h in steps)),
        "final_relative_update": float(history[-1].get("relative_update", np.nan)),
        "final_relative_energy_change": float(history[-1].get("relative_energy_change", np.nan)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default=str(ROOT / "results" / "08_amdi_parameter_sweep"))
    parser.add_argument("--operating-point", choices=["best_rmse", "best_ssim", "best_under_10pct_complexity"],
                        default="best_under_10pct_complexity")
    parser.add_argument("--initial-budgets", default="0.10,0.12,0.14",
                        help="comma-separated target initial C_rel values")
    parser.add_argument("--noise-seeds", default="11,17,23,31,43")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--update-tol", type=float, default=2.0e-5)
    parser.add_argument("--energy-tol", type=float, default=2.0e-7)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    best = json.loads((sweep_dir / "best_parameters.json").read_text(encoding="utf-8"))
    meta = json.loads((sweep_dir / "sweep_metadata.json").read_text(encoding="utf-8"))
    row = best[args.operating_point]
    n = int(meta["n"])
    sigma = float(meta["noise_sigma"])
    max_level = int(np.log2(n))
    truth = build_truth(meta["image"], n)
    budgets = [float(v) for v in args.initial_budgets.split(",") if v.strip()]
    seeds = [int(v) for v in args.noise_seeds.split(",") if v.strip()]
    out = ensure_dir(ROOT / "results" / "17_standardized_initialization")

    rows = []
    for seed in seeds:
        noisy = add_gaussian_noise(truth, sigma, seed=seed)
        full = full_coefficients(noisy, max_level=max_level)
        for budget in budgets:
            threshold, initial, target_basis = _tree_for_complexity_budget(
                full, 2, max_level, budget, n*n, min_level=2
            )
            initial_c = initial.basis_size() / float(n*n)
            image, tree, _, history = _run(
                noisy, full, initial, row,
                args.max_iterations, args.update_tol, args.energy_tol,
            )
            image = np.clip(image, 0.0, 1.0)
            result = {
                "noise_seed": seed,
                "target_initial_C_rel": budget,
                "target_initial_basis_size": target_basis,
                "automatic_threshold": threshold,
                "initial_basis_size": initial.basis_size(),
                "initial_C_rel": initial_c,
                **all_metrics(image, truth, active=tree.basis_size(), full=n*n),
                "final_basis_size": tree.basis_size(),
                **_diagnostics(history),
            }
            rows.append(result)
            print(
                f"seed={seed:3d} target_C0={budget:.3f} actual_C0={initial_c:.4f} "
                f"-> RMSE={result['RMSE']:.5f} SSIM={result['SSIM']:.4f} C_final={result['C_rel']:.4f} "
                f"iters={result['iterations']}"
            )

    write_csv(out / "standardized_initialization_runs.csv", rows)

    summary_rows = []
    for budget in budgets:
        rr = [r for r in rows if abs(r["target_initial_C_rel"] - budget) < 1e-15]
        summary_rows.append({
            "target_initial_C_rel": budget,
            "n_noise_realizations": len(rr),
            "initial_C_rel_mean": float(np.mean([r["initial_C_rel"] for r in rr])),
            "initial_C_rel_std": float(np.std([r["initial_C_rel"] for r in rr], ddof=1)) if len(rr)>1 else 0.0,
            "RMSE_mean": float(np.mean([r["RMSE"] for r in rr])),
            "RMSE_std": float(np.std([r["RMSE"] for r in rr], ddof=1)) if len(rr)>1 else 0.0,
            "SSIM_mean": float(np.mean([r["SSIM"] for r in rr])),
            "SSIM_std": float(np.std([r["SSIM"] for r in rr], ddof=1)) if len(rr)>1 else 0.0,
            "final_C_rel_mean": float(np.mean([r["C_rel"] for r in rr])),
            "final_C_rel_std": float(np.std([r["C_rel"] for r in rr], ddof=1)) if len(rr)>1 else 0.0,
            "all_energy_monotone": bool(all(r["energy_monotone_all_steps"] for r in rr)),
            "minimum_safeguard_accept_rate": float(min(r["safeguard_accept_rate"] for r in rr)),
        })
    write_csv(out / "standardized_initialization_summary.csv", summary_rows)

    write_json(out / "standardized_initialization_protocol.json", {
        "operating_point": args.operating_point,
        "initialization_rule": "automatic Haar detail threshold chosen by bisection to meet a prescribed initial relative-complexity budget",
        "uses_ground_truth_for_initialization": False,
        "tested_initial_complexity_budgets": budgets,
        "noise_seeds": seeds,
        "recommended_default_budget_to_assess": 0.12,
        "note": "The run determines whether replacing a raw coefficient threshold by C_rel^0 reduces scale-dependent initialization sensitivity; no robustness claim is hard-coded.",
    })

    x = np.asarray(budgets, float)
    rm = np.asarray([r["RMSE_mean"] for r in summary_rows])
    rm_sd = np.asarray([r["RMSE_std"] for r in summary_rows])
    ss = np.asarray([r["SSIM_mean"] for r in summary_rows])
    ss_sd = np.asarray([r["SSIM_std"] for r in summary_rows])
    cf = np.asarray([r["final_C_rel_mean"] for r in summary_rows])
    cf_sd = np.asarray([r["final_C_rel_std"] for r in summary_rows])

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    axes[0].errorbar(x, rm, yerr=rm_sd, marker="o", capsize=3)
    axes[0].set_ylabel("RMSE [-]")
    axes[1].errorbar(x, ss, yerr=ss_sd, marker="o", capsize=3)
    axes[1].set_ylabel("SSIM [-]")
    axes[2].errorbar(x, cf, yerr=cf_sd, marker="o", capsize=3)
    axes[2].set_ylabel(r"Final $\mathcal{C}_{\rm rel}$ [-]")
    for ax in axes:
        ax.set_xlabel(r"Prescribed initial $\mathcal{C}_{\rm rel}^{0}$ [-]")
        ax.grid(alpha=0.2)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "standardized_initialization_sensitivity")

    print("\nSummary over noise realizations:")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    main()
