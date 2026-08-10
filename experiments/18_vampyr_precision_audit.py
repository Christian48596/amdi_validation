#!/usr/bin/env python3
"""Audit the completed VAMPyR precision/complexity convergence experiment.

Reads the CSV produced by experiment 10 and checks the numerical statements
we may want to use in the manuscript: tighter requested precision should not
systematically reduce adaptive complexity, the L2 distance to the tighter
reference should decrease, and the depth cap should not dominate the sequence.
The script reports the raw rows, monotonicity flags, relative reductions and an
empirical log-log error-versus-complexity slope when enough finite data exist.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.io_utils import ensure_dir, write_csv, write_json


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_bool(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "results" / "10_vampyr_precision_convergence" / "vampyr_precision_convergence.csv"))
    parser.add_argument("--metadata", default=str(ROOT / "results" / "10_vampyr_precision_convergence" / "vampyr_precision_metadata.json"))
    args = parser.parse_args()

    src = Path(args.csv)
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}. Run experiment 10 first.")
    rows_raw = _read(src)
    if len(rows_raw) < 2:
        raise RuntimeError("Need at least two VAMPyR precision rows to audit convergence.")

    rows = []
    for r in rows_raw:
        rows.append({
            "precision": float(r["precision"]),
            "L2_distance_to_reference": float(r["L2_distance_to_reference"]),
            "depth_hit_cap": _to_bool(r.get("depth_hit_cap", False)),
            "n_nodes": int(float(r["n_nodes"])),
            "n_end_nodes": int(float(r["n_end_nodes"])),
            "n_root_nodes": int(float(r.get("n_root_nodes", 1))),
            "depth": int(float(r["depth"])),
            "root_scale": int(float(r.get("root_scale", 0))),
            "norm": float(r.get("norm", np.nan)),
        })

    # Sort from loose to tight requested precision (decreasing numerical epsilon).
    rows = sorted(rows, key=lambda r: r["precision"], reverse=True)
    p = np.asarray([r["precision"] for r in rows], float)
    nend = np.asarray([r["n_end_nodes"] for r in rows], float)
    err = np.asarray([r["L2_distance_to_reference"] for r in rows], float)
    finite = np.isfinite(err)

    complexity_nondecreasing = bool(np.all(np.diff(nend) >= 0))
    finite_err = err[finite]
    error_nonincreasing = bool(len(finite_err) >= 2 and np.all(np.diff(finite_err) <= 1.0e-14))
    any_depth_cap = bool(any(r["depth_hit_cap"] for r in rows))
    all_errors_finite = bool(np.all(finite))

    positive = finite & (err > 0.0) & (nend > 0.0)
    empirical_slope = float("nan")
    if np.count_nonzero(positive) >= 3 and np.unique(nend[positive]).size >= 2:
        coeff = np.polyfit(np.log(nend[positive]), np.log(err[positive]), 1)
        empirical_slope = float(coeff[0])

    error_reduction_factor = float("nan")
    if np.count_nonzero(finite) >= 2:
        e0 = err[np.flatnonzero(finite)[0]]
        e1 = err[np.flatnonzero(finite)[-1]]
        if e1 > 0:
            error_reduction_factor = float(e0 / e1)
        elif e0 > 0 and e1 == 0:
            error_reduction_factor = float("inf")

    complexity_growth_factor = float(nend[-1] / max(nend[0], 1.0))

    audit = {
        "n_precision_levels": len(rows),
        "loose_precision": float(p[0]),
        "tight_precision": float(p[-1]),
        "complexity_nondecreasing_as_precision_tightens": complexity_nondecreasing,
        "L2_error_nonincreasing_as_precision_tightens": error_nonincreasing,
        "all_L2_errors_finite": all_errors_finite,
        "any_projection_hit_depth_cap": any_depth_cap,
        "end_node_growth_factor_loose_to_tight": complexity_growth_factor,
        "L2_error_reduction_factor_loose_to_tight": error_reduction_factor,
        "empirical_log_error_vs_log_end_nodes_slope": empirical_slope,
        "publication_ready_monotone_precision_trend": bool(
            complexity_nondecreasing and error_nonincreasing and all_errors_finite and not any_depth_cap
        ),
        "interpretation_note": "A negative empirical slope indicates decreasing reference error as adaptive complexity increases. The slope is descriptive, not a proved convergence order.",
    }

    meta = {}
    mpath = Path(args.metadata)
    if mpath.exists():
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        audit["metadata"] = meta

    out = ensure_dir(ROOT / "results" / "18_vampyr_precision_audit")
    write_csv(out / "vampyr_precision_rows_audited.csv", rows)
    write_json(out / "vampyr_precision_audit.json", audit)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    axes[0].loglog(p, nend, marker="o")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Requested VAMPyR precision [-]")
    axes[0].set_ylabel("End nodes [count]")
    axes[0].grid(alpha=0.2)

    good = finite & (err > 0)
    if np.any(good):
        axes[1].loglog(nend[good], err[good], marker="o")
        axes[1].set_xlabel("End nodes [count]")
        axes[1].set_ylabel(r"$L^2$ distance to reference [-]")
        axes[1].grid(alpha=0.2)
    else:
        axes[1].axis("off")
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "vampyr_precision_audit")

    print("VAMPyR precision rows (loose -> tight):")
    for r in rows:
        print(r)
    print("\nAudit:")
    for k, v in audit.items():
        if k != "metadata":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
