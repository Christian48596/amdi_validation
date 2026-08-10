#!/usr/bin/env python3
"""Memory-safe VAMPyR precision/complexity convergence in 2D.

This experiment deliberately uses a smooth multiscale target.  A true jump
combined with an extremely tight tolerance can force adaptive refinement to
very deep levels and create millions of 2D nodes.  That behavior is useful in
its own right, but it is not a practical precision-convergence benchmark.

The MRA therefore has an explicit max_depth and the reference projection is
constructed at a tighter (but finite) precision on the same capped MRA.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.vampyr_adapter import adaptive_project_on_mra, l2_distance, make_mra, tree_summary


def reference_function(r):
    x, y = float(r[0]), float(r[1])
    value = 0.20 + 0.50 * x
    value += 0.50 * np.exp(-180.0 * ((x - 0.30) ** 2 + (y - 0.70) ** 2))

    # Smooth edge-like transition rather than a discontinuous Heaviside jump.
    # This still tests localized refinement but prevents pathological node
    # growth at very tight projection tolerances.
    edge = 0.5 * (1.0 + np.tanh((x - 0.62) / 0.020))
    gate_y = 0.5 * (1.0 + np.tanh((y - 0.25) / 0.025))
    value += 0.25 * edge * gate_y

    # Smoothly localized oscillatory component.
    window = np.exp(-55.0 * ((x - 0.75) ** 2 + (y - 0.18) ** 2))
    value += 0.06 * window * np.sin(30.0 * np.pi * x) * np.sin(22.0 * np.pi * y)
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--reference-precision", type=float, default=1.0e-5)
    parser.add_argument("--max-end-nodes", type=int, default=250000,
                        help="Stop before tighter tests if a projection exceeds this size.")
    args = parser.parse_args()

    out = ensure_dir(ROOT / "results" / "10_vampyr_precision_convergence")
    precisions = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5]

    try:
        vp, mra = make_mra(2, order=args.order, max_depth=args.max_depth)
        print(f"Building capped VAMPyR reference: precision={args.reference_precision:g}, max_depth={args.max_depth}")
        reference = adaptive_project_on_mra(reference_function, vp, mra, args.reference_precision)
    except RuntimeError as exc:
        print(exc)
        print("Skipping optional VAMPyR precision experiment.")
        return

    reference_summary = tree_summary(reference)
    print("Reference summary:", reference_summary)

    rows = []
    for prec in precisions:
        print(f"Projecting at precision={prec:g} ...", flush=True)
        tree = adaptive_project_on_mra(reference_function, vp, mra, prec)
        summary = tree_summary(tree)
        try:
            error = l2_distance(tree, reference, vp)
        except Exception as exc:
            error = float("nan")
            print(f"Warning: could not evaluate L2 tree distance at precision={prec:g}: {exc}")
        row = {
            "precision": prec,
            "L2_distance_to_reference": error,
            "depth_hit_cap": bool(summary["depth"] >= args.max_depth),
            **summary,
        }
        rows.append(row)
        print(row)

        if summary["n_end_nodes"] > args.max_end_nodes:
            print(
                f"Stopping precision sequence safely: {summary['n_end_nodes']} end nodes "
                f"exceed --max-end-nodes={args.max_end_nodes}."
            )
            break

    write_csv(out / "vampyr_precision_convergence.csv", rows)
    write_json(out / "vampyr_precision_metadata.json", {
        "order": args.order,
        "max_depth": args.max_depth,
        "reference_precision": args.reference_precision,
        "reference_summary": reference_summary,
        "error_definition": "L2 distance to a tighter VAMPyR projection on the same depth-capped MRA",
        "target": "smooth multiscale 2D function (ramp + Gaussian + tanh edge + localized oscillation)",
        "safety_note": "max_depth prevents pathological 2D node growth at very tight tolerances",
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    p = np.asarray([r["precision"] for r in rows], float)
    nodes = np.asarray([r["n_end_nodes"] for r in rows], float)
    err = np.asarray([r["L2_distance_to_reference"] for r in rows], float)

    axes[0].loglog(p, nodes, marker="o")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Requested VAMPyR precision [-]")
    axes[0].set_ylabel("End nodes [count]")
    axes[0].grid(alpha=0.2)

    finite = np.isfinite(err) & (err > 0.0)
    if np.any(finite):
        axes[1].loglog(nodes[finite], err[finite], marker="o")
        axes[1].set_xlabel("End nodes [count]")
        axes[1].set_ylabel(r"$L^2$ distance to reference [-]")
        axes[1].grid(alpha=0.2)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "L2 tree distance unavailable\nin this VAMPyR build", ha="center", va="center")

    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "vampyr_precision_convergence")


if __name__ == "__main__":
    main()
