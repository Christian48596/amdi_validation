#!/usr/bin/env python3
"""Optional VAMPyR adaptive-grid cross-check.

This script is deliberately separate from the AMDI coefficient solver because
current VAMPyR Python bindings expose node geometry/norms but not raw node
coefficient arrays.
"""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.vampyr_adapter import adaptive_project, end_node_table, tree_summary


def f2(r):
    x, y = r[0], r[1]
    val = 0.2 + 0.5 * x
    val += 0.5 * np.exp(-180.0 * ((x - 0.3) ** 2 + (y - 0.7) ** 2))
    if x > 0.62 and y > 0.25:
        val += 0.25
    return float(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=float, default=1.0e-3)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()
    out = ensure_dir(ROOT / "results" / "07_vampyr_projection_check")
    try:
        tree = adaptive_project(f2, dim=2, order=5, precision=args.precision, max_depth=args.max_depth)
    except RuntimeError as exc:
        print(exc)
        print("Skipping optional VAMPyR test.")
        return

    summary = tree_summary(tree)
    rows = end_node_table(tree)
    write_json(out / "tree_summary.json", summary)
    write_csv(out / "end_nodes.csv", rows)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for row in rows:
        lo = row["lower"]; hi = row["upper"]
        rect = plt.Rectangle((lo[0], lo[1]), hi[0]-lo[0], hi[1]-lo[1], fill=False, lw=0.45)
        ax.add_patch(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [-]"); ax.set_ylabel(r"$y$ [-]")
    add_panel_label(ax, "(a)")
    fig.tight_layout()
    save_publication_figure(fig, out, "vampyr_adaptive_grid")
    print(summary)


if __name__ == "__main__":
    main()
