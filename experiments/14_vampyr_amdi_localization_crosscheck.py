#!/usr/bin/env python3
"""Independent VAMPyR/MRCPP localization cross-check for the AMDI test image.

The same four-region target is represented by (i) the self-contained adaptive
Haar backend used for AMDI algebra and (ii) an adaptive VAMPyR ScalingProjector.
The bases and refinement criteria are intentionally different.  The purpose is
therefore qualitative/structural: both representations should allocate more
local resolution to difficult edge/texture regions than to the constant one.
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

from amdi.haar import adaptive_tree_from_detail_threshold, full_coefficients, refinement_level_map
from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.synthetic import four_region_image
from amdi.vampyr_adapter import adaptive_project, end_node_table, tree_summary


def four_region_function(r):
    x, y = float(r[0]), float(r[1])
    if x < 0.5 and y < 0.5:
        value = 0.25
    elif x >= 0.5 and y < 0.5:
        value = 0.15 + 0.7 * (x - 0.5) / 0.5
    elif x < 0.5 and y >= 0.5:
        value = 0.9 if ((x - 0.25)**2 + (y - 0.75)**2) <= 0.12**2 else 0.15
    else:
        value = 0.5 + 0.28 * np.sin(18*np.pi*x) * np.sin(18*np.pi*y)
    return float(np.clip(value, 0.0, 1.0))


def region_name(x, y):
    if x < 0.5 and y < 0.5: return "constant"
    if x >= 0.5 and y < 0.5: return "gradient"
    if x < 0.5 and y >= 0.5: return "edge"
    return "texture"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--haar-threshold", type=float, default=0.006)
    parser.add_argument("--vampyr-order", type=int, default=5)
    parser.add_argument("--vampyr-precision", type=float, default=1.0e-3)
    parser.add_argument("--vampyr-max-depth", type=int, default=8)
    args = parser.parse_args()

    if args.n <= 0 or args.n & (args.n - 1):
        raise ValueError("--n must be a positive power of two")
    out = ensure_dir(ROOT / "results" / "14_vampyr_amdi_localization_crosscheck")

    truth = four_region_image(args.n)
    max_level = int(np.log2(args.n))
    full = full_coefficients(truth, max_level=max_level)
    haar_tree = adaptive_tree_from_detail_threshold(full, 2, max_level, threshold=args.haar_threshold, min_level=2)
    haar_levels = refinement_level_map(haar_tree, truth.shape)

    try:
        vtree = adaptive_project(four_region_function, dim=2, order=args.vampyr_order, precision=args.vampyr_precision, max_depth=args.vampyr_max_depth)
    except RuntimeError as exc:
        print(exc)
        print("Skipping optional VAMPyR cross-check.")
        return
    vrows = end_node_table(vtree)
    write_csv(out / "vampyr_end_nodes.csv", vrows)
    write_json(out / "vampyr_tree_summary.json", tree_summary(vtree))

    regions = {
        "constant": (slice(0, args.n//2), slice(0, args.n//2)),
        "gradient": (slice(0, args.n//2), slice(args.n//2, args.n)),
        "edge": (slice(args.n//2, args.n), slice(0, args.n//2)),
        "texture": (slice(args.n//2, args.n), slice(args.n//2, args.n)),
    }
    regional = []
    for name, sl in regions.items():
        local_v = []
        for row in vrows:
            x, y = row["center"][:2]
            if region_name(x, y) == name:
                lo = np.asarray(row["lower"][:2], float)
                hi = np.asarray(row["upper"][:2], float)
                area = max(float(np.prod(hi-lo)), 1e-300)
                # Effective dyadic level on a unit 2D domain; defined from cell area
                # so it is comparable even if VAMPyR's root scale differs.
                local_v.append(-0.5*np.log2(area))
        regional.append({
            "region": name,
            "haar_mean_level": float(np.mean(haar_levels[sl])),
            "haar_max_level": int(np.max(haar_levels[sl])),
            "vampyr_mean_effective_level": float(np.mean(local_v)) if local_v else float("nan"),
            "vampyr_max_effective_level": float(np.max(local_v)) if local_v else float("nan"),
            "vampyr_end_nodes": len(local_v),
        })
    write_csv(out / "regional_localization_crosscheck.csv", regional)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    im0 = show_normalized_image(axes[0], haar_levels, vmin=0, vmax=max_level)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("Refinement level [-]")

    for row in vrows:
        lo = row["lower"]; hi = row["upper"]
        rect = plt.Rectangle((lo[0], lo[1]), hi[0]-lo[0], hi[1]-lo[1], fill=False, lw=0.35)
        axes[1].add_patch(rect)
    axes[1].set_xlim(0,1); axes[1].set_ylim(0,1); axes[1].set_aspect("equal")
    axes[1].set_xlabel(r"$x$ [-]"); axes[1].set_ylabel(r"$y$ [-]")

    names = [r["region"] for r in regional]
    x = np.arange(len(names)); width=0.38
    hvals = [r["haar_mean_level"] for r in regional]
    vvals = [r["vampyr_mean_effective_level"] for r in regional]
    axes[2].bar(x-width/2, hvals, width, label="AMDI Haar")
    axes[2].bar(x+width/2, vvals, width, label="VAMPyR")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=25, ha="right")
    axes[2].set_xlabel("Region [-]")
    axes[2].set_ylabel("Mean local resolution level [-]")
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.2)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "vampyr_amdi_localization_crosscheck")

    print(tree_summary(vtree))
    for r in regional: print(r)


if __name__ == "__main__":
    main()
