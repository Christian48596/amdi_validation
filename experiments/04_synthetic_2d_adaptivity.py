#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.haar import (
    adaptive_tree_from_detail_threshold,
    full_coefficients,
    project_to_tree,
    reconstruct,
    refinement_level_map,
)
from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.metrics import all_metrics
from amdi.synthetic import four_region_image


def main():
    out = ensure_dir(ROOT / "results" / "04_synthetic_2d_adaptivity")
    n = 128
    truth = four_region_image(n)
    max_level = 7
    full = full_coefficients(truth, max_level=max_level)
    tree = adaptive_tree_from_detail_threshold(full, dim=2, max_level=max_level, threshold=0.006, min_level=2)
    coeffs = project_to_tree(truth, tree)
    recon = reconstruct(coeffs, tree, truth.shape)
    levels = refinement_level_map(tree, truth.shape)

    metrics = all_metrics(recon, truth, active=tree.basis_size(), full=n * n)
    metrics.update({"basis_size": tree.basis_size(), "n_leaves": len(tree.leaves()), "max_level_used": int(levels.max())})
    write_json(out / "metrics.json", metrics)

    regions = {
        "constant": (slice(0, n // 2), slice(0, n // 2)),
        "gradient": (slice(0, n // 2), slice(n // 2, n)),
        "edge": (slice(n // 2, n), slice(0, n // 2)),
        "texture": (slice(n // 2, n), slice(n // 2, n)),
    }
    rows = []
    for name, sl in regions.items():
        rows.append({"region": name, "mean_refinement_level": float(levels[sl].mean()), "max_refinement_level": int(levels[sl].max())})
    write_csv(out / "regional_refinement.csv", rows)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    show_normalized_image(axes[0], truth, cmap="gray", vmin=0, vmax=1)
    show_normalized_image(axes[1], recon, cmap="gray", vmin=0, vmax=1)
    im = show_normalized_image(axes[2], levels, vmin=0, vmax=max_level)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label("Refinement level [-]")
    err_im = show_normalized_image(axes[3], np.abs(recon - truth), cmap="magma")
    cb_err = fig.colorbar(err_im, ax=axes[3], fraction=0.046, pad=0.04)
    cb_err.set_label("Absolute error [-]")
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "synthetic_adaptivity")
    print(metrics)


if __name__ == "__main__":
    main()
