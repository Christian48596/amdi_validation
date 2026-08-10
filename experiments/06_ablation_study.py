#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from amdi.haar import AdaptiveHaarTree, adaptive_tree_from_detail_threshold, full_coefficients
from amdi.io_utils import ensure_dir, write_csv
from amdi.metrics import all_metrics
from amdi.solver import adaptive_frozen_denoise
from amdi.synthetic import four_region_image, add_gaussian_noise


def run_variant(noisy, full, initial, name):
    if name == "AMDI_full":
        ep = EnergyParameters(alpha=0.010, beta=0.0002, tau=1e-7, smooth_l1=False)
        gp = GraphParameters(k_neighbors=10, sigma_c=0.12, state_dependent=True, refinement_decay=0.35)
        adapt = True
        tree = initial
    elif name == "AMDI_fixed":
        ep = EnergyParameters(alpha=0.010, beta=0.0002, tau=0.0, smooth_l1=False)
        gp = GraphParameters(k_neighbors=10, sigma_c=0.12, state_dependent=True, refinement_decay=0.35)
        adapt = False
        tree = AdaptiveHaarTree.uniform(2, 5)
    elif name == "AMDI_no_diffusion":
        ep = EnergyParameters(alpha=0.0, beta=0.0002, tau=1e-7, smooth_l1=False)
        gp = GraphParameters(k_neighbors=10, sigma_c=0.12, state_dependent=True, refinement_decay=0.35)
        adapt = True
        tree = initial
    elif name == "AMDI_linear":
        ep = EnergyParameters(alpha=0.010, beta=0.0002, tau=1e-7, smooth_l1=False)
        gp = GraphParameters(k_neighbors=10, sigma_c=0.12, state_dependent=False, refinement_decay=0.35)
        adapt = True
        tree = initial
    else:
        raise ValueError(name)
    return adaptive_frozen_denoise(noisy, tree, full, ep, gp, h=0.6, outer_iterations=6, adapt=adapt, zeta=1e-6)


def main():
    out = ensure_dir(ROOT / "results" / "06_ablation_study")
    n = 64
    truth = four_region_image(n)
    noisy = add_gaussian_noise(truth, 0.08, seed=11)
    full = full_coefficients(noisy, max_level=6)
    initial = adaptive_tree_from_detail_threshold(full, 2, 6, threshold=0.012, min_level=2)

    names = ["AMDI_full", "AMDI_fixed", "AMDI_no_diffusion", "AMDI_linear"]
    results = {}
    rows = []
    for name in names:
        image, tree, _, history = run_variant(noisy, full, initial.copy(), name)
        image = np.clip(image, 0, 1)
        results[name] = image
        rows.append({"variant": name, **all_metrics(image, truth, active=tree.basis_size(), full=n*n), "basis_size": tree.basis_size(), "final_energy": history[-1]["energy"]})
    write_csv(out / "ablation_metrics.csv", rows)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4))
    show_normalized_image(axes[0], truth, cmap="gray", vmin=0, vmax=1)
    for ax, name in zip(axes[1:], names):
        show_normalized_image(ax, results[name], cmap="gray", vmin=0, vmax=1)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "ablation")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
