#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.benchmarks import heat_diffusion, perona_malik, haar_soft_threshold, total_variation
from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from amdi.haar import AdaptiveHaarTree, adaptive_tree_from_detail_threshold, full_coefficients
from amdi.io_utils import ensure_dir, write_csv
from amdi.metrics import all_metrics
from amdi.solver import adaptive_frozen_denoise
from amdi.synthetic import four_region_image, add_gaussian_noise


def main():
    out = ensure_dir(ROOT / "results" / "05_denoising_benchmark")
    n = 64
    sigma = 0.08
    truth = four_region_image(n)
    noisy = add_gaussian_noise(truth, sigma, seed=11)
    max_level = 6
    full = full_coefficients(noisy, max_level=max_level)

    initial = adaptive_tree_from_detail_threshold(full, 2, max_level, threshold=0.012, min_level=2)
    ep = EnergyParameters(alpha=0.010, beta=0.0002, tau=1e-7, smooth_l1=False)
    gp = GraphParameters(k_neighbors=10, sigma_c=0.12, refinement_decay=0.35)
    amdi, tree, _, history = adaptive_frozen_denoise(noisy, initial, full, ep, gp, h=0.6, outer_iterations=6, adapt=True, zeta=1e-6)

    fixed_tree = AdaptiveHaarTree.uniform(2, level=5)
    fixed, fixed_tree, _, _ = adaptive_frozen_denoise(noisy, fixed_tree, full, ep, gp, h=0.6, outer_iterations=6, adapt=False)

    methods = {
        "Noisy": noisy,
        "Heat": heat_diffusion(noisy, sigma=0.9),
        "Anisotropic": perona_malik(noisy, iterations=20, kappa=0.09, dt=0.16),
        "Wavelet soft": haar_soft_threshold(noisy, threshold=sigma * np.sqrt(2.0 * np.log(n * n)) / n),
        "TV": total_variation(noisy, weight=0.075),
        "Fixed intrinsic": np.clip(fixed, 0, 1),
        "AMDI": np.clip(amdi, 0, 1),
    }

    rows = []
    for name, img in methods.items():
        active = tree.basis_size() if name == "AMDI" else (fixed_tree.basis_size() if name == "Fixed intrinsic" else n * n)
        row = {"method": name, **all_metrics(img, truth, active=active, full=n*n)}
        rows.append(row)
    write_csv(out / "benchmark_metrics.csv", rows)
    write_csv(out / "amdi_energy_history.csv", history)

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.6))
    display = [("Truth", truth)] + list(methods.items())[:7]
    flat_axes = axes.ravel()
    for ax, (_, img) in zip(flat_axes, display):
        show_normalized_image(ax, img, cmap="gray", vmin=0, vmax=1)
    add_panel_labels(flat_axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "denoising_comparison")

    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
