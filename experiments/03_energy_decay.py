#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.energy import EnergyParameters, total_energy
from amdi.graph import GraphParameters
from amdi.haar import AdaptiveHaarTree, full_coefficients, project_to_tree, coefficients_vector
from amdi.io_utils import ensure_dir, write_csv
from amdi.solver import exact_split_step, candidate_trees_from_data
from amdi.synthetic import reference_1d, add_gaussian_noise


def main():
    out = ensure_dir(ROOT / "results" / "03_energy_decay")
    _, truth = reference_1d(128)
    noisy = add_gaussian_noise(truth, 0.05, seed=7)
    full = full_coefficients(noisy, max_level=7)

    tree = AdaptiveHaarTree.uniform(1, 3)
    current = project_to_tree(noisy, tree)
    ep = EnergyParameters(alpha=0.04, beta=0.001, tau=2e-5, smooth_l1=True)
    gp = GraphParameters(k_neighbors=6, sigma_c=0.18, refinement_decay=0.4)
    h = 0.5
    zeta = 1e-5

    rows = []
    E0, _ = total_energy(coefficients_vector(current, tree), tree, full, ep, gp)
    rows.append({"iteration": 0, "energy": E0, "delta_E": 0.0, "basis_size": tree.basis_size(), "increment_term": 0.0, "inequality_residual": 0.0})

    for n in range(8):
        old_tree = tree.copy()
        old_coeff = current.copy()
        old_vec = coefficients_vector(old_coeff, old_tree)
        old_E, _ = total_energy(old_vec, old_tree, full, ep, gp)

        candidates = candidate_trees_from_data(tree, current, full, refine_fraction=0.25, coarsen_fraction=0.25)
        fixed, adapted = exact_split_step(tree, current, candidates, full, h, ep, gp, zeta=zeta)
        tree, current = adapted.tree, adapted.coeffs

        # Exact split inequality uses the fixed-tree increment plus tree-change penalty.
        fixed_vec = coefficients_vector(fixed.coeffs, old_tree)
        incr = 0.5 / h * float(np.dot(fixed_vec - old_vec, fixed_vec - old_vec))
        tree_pen = 0.5 * zeta * old_tree.distance(tree) ** 2
        residual = adapted.energy + incr + tree_pen - old_E
        rows.append({
            "iteration": n + 1,
            "energy": adapted.energy,
            "delta_E": adapted.energy - old_E,
            "basis_size": tree.basis_size(),
            "increment_term": incr,
            "inequality_residual": residual,
        })

    write_csv(out / "energy_history.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    iterations = [r["iteration"] for r in rows]
    energies = [r["energy"] for r in rows]
    residuals = [r["inequality_residual"] for r in rows[1:]]

    axes[0].plot(iterations, energies, "o-")
    axes[0].set_xlabel("Iteration [-]")
    axes[0].set_ylabel(r"$\mathcal{E}(V^n)$ [-]")
    axes[0].grid(alpha=0.2)

    axes[1].plot(iterations[1:], residuals, "o-")
    axes[1].axhline(0.0, lw=1.0, ls="--")
    axes[1].set_xlabel("Iteration [-]")
    axes[1].set_ylabel(r"Discrete inequality residual $R_n$ [-]")
    axes[1].grid(alpha=0.2)

    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, out, "energy_decay")

    worst = max(r["inequality_residual"] for r in rows[1:])
    print(f"Largest split-energy inequality residual: {worst:.6e}")


if __name__ == "__main__":
    main()
