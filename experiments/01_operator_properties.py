#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.graph import GraphParameters, build_graph_laplacian, structural_diagnostics
from amdi.haar import AdaptiveHaarTree, project_to_tree, coefficients_vector
from amdi.io_utils import ensure_dir, write_json
from amdi.synthetic import reference_1d


def main():
    out = ensure_dir(ROOT / "results" / "01_operator_properties")
    _, u = reference_1d(256)
    tree = AdaptiveHaarTree.uniform(dim=1, level=5)
    coeffs = project_to_tree(u, tree)
    c = coefficients_vector(coeffs, tree)
    L, _ = build_graph_laplacian(tree, c, GraphParameters(k_neighbors=8, sigma_c=0.2))
    d = structural_diagnostics(L)
    eig = d.pop("eigenvalues")
    write_json(out / "diagnostics.json", d)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(np.arange(len(eig)), eig, "o", ms=4)
    ax.axhline(0.0, lw=1.0, ls="--")
    ax.set_xlabel("Eigenvalue index [-]")
    ax.set_ylabel(r"$\lambda_j(L_V)$ [-]")
    add_panel_label(ax, "(a)")
    fig.tight_layout()
    save_publication_figure(fig, out, "operator_spectrum")

    print("Operator diagnostics")
    for k, v in d.items():
        print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
