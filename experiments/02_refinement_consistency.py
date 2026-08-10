#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from amdi.plotting import add_panel_label, add_panel_labels, save_publication_figure, show_normalized_image
import numpy as np

from amdi.graph import GraphParameters, build_graph_laplacian
from amdi.haar import AdaptiveHaarTree, project_to_tree, coefficients_vector, transfer_coefficients
from amdi.io_utils import ensure_dir, write_csv
from amdi.synthetic import smooth_reference_1d


def defect(tree_c, tree_f, coeff_c, gp):
    c = coefficients_vector(coeff_c, tree_c)
    Lc, _ = build_graph_laplacian(tree_c, c, gp)

    coeff_embed = transfer_coefficients(coeff_c, tree_c, tree_f)
    cf = coefficients_vector(coeff_embed, tree_f)
    Lf, _ = build_graph_laplacian(tree_f, cf, gp)

    y_c = Lc @ c
    y_dict = dict(zip(tree_c.basis_indices(), np.asarray(y_c).ravel().tolist()))
    y_embed = transfer_coefficients(y_dict, tree_c, tree_f)
    yf_from_c = coefficients_vector(y_embed, tree_f)
    yf = Lf @ cf
    denom = max(np.linalg.norm(c), 1.0e-15)
    return float(np.linalg.norm(yf - yf_from_c) / denom)


def main():
    out = ensure_dir(ROOT / "results" / "02_refinement_consistency")
    _, u = smooth_reference_1d(512)
    rows = []
    gps = {
        "state_dependent": GraphParameters(k_neighbors=8, sigma_c=0.25, state_dependent=True, refinement_decay=2.0),
        "linear": GraphParameters(k_neighbors=8, sigma_c=0.25, state_dependent=False, refinement_decay=2.0),
    }
    for label, gp in gps.items():
        for level in range(2, 7):
            coarse = AdaptiveHaarTree.uniform(1, level)
            fine = AdaptiveHaarTree.uniform(1, level + 1)
            coeff_c = project_to_tree(u, coarse)
            e = defect(coarse, fine, coeff_c, gp)
            rows.append({"model": label, "coarse_level": level, "fine_level": level + 1, "e_rc": e})
    write_csv(out / "refinement_commutator.csv", rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for label in gps:
        rr = [r for r in rows if r["model"] == label]
        ax.semilogy([r["coarse_level"] for r in rr], [r["e_rc"] for r in rr], "o-", label=label.replace("_", " "))
    ax.set_xlabel("Coarse uniform level [-]")
    ax.set_ylabel(r"$e_{\rm rc}$ [-]")
    add_panel_label(ax, "(a)")
    ax.legend()
    fig.tight_layout()
    save_publication_figure(fig, out, "refinement_commutator")

    print("Wrote", out / "refinement_commutator.csv")
    print("NOTE: this experiment measures the defect; convergence must be judged from the generated data, not assumed.")


if __name__ == "__main__":
    main()
