import numpy as np

from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from amdi.haar import AdaptiveHaarTree, full_coefficients
from amdi.solver import adaptive_frozen_denoise


def test_frozen_solver_reports_monotone_energy_and_diagnostics():
    n = 16
    y, x = np.mgrid[0:n, 0:n]
    u = 0.2 + 0.6 * (x >= n//2)
    full = full_coefficients(u, 4)
    tree = AdaptiveHaarTree.uniform(2, 2)
    _, _, _, history = adaptive_frozen_denoise(
        u, tree, full,
        EnergyParameters(alpha=0.01, beta=0.0, tau=1e-8, smooth_l1=False),
        GraphParameters(sigma_c=0.15),
        h=0.5, outer_iterations=3, adapt=True,
    )
    assert len(history) == 4
    assert all("relative_update" in h for h in history)
    assert all("tree_distance" in h for h in history)
    assert all(bool(h["energy_monotone"]) for h in history[1:])
