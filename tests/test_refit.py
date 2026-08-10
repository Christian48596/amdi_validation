import numpy as np

from amdi.graph import GraphParameters
from amdi.haar import AdaptiveHaarTree, coefficients_vector, full_coefficients, project_to_tree
from amdi.solver import frozen_tree_refit
from amdi.energy import active_data_vector


def test_zero_alpha_refit_is_active_data_projection():
    rng = np.random.default_rng(4)
    u = rng.normal(size=(16, 16))
    full = full_coefficients(u, 4)
    tree = AdaptiveHaarTree.uniform(2, 2)
    start = project_to_tree(u, tree)
    refit = frozen_tree_refit(tree, start, full, 0.0, GraphParameters())
    assert np.allclose(coefficients_vector(refit, tree), active_data_vector(full, tree), atol=1e-13)
