import numpy as np
from amdi.graph import GraphParameters, build_graph_laplacian
from amdi.haar import AdaptiveHaarTree, project_to_tree, coefficients_vector


def test_laplacian_symmetry_psd_kernel():
    x = (np.arange(64)+0.5)/64
    u = np.sin(2*np.pi*x)
    tree = AdaptiveHaarTree.uniform(1, 4)
    c = coefficients_vector(project_to_tree(u, tree), tree)
    L, _ = build_graph_laplacian(tree, c, GraphParameters(k_neighbors=6))
    A = L.toarray()
    assert np.linalg.norm(A-A.T) < 1e-12
    assert np.linalg.norm(A @ np.ones(len(c))) < 1e-12
    assert np.linalg.eigvalsh(A).min() > -1e-11
