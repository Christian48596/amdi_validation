import numpy as np
from amdi.haar import AdaptiveHaarTree, full_coefficients, project_to_tree, reconstruct, transfer_coefficients


def test_uniform_basis_dimension():
    for level in range(1, 5):
        t1 = AdaptiveHaarTree.uniform(1, level)
        assert t1.basis_size() == 2**level
        t2 = AdaptiveHaarTree.uniform(2, level)
        assert t2.basis_size() == 4**level


def test_projection_reconstruction_full_1d():
    rng = np.random.default_rng(1)
    u = rng.normal(size=64)
    tree = AdaptiveHaarTree.uniform(1, 6)
    c = project_to_tree(u, tree)
    ur = reconstruct(c, tree, u.shape)
    assert np.max(np.abs(ur-u)) < 1e-12


def test_transfer_refine_then_coarsen():
    u = np.linspace(0, 1, 64)
    coarse = AdaptiveHaarTree.uniform(1, 3)
    fine = AdaptiveHaarTree.uniform(1, 4)
    c = project_to_tree(u, coarse)
    cf = transfer_coefficients(c, coarse, fine)
    back = transfer_coefficients(cf, fine, coarse)
    for k, v in c.items():
        assert abs(back[k] - v) < 1e-14
