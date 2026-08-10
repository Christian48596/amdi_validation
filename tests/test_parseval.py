import numpy as np
from amdi.haar import AdaptiveHaarTree, project_to_tree, coefficients_vector


def test_parseval_2d_full_haar():
    rng=np.random.default_rng(2)
    u=rng.normal(size=(16,16))
    tree=AdaptiveHaarTree.uniform(2,4)
    c=coefficients_vector(project_to_tree(u,tree),tree)
    lhs=np.mean(u*u)
    rhs=np.dot(c,c)
    assert abs(lhs-rhs) < 1e-12
