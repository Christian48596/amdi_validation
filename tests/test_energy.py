import numpy as np
from amdi.energy import EnergyParameters, total_energy
from amdi.graph import GraphParameters
from amdi.haar import AdaptiveHaarTree, full_coefficients, project_to_tree, coefficients_vector
from amdi.solver import exact_fixed_tree_step


def test_exact_fixed_tree_step_decreases_incremental_objective():
    x=(np.arange(64)+0.5)/64
    noisy=np.sin(2*np.pi*x)+0.03*np.cos(14*np.pi*x)
    full=full_coefficients(noisy, 6)
    tree=AdaptiveHaarTree.uniform(1, 3)
    current=project_to_tree(noisy, tree)
    ep=EnergyParameters(alpha=0.03,beta=0.001,tau=1e-5)
    gp=GraphParameters(k_neighbors=6,sigma_c=0.2)
    c0=coefficients_vector(current,tree)
    E0,_=total_energy(c0,tree,full,ep,gp)
    step=exact_fixed_tree_step(tree,current,full,0.5,ep,gp)
    c1=coefficients_vector(step.coeffs,tree)
    incremental=0.5/0.5*np.dot(c1-c0,c1-c0)+step.energy
    assert incremental <= E0 + 1e-8
