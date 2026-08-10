"""AMDI state energy for the orthonormal Haar validation backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .graph import GraphParameters, build_graph_laplacian
from .haar import AdaptiveHaarTree, BasisIndex


@dataclass(frozen=True)
class EnergyParameters:
    alpha: float = 0.05
    beta: float = 0.002
    tau: float = 1.0e-5
    gamma_level: float = 0.1
    sparsity_epsilon: float = 1.0e-6
    smooth_l1: bool = True


def active_data_vector(full_data: Dict[BasisIndex, float], tree: AdaptiveHaarTree) -> np.ndarray:
    return np.asarray([full_data.get(i, 0.0) for i in tree.basis_indices()], dtype=float)


def fidelity_energy(c: np.ndarray, tree: AdaptiveHaarTree, full_data: Dict[BasisIndex, float]) -> float:
    """Exact L2 fidelity using Parseval, including inactive tail energy."""
    f_active = active_data_vector(full_data, tree)
    active_keys = set(tree.basis_indices())
    tail = sum(v * v for k, v in full_data.items() if k not in active_keys)
    return 0.5 * (float(np.dot(c - f_active, c - f_active)) + float(tail))


def sparsity_energy(c: np.ndarray, tree: AdaptiveHaarTree, params: EnergyParameters) -> float:
    levels = np.asarray([idx.level for idx in tree.basis_indices()], dtype=float)
    nu = 1.0 + 0.05 * levels
    if params.smooth_l1:
        phi = np.sqrt(c * c + params.sparsity_epsilon**2) - params.sparsity_epsilon
    else:
        phi = np.abs(c)
    return params.beta * float(np.dot(nu, phi))


def tree_complexity(tree: AdaptiveHaarTree, params: EnergyParameters) -> float:
    return float(sum(1.0 + params.gamma_level * idx.level for idx in tree.basis_indices()))


def total_energy(
    c: np.ndarray,
    tree: AdaptiveHaarTree,
    full_data: Dict[BasisIndex, float],
    eparams: EnergyParameters = EnergyParameters(),
    gparams: GraphParameters = GraphParameters(),
) -> tuple[float, dict]:
    c = np.asarray(c, dtype=float)
    L, W = build_graph_laplacian(tree, c, gparams)
    fidelity = fidelity_energy(c, tree, full_data)
    diffusion = 0.5 * eparams.alpha * float(c @ (L @ c))
    sparsity = sparsity_energy(c, tree, eparams)
    complexity = eparams.tau * tree_complexity(tree, eparams)
    total = fidelity + diffusion + sparsity + complexity
    return total, {
        "fidelity": fidelity,
        "diffusion": diffusion,
        "sparsity": sparsity,
        "complexity": complexity,
        "L": L,
        "W": W,
    }
