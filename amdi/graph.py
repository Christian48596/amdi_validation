"""Construction of the intrinsic adaptive graph Laplacian.

The interaction relation is hierarchical and refinement-stable:

* wavelet orientations on the same support interact;
* neighboring supports at the same level interact;
* a refined child support interacts with the corresponding parent support;
* the root scaling mode interacts with root-level wavelets.

Refining a tree therefore does not delete or rewire interactions that already
existed on the coarse tree.  New cross-level interactions are multiplied by a
level-decay factor, which makes this construction suitable for the numerical
refinement-commutator experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from .haar import AdaptiveHaarTree, basis_geometry


@dataclass(frozen=True)
class GraphParameters:
    # Retained for input compatibility; the hierarchical graph has an explicit
    # bounded degree and does not use a nearest-neighbour search.
    k_neighbors: int = 10
    sigma_x: float = 2.0
    sigma_level: float = 1.5
    sigma_c: float = 0.25
    state_dependent: bool = True
    refinement_decay: float = 1.5
    weight_floor: float = 1.0e-14


def _interaction_pairs(tree: AdaptiveHaarTree):
    idxs = tree.basis_indices()
    support = {}
    for i, idx in enumerate(idxs):
        support.setdefault((idx.level, idx.translation), []).append(i)

    pairs = set()

    # Same-support wavelet orientations.
    for group in support.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                pairs.add((min(group[a], group[b]), max(group[a], group[b]), "same"))

    # Same-level neighboring supports.  Connect matching orientations only;
    # this keeps the interaction degree bounded independently of refinement.
    by_key = {(idx.level, idx.translation, idx.orientation): i for i, idx in enumerate(idxs)}
    for i, idx in enumerate(idxs):
        if idx.kind != "wavelet":
            continue
        for d in range(tree.dim):
            for step in (-1, 1):
                tr = list(idx.translation)
                tr[d] += step
                if tr[d] < 0 or tr[d] >= 2**idx.level:
                    continue
                key = (idx.level, tuple(tr), idx.orientation)
                j = by_key.get(key)
                if j is not None and i != j:
                    pairs.add((min(i, j), max(i, j), "same_level"))

    # Parent-child interactions, matching orientations.  These are the only
    # edges that couple newly introduced fine details back to the old space.
    for i, idx in enumerate(idxs):
        if idx.kind != "wavelet" or idx.level == 0:
            continue
        ptr = tuple(v // 2 for v in idx.translation)
        pkey = (idx.level - 1, ptr, idx.orientation)
        j = by_key.get(pkey)
        if j is not None:
            pairs.add((min(i, j), max(i, j), "parent_child"))

    # Root scaling to root wavelets.
    root_scaling = 0
    for j, idx in enumerate(idxs):
        if idx.kind == "wavelet" and idx.level == 0:
            pairs.add((min(root_scaling, j), max(root_scaling, j), "root"))

    return sorted(pairs)


def build_weight_matrix(
    tree: AdaptiveHaarTree,
    c: np.ndarray,
    params: GraphParameters = GraphParameters(),
) -> csr_matrix:
    c = np.asarray(c, dtype=float)
    centers, levels, sizes = basis_geometry(tree)
    n = len(c)
    if n != len(centers):
        raise ValueError("Coefficient vector does not match active basis")
    if n <= 1:
        return csr_matrix((n, n), dtype=float)

    rows, cols, data = [], [], []
    for i, j, relation in _interaction_pairs(tree):
        dist = float(np.linalg.norm(centers[i] - centers[j]))
        scale = max(0.5 * (sizes[i] + sizes[j]), 1.0e-15)
        kx = np.exp(-((dist / (max(params.sigma_x, 1.0e-15) * scale)) ** 2))
        kl = np.exp(-abs(int(levels[i]) - int(levels[j])) / max(params.sigma_level, 1.0e-15))
        kc = 1.0
        if params.state_dependent:
            kc = np.exp(-(((c[i] - c[j]) / max(params.sigma_c, 1.0e-15)) ** 2))

        # Same-level/same-support weights remain unchanged when deeper levels
        # are added.  Only new cross-level couplings decay with refinement.
        decay = 1.0
        if relation == "parent_child":
            child_level = max(int(levels[i]), int(levels[j]))
            decay = 2.0 ** (-params.refinement_decay * child_level)

        w = float(kx * kl * kc * decay)
        if w <= params.weight_floor:
            continue
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])

    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def build_graph_laplacian(
    tree: AdaptiveHaarTree,
    c: np.ndarray,
    params: GraphParameters = GraphParameters(),
) -> tuple[csr_matrix, csr_matrix]:
    W = build_weight_matrix(tree, c, params)
    degree = np.asarray(W.sum(axis=1)).ravel()
    L = (diags(degree) - W).tocsr()
    return L, W


def local_graph_variation(W: csr_matrix, c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    out = np.zeros_like(c)
    coo = W.tocoo()
    diff2 = (c[coo.row] - c[coo.col]) ** 2
    np.add.at(out, coo.row, coo.data * diff2)
    return out


def structural_diagnostics(L: csr_matrix) -> dict:
    dense = L.toarray()
    symmetry = float(np.linalg.norm(dense - dense.T, ord="fro"))
    row_sum = float(np.linalg.norm(dense @ np.ones(dense.shape[0])))
    eig = np.linalg.eigvalsh(0.5 * (dense + dense.T))
    return {
        "symmetry_error": symmetry,
        "kernel_error": row_sum,
        "min_eigenvalue": float(eig[0]),
        "max_eigenvalue": float(eig[-1]),
        "spectral_gap": float(eig[1]) if len(eig) > 1 else 0.0,
        "eigenvalues": eig,
    }
