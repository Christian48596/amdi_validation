"""Exact small-scale and frozen-weight AMDI solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh, spsolve

from .energy import EnergyParameters, active_data_vector, total_energy
from .graph import GraphParameters, build_graph_laplacian, local_graph_variation
from .haar import (
    AdaptiveHaarTree,
    BasisIndex,
    coefficients_vector,
    reconstruct,
    transfer_coefficients,
    vector_to_coefficients,
)


@dataclass
class StepResult:
    tree: AdaptiveHaarTree
    coeffs: Dict[BasisIndex, float]
    energy: float
    diagnostics: dict


def exact_fixed_tree_step(
    tree: AdaptiveHaarTree,
    current: Dict[BasisIndex, float],
    full_data: Dict[BasisIndex, float],
    h: float,
    eparams: EnergyParameters,
    gparams: GraphParameters,
) -> StepResult:
    c0 = coefficients_vector(current, tree)

    def objective(c):
        E, _ = total_energy(c, tree, full_data, eparams, gparams)
        return 0.5 / h * float(np.dot(c - c0, c - c0)) + E

    opt = minimize(
        objective,
        c0,
        method="L-BFGS-B",
        options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 800, "maxls": 40},
    )
    c = np.asarray(opt.x, float)
    E, parts = total_energy(c, tree, full_data, eparams, gparams)
    return StepResult(
        tree=tree.copy(),
        coeffs=vector_to_coefficients(c, tree),
        energy=E,
        diagnostics={"optimizer_success": bool(opt.success), "optimizer_message": str(opt.message), **parts},
    )


def choose_tree_variationally(
    source_tree: AdaptiveHaarTree,
    source_coeffs: Dict[BasisIndex, float],
    candidates: Iterable[AdaptiveHaarTree],
    full_data: Dict[BasisIndex, float],
    eparams: EnergyParameters,
    gparams: GraphParameters,
    zeta: float = 0.0,
) -> StepResult:
    best = None
    for candidate in candidates:
        cc = transfer_coefficients(source_coeffs, source_tree, candidate)
        cv = coefficients_vector(cc, candidate)
        E, parts = total_energy(cv, candidate, full_data, eparams, gparams)
        score = E + 0.5 * zeta * float(source_tree.distance(candidate) ** 2)
        if best is None or score < best[0]:
            best = (score, candidate.copy(), cc, E, parts)
    assert best is not None
    _, tree, coeffs, E, parts = best
    return StepResult(tree=tree, coeffs=coeffs, energy=E, diagnostics=parts)


def exact_split_step(
    tree: AdaptiveHaarTree,
    current: Dict[BasisIndex, float],
    candidates: Iterable[AdaptiveHaarTree],
    full_data: Dict[BasisIndex, float],
    h: float,
    eparams: EnergyParameters,
    gparams: GraphParameters,
    zeta: float = 0.0,
) -> tuple[StepResult, StepResult]:
    fixed = exact_fixed_tree_step(tree, current, full_data, h, eparams, gparams)
    adapted = choose_tree_variationally(tree, fixed.coeffs, candidates, full_data, eparams, gparams, zeta)
    return fixed, adapted


def _soft_threshold(x: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def frozen_weight_fista(
    tree: AdaptiveHaarTree,
    current: Dict[BasisIndex, float],
    full_data: Dict[BasisIndex, float],
    h: float,
    eparams: EnergyParameters,
    gparams: GraphParameters,
    max_iter: int = 500,
    tol: float = 1.0e-9,
) -> Dict[BasisIndex, float]:
    c0 = coefficients_vector(current, tree)
    f = active_data_vector(full_data, tree)
    L, _ = build_graph_laplacian(tree, c0, gparams)
    n = len(c0)
    if n == 1:
        lambda_max = 0.0
    elif n <= 128:
        lambda_max = float(np.linalg.eigvalsh(L.toarray())[-1])
    else:
        try:
            lambda_max = float(eigsh(L, k=1, which="LA", return_eigenvectors=False)[0])
        except Exception:
            lambda_max = float(2.0 * np.max(np.asarray(L.diagonal())))
    lipschitz = 1.0 / h + 1.0 + eparams.alpha * max(lambda_max, 0.0)
    step = 1.0 / max(lipschitz, 1.0e-15)
    levels = np.asarray([idx.level for idx in tree.basis_indices()], float)
    nu = 1.0 + 0.05 * levels

    x = c0.copy()
    y = x.copy()
    t = 1.0
    for _ in range(max_iter):
        grad = (y - c0) / h + (y - f) + eparams.alpha * (L @ y)
        x_new = _soft_threshold(y - step * grad, step * eparams.beta * nu)
        if np.linalg.norm(x_new - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_new
            break
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return vector_to_coefficients(x, tree)


def safeguard_candidate(
    tree: AdaptiveHaarTree,
    old: Dict[BasisIndex, float],
    candidate: Dict[BasisIndex, float],
    full_data: Dict[BasisIndex, float],
    eparams: EnergyParameters,
    gparams: GraphParameters,
    max_backtracks: int = 20,
    return_diagnostics: bool = False,
):
    """Energy safeguard for the frozen-weight proposal.

    The default return value is backward compatible with v0.2:
    ``(coefficients, energy, accepted)``.  When ``return_diagnostics`` is
    true, a fourth dictionary reports the accepted line-search factor and
    number of backtracks.
    """
    old_vec = coefficients_vector(old, tree)
    cand_vec = coefficients_vector(candidate, tree)
    Eold, _ = total_energy(old_vec, tree, full_data, eparams, gparams)
    direction = cand_vec - old_vec
    factor = 1.0
    for backtracks in range(max_backtracks + 1):
        trial = old_vec + factor * direction
        E, _ = total_energy(trial, tree, full_data, eparams, gparams)
        if E <= Eold + 1.0e-12:
            result = (vector_to_coefficients(trial, tree), E, True)
            if return_diagnostics:
                return (*result, {"factor": factor, "backtracks": backtracks})
            return result
        factor *= 0.5
    result = (old.copy(), Eold, False)
    if return_diagnostics:
        return (*result, {"factor": 0.0, "backtracks": max_backtracks + 1})
    return result


def frozen_tree_refit(
    tree: AdaptiveHaarTree,
    start: Dict[BasisIndex, float],
    full_data: Dict[BasisIndex, float],
    alpha: float,
    gparams: GraphParameters,
) -> Dict[BasisIndex, float]:
    """Debias/refit coefficients on a fixed selected tree.

    Interaction weights are frozen at ``start`` and the no-sparsity problem

        1/2 ||c-f||^2 + alpha/2 c^T L(start) c

    is solved exactly.  The tree, and therefore the representation
    complexity, is unchanged.  ``alpha=0`` reduces to the orthogonal data
    projection on the selected adaptive space.
    """
    c0 = coefficients_vector(start, tree)
    f = active_data_vector(full_data, tree)
    L, _ = build_graph_laplacian(tree, c0, gparams)
    if alpha <= 0.0:
        x = f
    else:
        from scipy.sparse import eye
        A = eye(len(c0), format="csr") + float(alpha) * L
        x = np.asarray(spsolve(A, f), float)
    return vector_to_coefficients(x, tree)


def candidate_trees_from_data(
    tree: AdaptiveHaarTree,
    coeffs: Dict[BasisIndex, float],
    full_data: Dict[BasisIndex, float],
    refine_fraction: float = 0.10,
    coarsen_fraction: float = 0.10,
) -> List[AdaptiveHaarTree]:
    """Construct current/refine/coarsen candidates with automatic ancestry."""
    candidates = [tree.copy()]

    # Refinement score: prospective data detail energy on leaves.
    leaves = [c for c in tree.leaves() if c[0] < tree.max_level]
    scores = []
    for level, tr in leaves:
        detail = sum(
            full_data.get(BasisIndex("wavelet", level, tr, o), 0.0) ** 2
            for o in range(1, 2**tree.dim)
        )
        scores.append((detail, (level, tr)))
    scores.sort(reverse=True)
    nref = max(1, int(np.ceil(refine_fraction * len(scores)))) if scores else 0
    if nref:
        refined = tree.copy()
        for _, cell in scores[:nref]:
            refined.refine(cell)
        candidates.append(refined)

    # Coarsening score: current detail energy on prunable cells.
    cscores = []
    for level, tr in tree.prunable_cells():
        detail = sum(
            coeffs.get(BasisIndex("wavelet", level, tr, o), 0.0) ** 2
            for o in range(1, 2**tree.dim)
        )
        cscores.append((detail, (level, tr)))
    cscores.sort()
    ncrs = max(1, int(np.ceil(coarsen_fraction * len(cscores)))) if cscores else 0
    if ncrs:
        coarsened = tree.copy()
        for _, cell in cscores[:ncrs]:
            coarsened.coarsen(cell)
        candidates.append(coarsened)

    # Remove duplicate structural states.
    unique = []
    seen = set()
    for cand in candidates:
        key = frozenset(cand.refined)
        if key not in seen:
            unique.append(cand)
            seen.add(key)
    return unique


def adaptive_frozen_denoise(
    noisy: np.ndarray,
    initial_tree: AdaptiveHaarTree,
    full_data: Dict[BasisIndex, float],
    eparams: EnergyParameters,
    gparams: GraphParameters,
    h: float = 0.8,
    outer_iterations: int = 8,
    adapt: bool = True,
    zeta: float = 0.0,
    refine_fraction: float = 0.10,
    coarsen_fraction: float = 0.10,
    stop_on_convergence: bool = False,
    update_tol: float = 1.0e-5,
    energy_tol: float = 1.0e-8,
    patience: int = 2,
    min_iterations: int = 3,
) -> tuple[np.ndarray, AdaptiveHaarTree, Dict[BasisIndex, float], list]:
    """Frozen-weight adaptive denoising with convergence diagnostics.

    The solver records energy, relative image update, active complexity, tree
    changes, safeguard line-search information, and monotonicity at every
    outer iteration.  Optional stopping requires both the relative update and
    relative energy change to remain below tolerance for ``patience``
    consecutive iterations.
    """
    tree = initial_tree.copy()
    current = {idx: full_data.get(idx, 0.0) for idx in tree.basis_indices()}
    history = []

    previous_image = reconstruct(current, tree, noisy.shape)
    E0, _ = total_energy(coefficients_vector(current, tree), tree, full_data, eparams, gparams)
    history.append({
        "iteration": 0,
        "energy": E0,
        "energy_change": 0.0,
        "relative_energy_change": 0.0,
        "relative_update": 0.0,
        "basis_size": tree.basis_size(),
        "tree_distance": 0,
        "safeguard_accepted": True,
        "safeguard_factor": 1.0,
        "safeguard_backtracks": 0,
        "energy_monotone": True,
    })

    stable_count = 0
    for it in range(1, outer_iterations + 1):
        old_tree = tree.copy()
        old_energy = float(history[-1]["energy"])

        candidate = frozen_weight_fista(tree, current, full_data, h, eparams, gparams)
        candidate, _, accepted, sg = safeguard_candidate(
            tree, current, candidate, full_data, eparams, gparams,
            return_diagnostics=True,
        )

        if adapt:
            candidates = candidate_trees_from_data(
                tree, candidate, full_data,
                refine_fraction=refine_fraction,
                coarsen_fraction=coarsen_fraction,
            )
            selected = choose_tree_variationally(tree, candidate, candidates, full_data, eparams, gparams, zeta)
            tree = selected.tree
            current = selected.coeffs
            E = selected.energy
        else:
            current = candidate
            E, _ = total_energy(coefficients_vector(current, tree), tree, full_data, eparams, gparams)

        image = reconstruct(current, tree, noisy.shape)
        relative_update = float(np.linalg.norm(image - previous_image) / max(np.linalg.norm(previous_image), 1.0e-15))
        dE = float(E - old_energy)
        rel_dE = float(abs(dE) / max(abs(old_energy), 1.0e-15))
        monotone = bool(E <= old_energy + 1.0e-10)
        tree_distance = int(old_tree.distance(tree))

        history.append({
            "iteration": it,
            "energy": float(E),
            "energy_change": dE,
            "relative_energy_change": rel_dE,
            "relative_update": relative_update,
            "basis_size": tree.basis_size(),
            "tree_distance": tree_distance,
            "safeguard_accepted": bool(accepted),
            "safeguard_factor": float(sg["factor"]),
            "safeguard_backtracks": int(sg["backtracks"]),
            "energy_monotone": monotone,
        })
        previous_image = image

        if stop_on_convergence and it >= min_iterations:
            if relative_update <= update_tol and rel_dE <= energy_tol:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= max(1, int(patience)):
                break

    image = reconstruct(current, tree, noisy.shape)
    return image, tree, current, history


def marking_indicator(
    tree: AdaptiveHaarTree,
    coeffs: Dict[BasisIndex, float],
    full_data: Dict[BasisIndex, float],
    alpha: float,
    gparams: GraphParameters,
    theta_c: float = 1.0,
    theta_d: float = 1.0,
    theta_j: float = 1.0,
) -> np.ndarray:
    c = coefficients_vector(coeffs, tree)
    f = active_data_vector(full_data, tree)
    _, W = build_graph_laplacian(tree, c, gparams)
    local = local_graph_variation(W, c)
    return np.sqrt(theta_c * c * c + theta_d * (c - f) ** 2 + theta_j * alpha * local)
