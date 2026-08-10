"""Optional VAMPyR adapter for independent MRA/grid validation.

The AMDI coefficient algebra is implemented in the self-contained Haar backend.
This adapter is used only to cross-check adaptive projection and tree geometry
with VAMPyR/MRCPP.

VAMPyR's shorthand ``MultiResolutionAnalysis(box=[a, b], ...)`` constructor
expects an integer two-entry sequence in the currently distributed bindings.
For the unit domain we therefore use ``box=[0, 1]`` rather than
``[0.0, 1.0]``. This matches the shorthand used in the official VAMPyR
examples.
"""

from __future__ import annotations

import numpy as np


def _module(dim: int):
    try:
        if dim == 1:
            from vampyr import vampyr1d as vp
        elif dim == 2:
            from vampyr import vampyr2d as vp
        else:
            raise ValueError("Only dim=1/2 are used here")
        return vp
    except ImportError as exc:
        raise RuntimeError(
            "VAMPyR is not installed. Install it from conda-forge with "
            "`conda install -c conda-forge vampyr`."
        ) from exc


def _unit_mra(vp, order: int, max_depth: int = 9):
    """Construct an MRA on the unit domain with an explicit depth cap.

    The depth cap is important for 2D discontinuous/sharp functions: a very
    small adaptive projection tolerance can otherwise generate millions of
    nodes before the requested tolerance is reached.
    """
    return vp.MultiResolutionAnalysis(
        box=[0, 1], order=int(order), max_depth=int(max_depth)
    )


def adaptive_project(function, dim: int, order: int = 5, precision: float = 1.0e-4, max_depth: int = 9):
    vp = _module(dim)
    mra = _unit_mra(vp, order, max_depth=max_depth)
    projector = vp.ScalingProjector(mra, float(precision))
    return projector(function)


def fixed_scale_project(function, dim: int, scale: int, order: int = 5, max_depth: int = 9):
    vp = _module(dim)
    mra = _unit_mra(vp, order, max_depth=max_depth)
    projector = vp.ScalingProjector(mra, int(scale))
    return projector(function)


def end_node_table(tree) -> list[dict]:
    rows = []
    for i in range(int(tree.nEndNodes())):
        node = tree.fetchEndNode(i)
        idx = node.index()
        rows.append({
            "i": i,
            "scale": int(node.scale()),
            "translation": tuple(int(v) for v in idx.translation()),
            "center": tuple(float(v) for v in np.asarray(node.center()).ravel()),
            "lower": tuple(float(v) for v in np.asarray(node.lowerBounds()).ravel()),
            "upper": tuple(float(v) for v in np.asarray(node.upperBounds()).ravel()),
            "norm": float(node.norm()),
            "wavelet_norm": float(node.waveletNorm()),
            "scaling_norm": float(node.scalingNorm()),
        })
    return rows


def tree_summary(tree) -> dict:
    return {
        "n_nodes": int(tree.nNodes()),
        "n_end_nodes": int(tree.nEndNodes()),
        "n_root_nodes": int(tree.nRootNodes()),
        "depth": int(tree.depth()),
        "root_scale": int(tree.rootScale()),
        "norm": float(tree.norm()),
    }


def make_mra(dim: int, order: int = 5, max_depth: int = 9):
    """Public constructor used by convergence experiments."""
    vp = _module(dim)
    return vp, _unit_mra(vp, order, max_depth=max_depth)


def adaptive_project_on_mra(function, vp, mra, precision: float):
    """Project using an already constructed MRA so different tolerances share it."""
    projector = vp.ScalingProjector(mra, float(precision))
    return projector(function)


def l2_distance(tree_a, tree_b, vp) -> float:
    """L2 distance between two compatible FunctionTrees using VAMPyR dot products."""
    aa = float(vp.dot(tree_a, tree_a))
    bb = float(vp.dot(tree_b, tree_b))
    ab = float(vp.dot(tree_a, tree_b))
    value = max(aa + bb - 2.0 * ab, 0.0)
    return float(np.sqrt(value))
