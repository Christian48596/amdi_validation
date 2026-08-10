"""AMDI validation package.

The core backend is an exact adaptive tensor-product Haar multiwavelet model
on the unit interval/square.  VAMPyR support is optional and used for
independent adaptive-grid/projection checks.
"""

from .haar import AdaptiveHaarTree, BasisIndex, full_coefficients, reconstruct
from .graph import GraphParameters, build_graph_laplacian
from .energy import EnergyParameters, total_energy

__all__ = [
    "AdaptiveHaarTree",
    "BasisIndex",
    "full_coefficients",
    "reconstruct",
    "GraphParameters",
    "build_graph_laplacian",
    "EnergyParameters",
    "total_energy",
]
