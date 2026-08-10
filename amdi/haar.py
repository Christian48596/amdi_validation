"""Adaptive tensor-product Haar multiwavelet backend.

The implementation uses the root scaling function together with the
(2**dim - 1) tensor-product Haar wavelets attached to every refined dyadic
cell.  The basis is orthonormal in L2([0,1]^dim), so the Gram matrix is the
identity.  This backend is intentionally simple and exact: it is used to
validate the operator/variational claims independently of any external MRA
library.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import numpy as np

Cell = Tuple[int, Tuple[int, ...]]


@dataclass(frozen=True, order=True)
class BasisIndex:
    kind: str                  # "scaling" or "wavelet"
    level: int
    translation: Tuple[int, ...]
    orientation: int = 0       # bit mask, 1..2**dim-1 for wavelets

    def label(self) -> str:
        if self.kind == "scaling":
            return "S0"
        return f"W(l={self.level},k={self.translation},o={self.orientation})"


class AdaptiveHaarTree:
    """Tree of refined dyadic cells.

    A cell in ``refined`` is split into 2**dim children.  The active
    orthonormal basis consists of the root scaling function plus all Haar
    wavelet orientations associated with refined cells.
    """

    def __init__(self, dim: int, max_level: int, refined: Iterable[Cell] | None = None):
        if dim not in (1, 2):
            raise ValueError("This validation backend supports dim=1 or dim=2.")
        if max_level < 0:
            raise ValueError("max_level must be nonnegative")
        self.dim = int(dim)
        self.max_level = int(max_level)
        self.refined: Set[Cell] = set(refined or [])
        self._enforce_ancestry()
        self._validate()

    @staticmethod
    def root(dim: int) -> Cell:
        return (0, (0,) * dim)

    def copy(self) -> "AdaptiveHaarTree":
        return AdaptiveHaarTree(self.dim, self.max_level, self.refined)

    def _enforce_ancestry(self) -> None:
        add: Set[Cell] = set()
        for level, tr in list(self.refined):
            if level >= self.max_level:
                continue
            cur_level, cur_tr = level, tr
            while cur_level > 0:
                cur_tr = tuple(v // 2 for v in cur_tr)
                cur_level -= 1
                add.add((cur_level, cur_tr))
        self.refined |= add

    def _validate(self) -> None:
        for level, tr in self.refined:
            if not (0 <= level < self.max_level):
                raise ValueError(f"Refined cell {(level, tr)} is outside [0,max_level).")
            if len(tr) != self.dim:
                raise ValueError("translation has wrong dimension")
            if any(k < 0 or k >= 2**level for k in tr):
                raise ValueError(f"Invalid translation {tr} at level {level}")

    def children(self, cell: Cell) -> List[Cell]:
        level, tr = cell
        if level >= self.max_level:
            return []
        return [
            (level + 1, tuple(2 * tr[d] + bit[d] for d in range(self.dim)))
            for bit in product((0, 1), repeat=self.dim)
        ]

    def parent(self, cell: Cell) -> Cell | None:
        level, tr = cell
        if level == 0:
            return None
        return (level - 1, tuple(v // 2 for v in tr))

    def leaves(self) -> List[Cell]:
        out: List[Cell] = []
        stack = [self.root(self.dim)]
        while stack:
            cell = stack.pop()
            level, _ = cell
            if cell in self.refined and level < self.max_level:
                stack.extend(reversed(self.children(cell)))
            else:
                out.append(cell)
        return sorted(out)

    def prunable_cells(self) -> List[Cell]:
        """Refined cells whose children are all leaves."""
        result = []
        for cell in self.refined:
            if all(child not in self.refined for child in self.children(cell)):
                result.append(cell)
        return sorted(result)

    def refine(self, cell: Cell) -> None:
        level, _ = cell
        if level >= self.max_level:
            return
        if cell not in self.leaves():
            return
        self.refined.add(cell)
        self._enforce_ancestry()

    def coarsen(self, cell: Cell) -> None:
        """Remove a refined cell and all refined descendants."""
        level, tr = cell
        kill = set()
        for lev, kt in self.refined:
            if lev < level:
                continue
            factor = 2 ** (lev - level)
            anc = tuple(v // factor for v in kt)
            if anc == tr:
                kill.add((lev, kt))
        self.refined -= kill

    def basis_indices(self) -> List[BasisIndex]:
        idx = [BasisIndex("scaling", 0, (0,) * self.dim, 0)]
        for level, tr in sorted(self.refined):
            for orientation in range(1, 2**self.dim):
                idx.append(BasisIndex("wavelet", level, tr, orientation))
        return idx

    def basis_size(self) -> int:
        return 1 + (2**self.dim - 1) * len(self.refined)

    def structural_size(self) -> int:
        return len(self.refined) + len(self.leaves())

    def distance(self, other: "AdaptiveHaarTree") -> int:
        if self.dim != other.dim:
            raise ValueError("Tree dimensions differ")
        return len(self.refined.symmetric_difference(other.refined))

    @classmethod
    def uniform(cls, dim: int, level: int) -> "AdaptiveHaarTree":
        """Uniform partition with leaves at ``level``."""
        tree = cls(dim=dim, max_level=level)
        for lev in range(level):
            for tr in product(range(2**lev), repeat=dim):
                tree.refined.add((lev, tuple(tr)))
        return tree


def _cell_slices(shape: Sequence[int], level: int, tr: Tuple[int, ...]) -> Tuple[slice, ...]:
    slices = []
    for n, k in zip(shape, tr):
        if n % (2**level) != 0:
            raise ValueError("Each grid dimension must be divisible by 2**level")
        width = n // (2**level)
        slices.append(slice(k * width, (k + 1) * width))
    return tuple(slices)


def _child_slices(shape: Sequence[int], level: int, tr: Tuple[int, ...]) -> Iterator[Tuple[Tuple[int, ...], Tuple[slice, ...]]]:
    parent = _cell_slices(shape, level, tr)
    for bits in product((0, 1), repeat=len(shape)):
        child = []
        for sl, bit in zip(parent, bits):
            width = sl.stop - sl.start
            half = width // 2
            if half == 0:
                raise ValueError("Grid is too coarse for requested Haar level")
            start = sl.start + bit * half
            child.append(slice(start, start + half))
        yield tuple(bits), tuple(child)


def _orientation_sign(bits: Tuple[int, ...], orientation: int) -> float:
    sign = 1.0
    for d, bit in enumerate(bits):
        if orientation & (1 << d):
            sign *= -1.0 if bit else 1.0
    return sign


def coefficient_for_index(array: np.ndarray, idx: BasisIndex) -> float:
    """Exact discrete quadrature coefficient for a dyadically aligned array.

    The array is interpreted as a piecewise-constant function on the unit
    domain.  Uniform pixel/cell quadrature is exact for the Haar basis.
    """
    arr = np.asarray(array, dtype=float)
    dim = arr.ndim
    pixel_volume = 1.0 / float(np.prod(arr.shape))
    if idx.kind == "scaling":
        return float(arr.sum() * pixel_volume)
    amplitude = 2.0 ** (dim * idx.level / 2.0)
    total = 0.0
    for bits, sl in _child_slices(arr.shape, idx.level, idx.translation):
        total += _orientation_sign(bits, idx.orientation) * float(arr[sl].sum())
    return amplitude * pixel_volume * total


def project_to_tree(array: np.ndarray, tree: AdaptiveHaarTree) -> Dict[BasisIndex, float]:
    return {idx: coefficient_for_index(array, idx) for idx in tree.basis_indices()}


def full_coefficients(array: np.ndarray, max_level: int | None = None) -> Dict[BasisIndex, float]:
    arr = np.asarray(array, dtype=float)
    if max_level is None:
        max_level = min(int(np.log2(n)) for n in arr.shape)
    tree = AdaptiveHaarTree.uniform(arr.ndim, max_level)
    return project_to_tree(arr, tree)


def coefficients_vector(coeffs: Dict[BasisIndex, float], tree: AdaptiveHaarTree) -> np.ndarray:
    return np.asarray([coeffs.get(i, 0.0) for i in tree.basis_indices()], dtype=float)


def vector_to_coefficients(vector: np.ndarray, tree: AdaptiveHaarTree) -> Dict[BasisIndex, float]:
    idx = tree.basis_indices()
    vec = np.asarray(vector, dtype=float)
    if len(vec) != len(idx):
        raise ValueError("Coefficient vector length does not match tree")
    return dict(zip(idx, vec.tolist()))


def transfer_coefficients(
    coeffs: Dict[BasisIndex, float],
    old_tree: AdaptiveHaarTree,
    new_tree: AdaptiveHaarTree,
) -> Dict[BasisIndex, float]:
    """Orthogonal transfer between adaptive Haar spaces.

    Refinement injects old coefficients and initializes new details to zero;
    coarsening drops details outside the target space.
    """
    del old_tree  # the basis keys contain all information needed here
    return {idx: coeffs.get(idx, 0.0) for idx in new_tree.basis_indices()}


def reconstruct(
    coeffs: Dict[BasisIndex, float],
    tree: AdaptiveHaarTree,
    shape: Sequence[int],
) -> np.ndarray:
    shape = tuple(int(n) for n in shape)
    out = np.zeros(shape, dtype=float)
    idxs = tree.basis_indices()
    root = idxs[0]
    out += coeffs.get(root, 0.0)
    dim = len(shape)
    for idx in idxs[1:]:
        c = coeffs.get(idx, 0.0)
        if c == 0.0:
            continue
        amplitude = 2.0 ** (dim * idx.level / 2.0)
        for bits, sl in _child_slices(shape, idx.level, idx.translation):
            out[sl] += c * amplitude * _orientation_sign(bits, idx.orientation)
    return out


def basis_geometry(tree: AdaptiveHaarTree):
    """Return centers, levels and support sizes for active basis functions."""
    centers = []
    levels = []
    sizes = []
    for idx in tree.basis_indices():
        if idx.kind == "scaling":
            centers.append((0.5,) * tree.dim)
            levels.append(0)
            sizes.append(1.0)
            continue
        h = 2.0 ** (-idx.level)
        centers.append(tuple((k + 0.5) * h for k in idx.translation))
        levels.append(idx.level)
        sizes.append(h)
    return np.asarray(centers, float), np.asarray(levels, int), np.asarray(sizes, float)


def refinement_level_map(tree: AdaptiveHaarTree, shape: Sequence[int]) -> np.ndarray:
    shape = tuple(int(n) for n in shape)
    out = np.zeros(shape, dtype=int)
    for level, tr in tree.leaves():
        out[_cell_slices(shape, level, tr)] = level
    return out


def adaptive_tree_from_detail_threshold(
    full: Dict[BasisIndex, float],
    dim: int,
    max_level: int,
    threshold: float,
    min_level: int = 1,
) -> AdaptiveHaarTree:
    """Build an admissible tree using prospective Haar detail energy."""
    tree = AdaptiveHaarTree(dim=dim, max_level=max_level)
    queue = [tree.root(dim)]
    while queue:
        cell = queue.pop(0)
        level, tr = cell
        if level >= max_level:
            continue
        # Use the total unresolved detail energy below this cell, not only
        # the immediate parent-level coefficient.  This prevents oscillatory
        # structures whose coarse Haar moments cancel from being missed.
        detail = 0.0
        for idx, value in full.items():
            if idx.kind != "wavelet" or idx.level < level:
                continue
            factor = 2 ** (idx.level - level)
            ancestor = tuple(v // factor for v in idx.translation)
            if ancestor == tr:
                detail += value * value
        must_refine = level < min_level
        if must_refine or np.sqrt(detail) >= threshold:
            tree.refined.add(cell)
            queue.extend(tree.children(cell))
    tree._enforce_ancestry()
    return tree
