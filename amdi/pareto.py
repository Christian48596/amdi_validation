"""Pareto-front and benchmark-selection utilities for AMDI experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def pareto_front(rows: Iterable[dict], x_key: str, y_key: str, *, minimize_y: bool = True) -> list[dict]:
    """Return nondominated rows for smaller ``x`` and better ``y``.

    ``x`` is always minimized (for AMDI this is relative complexity).  ``y``
    is minimized for error metrics such as RMSE and maximized for quality
    metrics such as SSIM.  Failed/non-finite rows are ignored.
    """
    cleaned = []
    for row in rows:
        try:
            x = float(row[x_key])
            y = float(row[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cleaned.append(row)

    out = []
    for i, row in enumerate(cleaned):
        xi, yi = float(row[x_key]), float(row[y_key])
        dominated = False
        for j, other in enumerate(cleaned):
            if i == j:
                continue
            xj, yj = float(other[x_key]), float(other[y_key])
            if minimize_y:
                weak = xj <= xi and yj <= yi
                strict = xj < xi or yj < yi
            else:
                weak = xj <= xi and yj >= yi
                strict = xj < xi or yj > yi
            if weak and strict:
                dominated = True
                break
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda r: float(r[x_key]))


def best_under_budgets(rows: Iterable[dict], budgets=(0.05, 0.10, 0.20, 0.40, 1.0), metric="RMSE", minimize=True) -> list[dict]:
    """Select the best successful row under each relative-complexity budget."""
    rows = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    selected = []
    for budget in budgets:
        feasible = []
        for row in rows:
            try:
                c = float(row["C_rel"])
                v = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(c) and np.isfinite(v) and c <= budget + 1.0e-15:
                feasible.append(row)
        if not feasible:
            continue
        best = min(feasible, key=lambda r: float(r[metric])) if minimize else max(feasible, key=lambda r: float(r[metric]))
        item = dict(best)
        item["complexity_budget"] = float(budget)
        selected.append(item)
    return selected
