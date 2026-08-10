"""Shared publication-figure utilities for the AMDI validation suite.

The manuscript figure policy is intentionally strict:

* no figure or panel titles are embedded in the graphics;
* panels are identified only by ``(a)``, ``(b)``, ... labels;
* every plotted axis carries an explicit unit (``[-]`` for dimensionless
  quantities and ``[count]`` for integer counts);
* spatial images use normalized coordinates on ``[0, 1]^2``;
* raster output is written at 600 dpi and vector PDF output is written in
  parallel.

The manuscript caption, not text baked into the image, should identify the
meaning of each panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def add_panel_label(ax, label: str, *, x: float = 0.02, y: float = 0.98) -> None:
    """Place a conventional manuscript panel label inside an axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
    )


def add_panel_labels(axes: Iterable, labels: Iterable[str] | None = None) -> None:
    """Add sequential panel labels to a collection of axes."""
    axes = list(axes)
    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(axes))]
    for ax, label in zip(axes, labels):
        add_panel_label(ax, label)


def show_normalized_image(
    ax,
    image,
    *,
    cmap: str = "gray",
    vmin=None,
    vmax=None,
    interpolation: str = "nearest",
):
    """Display an image on normalized spatial coordinates with unit labels.

    The image orientation follows the usual matrix/image convention: the first
    row appears at the top.  The numerical coordinate labels are nevertheless
    normalized to the unit square for a resolution-independent manuscript
    presentation.
    """
    im = ax.imshow(
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        extent=(0.0, 1.0, 1.0, 0.0),
        aspect="equal",
    )
    ax.set_xlabel(r"$x$ [-]")
    ax.set_ylabel(r"$y$ [-]")
    return im


def save_publication_figure(fig, out_dir: str | Path, stem: str, *, dpi: int = 600) -> None:
    """Write manuscript-ready PNG and PDF versions of a figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
