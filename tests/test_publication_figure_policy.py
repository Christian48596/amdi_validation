"""Regression tests for manuscript-figure formatting policy."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def _figure_sources():
    for path in sorted(EXPERIMENTS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "plt.subplots" in text or "plt.figure" in text:
            yield path, text


def test_no_embedded_titles_in_manuscript_figures():
    forbidden = (".set_title(", ".suptitle(", "plt.title(")
    offenders = []
    for path, text in _figure_sources():
        if any(token in text for token in forbidden):
            offenders.append(path.name)
    assert not offenders, f"Embedded figure titles found in: {offenders}"


def test_all_figures_use_publication_save_helper():
    offenders = []
    for path, text in _figure_sources():
        if "save_publication_figure(" not in text:
            offenders.append(path.name)
        if "dpi=300" in text:
            offenders.append(path.name + " (300 dpi)")
    assert not offenders, f"Non-publication figure output policy in: {offenders}"


def test_direct_axis_labels_declare_units():
    """Every literal direct x/y label must include a bracketed unit marker.

    Spatial-image labels are supplied centrally by ``show_normalized_image``
    and therefore do not appear as direct calls in those scripts.
    """
    offenders = []
    pattern = re.compile(r"set_(?:x|y)label\((?:r|f|rf|fr)?[\"']([^\"']+)[\"']")
    for path, text in _figure_sources():
        for label in pattern.findall(text):
            if "[" not in label or "]" not in label:
                offenders.append(f"{path.name}: {label}")
    assert not offenders, "Axis labels without explicit units: " + "; ".join(offenders)
