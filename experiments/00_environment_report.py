#!/usr/bin/env python3
"""Archive software versions used for the numerical validation."""
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy
import scipy
import skimage

from amdi.io_utils import ensure_dir, write_json


def main():
    out = ensure_dir(ROOT / "results" / "00_environment_report")
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit_image": skimage.__version__,
    }
    try:
        import vampyr
        report["vampyr_import"] = "ok"
        report["mrcpp_version"] = str(vampyr.mrcpp_version())
    except Exception as exc:
        report["vampyr_import"] = f"unavailable: {type(exc).__name__}: {exc}"
    write_json(out / "environment.json", report)
    for k, v in report.items(): print(f"{k}: {v}")


if __name__ == "__main__":
    main()
