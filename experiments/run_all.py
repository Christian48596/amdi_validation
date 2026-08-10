#!/usr/bin/env python3
"""Run the AMDI validation suite.

Default mode runs the compact structural/development validation (00--07).
``--extended`` adds calibration, application, robustness, holdout, VAMPyR,
matched-complexity, and standardized-initialization checks (08--18) and then
collects the publication summary.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

HERE = Path(__file__).resolve().parent

BASIC = [
    "00_environment_report.py",
    "01_operator_properties.py",
    "02_refinement_consistency.py",
    "03_energy_decay.py",
    "04_synthetic_2d_adaptivity.py",
    "05_denoising_benchmark.py",
    "06_ablation_study.py",
    "07_vampyr_projection_check.py",
]


def run(script: str, extra=None) -> None:
    print("\n" + "=" * 78)
    print("RUNNING", script)
    print("=" * 78)
    cmd = [sys.executable, str(HERE / script)] + list(extra or [])
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--sweep-budget", type=int, default=48)
    args = parser.parse_args()

    for script in BASIC:
        run(script)

    if args.extended:
        run("08_amdi_parameter_sweep.py", ["--budget", str(args.sweep_budget)])
        for script in (
            "09_quality_complexity_pareto.py",
            "10_vampyr_precision_convergence.py",
            "11_amdi_debias_refit.py",
            "12_robustness_convergence.py",
            "13_holdout_multiseed_benchmark.py",
            "14_vampyr_amdi_localization_crosscheck.py",
            "16_matched_complexity_ablation.py",
            "17_standardized_initialization.py",
            "18_vampyr_precision_audit.py",
            "15_collect_publication_results.py",
        ):
            run(script)


if __name__ == "__main__":
    main()
