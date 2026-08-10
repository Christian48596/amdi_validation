#!/usr/bin/env python3
"""Recommended end-to-end publication protocol for the AMDI manuscript.

The runner executes the complete validation chain in dependency order.  The
parameter sweep calibrates the AMDI operating points, later experiments consume
those frozen calibration files, and the publication collector is executed last.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(name: str, *args) -> None:
    print("\n" + "=" * 80)
    print("PUBLICATION RUN:", name)
    print("=" * 80)
    subprocess.run([sys.executable, str(HERE / name), *map(str, args)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-budget", type=int, default=192)
    parser.add_argument(
        "--holdout-seeds",
        default="101,103,107,109,113,127,131,137",
        help="Comma-separated unseen noise seeds used only after calibration.",
    )
    parser.add_argument("--robustness-max-iterations", type=int, default=20)
    parser.add_argument(
        "--final-check-seeds",
        default="11,17,23,31,43",
        help="Seeds for matched-complexity ablation and standardized initialization.",
    )
    parser.add_argument(
        "--vampyr-max-depth",
        type=int,
        default=8,
        help="Depth cap used by the memory-safe VAMPyR 2D experiments.",
    )
    parser.add_argument(
        "--vampyr-reference-precision",
        type=float,
        default=1.0e-5,
        help="Reference precision for the depth-capped VAMPyR convergence audit.",
    )
    args = parser.parse_args()

    for name in (
        "00_environment_report.py",
        "01_operator_properties.py",
        "02_refinement_consistency.py",
        "03_energy_decay.py",
        "04_synthetic_2d_adaptivity.py",
        "05_denoising_benchmark.py",
        "06_ablation_study.py",
        "07_vampyr_projection_check.py",
    ):
        run(name)

    run("08_amdi_parameter_sweep.py", "--budget", args.sweep_budget)
    run("09_quality_complexity_pareto.py")

    # Memory-safe VAMPyR convergence run.
    run(
        "10_vampyr_precision_convergence.py",
        "--max-depth",
        args.vampyr_max_depth,
        "--reference-precision",
        args.vampyr_reference_precision,
    )

    run("11_amdi_debias_refit.py")
    run("12_robustness_convergence.py", "--max-iterations", args.robustness_max_iterations)
    run("13_holdout_multiseed_benchmark.py", "--seeds", args.holdout_seeds)
    run("14_vampyr_amdi_localization_crosscheck.py", "--vampyr-max-depth", args.vampyr_max_depth)

    # Final targeted checks used to close the publication validation.
    run(
        "16_matched_complexity_ablation.py",
        "--noise-seeds",
        args.final_check_seeds,
        "--outer-iterations",
        12,
    )
    run(
        "17_standardized_initialization.py",
        "--initial-budgets",
        "0.10,0.12,0.14",
        "--noise-seeds",
        args.final_check_seeds,
        "--max-iterations",
        12,
    )
    run("18_vampyr_precision_audit.py")

    # Must be last so the collected summary includes experiments 16--18.
    run("15_collect_publication_results.py")


if __name__ == "__main__":
    main()
