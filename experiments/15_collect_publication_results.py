#!/usr/bin/env python3
"""Collect the key AMDI numerical-validation outputs into publication summaries."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amdi.io_utils import ensure_dir, write_csv, write_json
from amdi.validation import read_csv


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def add_scalar_metrics(table: list[dict], section: str, data: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in data and isinstance(data[key], (int, float, bool, str)):
            table.append({"section": section, "metric": key, "value": data[key]})


def main() -> None:
    results = ROOT / "results"
    out = ensure_dir(results / "publication_summary")
    summary: dict = {}
    table: list[dict] = []

    d = load_json(results / "01_operator_properties" / "diagnostics.json")
    if d:
        summary["operator"] = d
        add_scalar_metrics(
            table,
            "operator",
            d,
            ("symmetry_error", "kernel_error", "min_eigenvalue", "spectral_gap"),
        )

    p = results / "02_refinement_consistency" / "refinement_commutator.csv"
    if p.exists():
        rows = read_csv(p)
        summary["refinement_consistency"] = rows
        if rows:
            table.append({"section": "refinement", "metric": "first_row", "value": rows[0]})
            table.append({"section": "refinement", "metric": "last_row", "value": rows[-1]})

    p = results / "03_energy_decay" / "energy_history.csv"
    if p.exists():
        rows = read_csv(p)
        summary["energy_history"] = rows
        residual_keys = [k for k in rows[0] if "residual" in k.lower()] if rows else []
        for key in residual_keys:
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            if vals:
                table.append({"section": "energy", "metric": f"max_{key}", "value": max(vals)})

    d = load_json(results / "04_synthetic_2d_adaptivity" / "metrics.json")
    if d:
        summary["synthetic_adaptivity"] = d
        add_scalar_metrics(table, "adaptivity", d, ("RMSE", "PSNR", "SSIM", "C_rel", "basis_size"))

    json_sources = [
        ("09_quality_complexity_pareto", "benchmark_assessment.json", "application_benchmark"),
        ("10_vampyr_precision_convergence", "vampyr_precision_metadata.json", "vampyr_precision"),
        ("11_amdi_debias_refit", "best_refits.json", "debias_refit"),
        ("12_robustness_convergence", "robustness_assessment.json", "robustness"),
        ("13_holdout_multiseed_benchmark", "holdout_assessment.json", "holdout"),
        ("14_vampyr_amdi_localization_crosscheck", "vampyr_tree_summary.json", "vampyr_localization"),
        ("16_matched_complexity_ablation", "matched_complexity_ablation_assessment.json", "matched_complexity_ablation"),
        ("17_standardized_initialization", "standardized_initialization_protocol.json", "standardized_initialization_protocol"),
        ("18_vampyr_precision_audit", "vampyr_precision_audit.json", "vampyr_precision_audit"),
    ]
    for exp, fname, key in json_sources:
        d = load_json(results / exp / fname)
        if d is not None:
            summary[key] = d

    csv_sources = [
        (
            "14_vampyr_amdi_localization_crosscheck",
            "regional_localization_crosscheck.csv",
            "vampyr_amdi_regional_localization",
        ),
        (
            "16_matched_complexity_ablation",
            "matched_complexity_ablation_summary.csv",
            "matched_complexity_ablation_summary",
        ),
        (
            "17_standardized_initialization",
            "standardized_initialization_summary.csv",
            "standardized_initialization_summary",
        ),
        (
            "18_vampyr_precision_audit",
            "vampyr_precision_rows_audited.csv",
            "vampyr_precision_rows_audited",
        ),
    ]
    for exp, fname, key in csv_sources:
        p = results / exp / fname
        if p.exists():
            summary[key] = read_csv(p)

    # Compact headline rows for the newly added final checks.
    d = summary.get("robustness")
    if isinstance(d, dict):
        add_scalar_metrics(
            table,
            "robustness",
            d,
            (
                "all_runs_energy_monotone",
                "mean_safeguard_accept_rate",
                "noise_RMSE_mean",
                "noise_RMSE_std",
                "noise_SSIM_mean",
                "noise_SSIM_std",
                "noise_C_rel_mean",
                "noise_C_rel_std",
            ),
        )

    d = summary.get("vampyr_precision_audit")
    if isinstance(d, dict):
        add_scalar_metrics(
            table,
            "vampyr_precision",
            d,
            (
                "complexity_nondecreasing_as_precision_tightens",
                "L2_error_nonincreasing_as_precision_tightens",
                "any_projection_hit_depth_cap",
                "empirical_log_error_vs_log_end_nodes_slope",
                "publication_ready_monotone_precision_trend",
            ),
        )

    write_json(out / "publication_summary.json", summary)
    write_csv(out / "publication_key_metrics.csv", table)
    print(f"Wrote {out / 'publication_summary.json'}")
    print(f"Wrote {out / 'publication_key_metrics.csv'}")

    required = (
        "operator",
        "refinement_consistency",
        "energy_history",
        "synthetic_adaptivity",
        "application_benchmark",
        "robustness",
        "holdout",
        "vampyr_precision",
        "vampyr_localization",
        "matched_complexity_ablation",
        "standardized_initialization_summary",
        "vampyr_precision_audit",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        print("Missing optional/not-yet-run sections:", ", ".join(missing))


if __name__ == "__main__":
    main()
