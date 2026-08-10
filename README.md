# AMDI numerical validation suite

Reproducible numerical validation for the manuscript on **adaptive multiresolution diffusion operators (AMDI)**.  The repository is organized to test the mathematical structure of the intrinsic operator, the variational evolution, adaptive compression, application performance, robustness, and an independent VAMPyR/MRCPP multiresolution cross-check.

The code is intended to be readable enough that every numerical claim in the manuscript can be traced to a specific script and output file.

## Scope

The validation is divided into five layers:

1. **Operator structure** — symmetry, positive semidefiniteness, null mode, spectral gap.
2. **Variational evolution** — refinement consistency, exact split-scheme energy inequality, convergence diagnostics.
3. **Adaptive representation** — localization of refinement and compression on a controlled multiscale target.
4. **Application benchmark** — denoising, quality--complexity Pareto analysis, holdout evaluation, matched-complexity ablation, standardized initialization.
5. **Independent multiwavelet check** — VAMPyR/MRCPP adaptive `FunctionTree` projection, precision--complexity convergence, and regional localization.

The AMDI coefficient algebra uses an exact adaptive tensor-product Haar backend.  Haar is orthonormal, so the Gram matrix is `G = I`; this keeps the coefficient operator, transfer operators, and discrete energy checks transparent.  VAMPyR/MRCPP is used independently to verify adaptive multiwavelet behavior with a production multiwavelet implementation.

## Repository layout

```text
amdi_validation/
├── amdi/
│   ├── benchmarks.py
│   ├── energy.py
│   ├── graph.py
│   ├── haar.py
│   ├── io_utils.py
│   ├── metrics.py
│   ├── pareto.py
│   ├── plotting.py
│   ├── solver.py
│   ├── synthetic.py
│   ├── validation.py
│   └── vampyr_adapter.py
├── experiments/
│   ├── 00_environment_report.py
│   ├── 01_operator_properties.py
│   ├── 02_refinement_consistency.py
│   ├── 03_energy_decay.py
│   ├── 04_synthetic_2d_adaptivity.py
│   ├── 05_denoising_benchmark.py
│   ├── 06_ablation_study.py
│   ├── 07_vampyr_projection_check.py
│   ├── 08_amdi_parameter_sweep.py
│   ├── 09_quality_complexity_pareto.py
│   ├── 10_vampyr_precision_convergence.py
│   ├── 11_amdi_debias_refit.py
│   ├── 12_robustness_convergence.py
│   ├── 13_holdout_multiseed_benchmark.py
│   ├── 14_vampyr_amdi_localization_crosscheck.py
│   ├── 15_collect_publication_results.py
│   ├── 16_matched_complexity_ablation.py
│   ├── 17_standardized_initialization.py
│   ├── 18_vampyr_precision_audit.py
│   ├── run_all.py
│   └── run_publication.py
├── tests/
├── results/
│   └── .gitkeep
├── .github/workflows/tests.yml
├── .gitignore
├── CHANGELOG.md
├── GITHUB_RELEASE_CHECKLIST.md
├── MANUSCRIPT_FIGURES.md
├── PUBLICATION_PROTOCOL.md
├── Makefile
├── environment.yml
└── pyproject.toml
```

## Installation

### Recommended Conda environment

```bash
conda env create -f environment.yml
conda activate amdi-vampyr
pip install -e .
```

For an existing environment:

```bash
conda activate amdi-vampyr
pip install -e .
```

On a headless Linux/HPC system, use:

```bash
export MPLBACKEND=Agg
```

### Unit tests

```bash
pytest -q
```

The test suite covers the Haar representation, transfer/reconstruction, Parseval identity, graph-Laplacian structure, energy descent, Pareto utilities, fixed-tree refitting, convergence diagnostics, the VAMPyR adapter interface, and the manuscript-figure formatting policy.

The GitHub Actions workflow runs the unit tests on Python 3.10, 3.11, and 3.12 without requiring VAMPyR.

## Quick validation

Run experiments 00--07:

```bash
python experiments/run_all.py
```

or:

```bash
make structural
```

For a faster development version of the complete workflow:

```bash
python experiments/run_all.py --extended --sweep-budget 48
```

## Final publication workflow

The complete dependency-aware run is:

```bash
python experiments/run_publication.py --sweep-budget 192
```

The default publication configuration uses:

- eight unseen holdout noise realizations;
- five seeds for the final matched-complexity and initialization checks;
- a memory-safe VAMPyR depth cap of 8;
- a VAMPyR reference precision of `1e-5`.

A faster pre-publication run is:

```bash
python experiments/run_publication.py \
    --sweep-budget 48 \
    --holdout-seeds 101,103,107,109 \
    --robustness-max-iterations 12
```

The publication runner executes the final checks in dependency order and runs `15_collect_publication_results.py` last, so the collected summary contains experiments 16--18 as well.

## Experiment map

| ID | Script | Purpose |
|---:|---|---|
| 00 | `00_environment_report.py` | Archive Python/package/MRCPP versions. |
| 01 | `01_operator_properties.py` | Symmetry, PSD, kernel, spectral gap. |
| 02 | `02_refinement_consistency.py` | Refinement-commutator defect on nested spaces. |
| 03 | `03_energy_decay.py` | Exact energy decay and discrete inequality residual. |
| 04 | `04_synthetic_2d_adaptivity.py` | Controlled 2D localization and compression. |
| 05 | `05_denoising_benchmark.py` | Compact regression/development denoising benchmark. |
| 06 | `06_ablation_study.py` | Original full/fixed/no-diffusion/linear-weight comparison. |
| 07 | `07_vampyr_projection_check.py` | VAMPyR adaptive `FunctionTree` geometry. |
| 08 | `08_amdi_parameter_sweep.py` | AMDI calibration and Pareto sampling. |
| 09 | `09_quality_complexity_pareto.py` | Fair RMSE/SSIM baseline tuning and quality--complexity comparison. |
| 10 | `10_vampyr_precision_convergence.py` | Memory-safe VAMPyR precision--complexity convergence. |
| 11 | `11_amdi_debias_refit.py` | Fixed-tree coefficient refit diagnostic. |
| 12 | `12_robustness_convergence.py` | Parameter, time-step, noise, convergence, and safeguard robustness. |
| 13 | `13_holdout_multiseed_benchmark.py` | Frozen-parameter evaluation on unseen noise seeds. |
| 14 | `14_vampyr_amdi_localization_crosscheck.py` | AMDI-Haar/VAMPyR regional localization comparison. |
| 16 | `16_matched_complexity_ablation.py` | Ablation on exactly the same adaptive tree/complexity. |
| 17 | `17_standardized_initialization.py` | Initialization by prescribed initial relative complexity. |
| 18 | `18_vampyr_precision_audit.py` | Audit monotonic VAMPyR precision/error/complexity trend. |
| 15 | `15_collect_publication_results.py` | Collect all final JSON/CSV publication metrics. |

Experiment 15 is numbered earlier for historical continuity but is intentionally executed **last**.

## VAMPyR/MRCPP role

VAMPyR is not used merely to draw an adaptive grid.  The VAMPyR experiments construct a real `MultiResolutionAnalysis`, apply adaptive `ScalingProjector`s, create `FunctionTree` representations, inspect end-node geometry, and measure an `L2` distance between compatible adaptive trees.

The current AMDI graph operator remains explicitly implemented in `amdi/graph.py`.  The repository does not depend on undocumented access to raw VAMPyR node-coefficient arrays.

### Memory safety

Very tight 2D projection tolerances near sharp/discontinuous features can generate extremely large adaptive trees.  The final VAMPyR experiments therefore expose and use an explicit `max_depth` cap.  The precision-convergence target is smooth but multiscale, while the separate localization experiment retains edge/texture structure.

## Manuscript figure policy

All figure-producing experiments use `amdi.plotting.save_publication_figure` and follow the same policy:

- **no embedded figure titles or panel titles**;
- panels are identified only by `(a)`, `(b)`, ...;
- every plotted axis includes a unit in square brackets;
- dimensionless quantities use `[-]`;
- integer node counts use `[count]`;
- spatial images use normalized coordinates `x [-]` and `y [-]`;
- PNG files are written at **600 dpi**;
- a vector PDF is written in parallel;
- the manuscript caption supplies the scientific meaning of each panel.

A unit test prevents accidental reintroduction of titles, 300-dpi outputs, or literal axis labels without units.

See `MANUSCRIPT_FIGURES.md` for the recommended final figure set and panel ordering.

## Results and reproducibility

Generated numerical outputs are written under `results/` and are intentionally excluded from Git by default.  This prevents machine-specific generated files and large image/PDF collections from obscuring the source code history.

After a complete run, the compact final summaries are:

```text
results/publication_summary/publication_summary.json
results/publication_summary/publication_key_metrics.csv
```

If a release should archive a particular manuscript dataset, tag the source-code commit and attach the corresponding `results/` archive to the GitHub Release rather than committing all generated files to the main branch.

## Interpretation rules

The scripts are written to measure claims, not to force them.  In particular:

- do not claim refinement consistency unless experiment 02 shows a decreasing defect;
- do not claim theorem-level decay for a frozen-weight step without the energy safeguard;
- do not claim universal RMSE/SSIM superiority unless experiments 09 and 13 support it;
- report the AMDI accuracy--complexity tradeoff rather than only one metric;
- report initialization sensitivity in terms of the standardized initial complexity `C_rel^0`;
- interpret experiment 16 at matched complexity: it isolates coefficient/operator effects after the adaptive tree has been selected;
- treat the VAMPyR log--log slope as descriptive, not as a proved convergence order.

The intended numerical message is a reproducible **stability--structure--complexity tradeoff**.

## Development conventions

- Keep new numerical claims attached to explicit CSV/JSON output.
- Add a unit test for any new core invariant.
- Do not hard-code manuscript conclusions in numerical scripts.
- Use `amdi/plotting.py` for all new manuscript figures.
- Do not commit `results/`, caches, Conda environments, or generated binary files.

## GitHub upload

Before the first public push, follow `GITHUB_RELEASE_CHECKLIST.md`.  A minimal sequence is:

```bash
git init
git add .
git commit -m "Initial AMDI numerical validation release"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Choose and add the appropriate software license before making the repository public.
