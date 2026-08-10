# Manuscript figure map

The scripts generate both 600-dpi PNG and vector PDF output.  No generated manuscript figure contains a title; panel meaning is supplied by the manuscript caption.

## Recommended main-text figures

### Figure 1 — operator and refinement structure

Use:

```text
results/01_operator_properties/operator_spectrum.pdf
results/02_refinement_consistency/refinement_commutator.pdf
```

Recommended manuscript combination:

- panel (a): intrinsic operator spectrum;
- panel (b): refinement-commutator defect.

The individual files already have publication axis labels and units; combine them in LaTeX or in a separate manuscript assembly step rather than adding graphical titles.

### Figure 2 — variational stability

Use:

```text
results/03_energy_decay/energy_decay.pdf
```

Panel ordering:

- (a) state energy versus iteration;
- (b) discrete split-energy inequality residual versus iteration.

### Figure 3 — adaptive localization

Use:

```text
results/04_synthetic_2d_adaptivity/synthetic_adaptivity.pdf
```

Panel ordering:

- (a) reference target;
- (b) adaptive projection;
- (c) leaf refinement level;
- (d) absolute reconstruction error.

### Figure 4 — quality--complexity benchmark

Use:

```text
results/09_quality_complexity_pareto/quality_complexity_pareto.pdf
```

Panel ordering:

- (a) RMSE versus relative complexity;
- (b) SSIM versus relative complexity.

If representative images are needed, use:

```text
results/09_quality_complexity_pareto/selected_reconstructions.pdf
```

Panel ordering is fixed by the script:

- (a) truth;
- (b) noisy image;
- (c) RMSE-tuned best baseline;
- (d) SSIM-tuned best baseline;
- (e) AMDI best-RMSE operating point;
- (f) AMDI compressed operating point.

### Figure 5 — convergence robustness and initialization

Use:

```text
results/12_robustness_convergence/robustness_convergence.pdf
results/17_standardized_initialization/standardized_initialization_sensitivity.pdf
```

The robustness figure contains:

- (a) normalized energy;
- (b) relative iterate update;
- (c) adaptive relative complexity.

The standardized-initialization figure contains:

- (a) RMSE versus prescribed initial complexity;
- (b) SSIM versus prescribed initial complexity;
- (c) final relative complexity versus prescribed initial complexity.

### Figure 6 — matched-complexity ablation

Use:

```text
results/16_matched_complexity_ablation/matched_complexity_metrics.pdf
```

Panel ordering:

- (a) RMSE mean ± standard deviation;
- (b) SSIM mean ± standard deviation.

The corresponding reconstruction montage is:

```text
results/16_matched_complexity_ablation/matched_complexity_ablation.pdf
```

Panel identities are described in the manuscript caption rather than written in the image.

## Recommended VAMPyR figure

Use:

```text
results/14_vampyr_amdi_localization_crosscheck/vampyr_amdi_localization_crosscheck.pdf
```

Panel ordering:

- (a) AMDI-Haar local refinement level;
- (b) VAMPyR adaptive end-node grid;
- (c) regional mean-resolution comparison.

For a convergence-oriented VAMPyR figure use:

```text
results/18_vampyr_precision_audit/vampyr_precision_audit.pdf
```

Panel ordering:

- (a) requested precision versus end-node count;
- (b) `L2` reference error versus end-node count.

## Development/supplementary figures

The following are retained for transparency and regression testing but are not necessarily required in the main paper:

```text
results/05_denoising_benchmark/denoising_comparison.pdf
results/06_ablation_study/ablation.pdf
results/07_vampyr_projection_check/vampyr_adaptive_grid.pdf
results/08_amdi_parameter_sweep/amdi_parameter_sweep.pdf
results/08_amdi_parameter_sweep/amdi_selected_operating_points.pdf
results/10_vampyr_precision_convergence/vampyr_precision_convergence.pdf
results/11_amdi_debias_refit/debias_refit_metrics.pdf
results/11_amdi_debias_refit/debias_refit_reconstructions.pdf
results/12_robustness_convergence/parameter_sensitivity.pdf
results/13_holdout_multiseed_benchmark/holdout_multiseed_summary.pdf
```

## Figure formatting rule

Do not add titles to the exported PNG/PDF files.  Use manuscript captions to identify the scientific content of panels.  If a final journal layout requires different panel grouping, combine the title-free PDF files in LaTeX rather than rasterizing them again.
