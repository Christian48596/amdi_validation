# Changelog

## v0.4.0 — GitHub/publication-ready validation package

- Added experiments 16--18 to the permanent validation suite:
  - matched-complexity ablation;
  - standardized initialization by prescribed initial relative complexity;
  - VAMPyR precision-convergence audit.
- Updated `run_publication.py` so the complete final workflow executes experiments 00--18 in dependency order and collects results last.
- Updated the publication collector to include the matched-complexity, standardized-initialization, and VAMPyR-audit outputs.
- Added a memory-safe VAMPyR `max_depth` interface to the adapter and publication runner.
- Added `amdi/plotting.py` with a strict manuscript figure policy.
- Removed all embedded figure/panel titles from generated manuscript graphics.
- Added explicit units to every plotted axis (`[-]` for dimensionless quantities, `[count]` for node counts, normalized `x [-]`, `y [-]` for image coordinates).
- Standardized all manuscript PNG output to 600 dpi with vector PDF output in parallel.
- Expanded the exact energy-dissipation figure to include the discrete energy-inequality residual.
- Added automated tests enforcing the figure policy and the VAMPyR depth-cap API.
- Added GitHub Actions unit-test CI for Python 3.10--3.12.
- Added `MANUSCRIPT_FIGURES.md`, `GITHUB_RELEASE_CHECKLIST.md`, and a reproducibility-focused final README.

## v0.3.1 — VAMPyR memory-safety patch

- Added an explicit `max_depth` to VAMPyR MRA construction.
- Replaced the pathological tight-tolerance discontinuous precision test by a smooth multiscale 2D convergence target.
- Preserved the edge/texture target for the separate localization cross-check.

## v0.3.0 — robustness/publication package

- Added convergence diagnostics to the frozen-weight AMDI solver: relative update, relative energy change, tree distance, safeguard factor/backtracks and monotonicity.
- Added optional convergence stopping with patience.
- Added exact fixed-tree frozen-weight debias/refit at unchanged adaptive complexity.
- Updated the quality--complexity benchmark so every classical method is tuned separately for RMSE and SSIM.
- Added experiments 11--15: coefficient refit, robustness, holdout benchmark, VAMPyR localization, and publication collector.
- Expanded the VAMPyR role from adaptive-grid display to `FunctionTree` projection, precision--complexity convergence, and independent localization validation.

## v0.2.0

- Added AMDI parameter sweep, Pareto analysis, and VAMPyR precision convergence.
