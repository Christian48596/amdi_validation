# GitHub release checklist

Use this checklist before making the AMDI numerical-validation repository public.

## Source hygiene

- [ ] `pytest -q` passes.
- [ ] `git status` contains no generated `results/` files.
- [ ] No Conda environment, cache, `.DS_Store`, or editor metadata is staged.
- [ ] No private paths, usernames, API keys, tokens, or credentials are present.
- [ ] `README.md` matches the final repository structure.
- [ ] `PUBLICATION_PROTOCOL.md` matches the manuscript numerical protocol.
- [ ] `MANUSCRIPT_FIGURES.md` matches the final panel ordering.

## Figure policy

- [ ] No exported figure contains a title.
- [ ] Every plotted axis declares its unit.
- [ ] Dimensionless variables use `[-]`.
- [ ] Spatial axes use normalized coordinates `x [-]`, `y [-]`.
- [ ] PNG output is 600 dpi.
- [ ] PDF output is generated in parallel.

These requirements are also checked automatically by the unit suite.

## Reproducibility

- [ ] Record the final Git commit hash used for the manuscript.
- [ ] Run `python experiments/00_environment_report.py` on the publication machine.
- [ ] Run the complete publication workflow or document explicitly which existing run is being archived.
- [ ] Preserve `publication_summary.json` and `publication_key_metrics.csv` with the manuscript data archive.
- [ ] Preserve the VAMPyR/MRCPP version used for the final run.

## Licensing and citation

- [ ] Choose a software license before the repository becomes public.
- [ ] Add a `LICENSE` file appropriate to the project and coauthor/institutional requirements.
- [ ] Add a `CITATION.cff` once the manuscript author list, repository URL, and preferred software citation are final.

## Initial push

```bash
git init
git add .
git commit -m "Initial AMDI numerical validation release"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## Manuscript release

For the manuscript-associated code snapshot, create a tagged release, for example:

```bash
git tag -a v0.4.0 -m "AMDI manuscript numerical validation"
git push origin v0.4.0
```

Attach the complete generated results archive to the GitHub Release if the journal/data-availability policy permits it.  Do not commit large machine-generated result directories to the main development history.
