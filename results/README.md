# Generated results

All experiment outputs are written under this directory.  Generated CSV, JSON, PNG, and PDF files are intentionally excluded from Git by default.

For a complete publication run:

```bash
python experiments/run_publication.py --sweep-budget 192
```

The compact final summaries are written to:

```text
results/publication_summary/publication_summary.json
results/publication_summary/publication_key_metrics.csv
```

For a manuscript release, archive the complete generated `results/` directory separately and associate it with the exact Git commit/tag used for the calculation.
