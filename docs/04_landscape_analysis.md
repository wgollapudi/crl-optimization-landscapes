# Landscape Analysis

`analyze_landscapes.py` consumes `landscape_runs/` and writes analysis outputs:

```text
landscape_analysis/
  tables/
  figures/
```

The common command is:

```bash
cd src
python analyze_landscapes.py \
  --landscape-root landscape_runs \
  --outdir landscape_analysis \
  --checkpoint-kind best_val \
  --splits val
```

The script uses no pandas. It writes wide CSV tables using Python's `csv`
module and NumPy. Figures are produced with matplotlib when available; if
matplotlib is missing, tables are still written.

## Main Tables

`seed_table.csv` has one row per run/checkpoint/split and merges local probe
artifacts.

`pair_table.csv` has one row per pair/checkpoint/split and stores interpolation
descriptors.

`seed_summary.csv` and `pair_summary.csv` summarize numeric columns by regime,
checkpoint kind, and split.

`regime_differences.csv` compares every non-control regime against Regime A.
These bootstrap intervals are descriptive summaries, not formal claims of
independent sampling for pair rows.

`missing_artifacts.csv` records missing, malformed, or empty artifacts without
stopping analysis.

`hessian_status_counts.csv` summarizes Hessian status fields.

## Interpretation

Cross-regime comparisons should emphasize component metrics, especially
`recon_img`, because total objectives differ across regimes. The newer models
add theorem-motivated constraints to the loss, so total loss is descriptive
context rather than the primary apples-to-apples metric.

