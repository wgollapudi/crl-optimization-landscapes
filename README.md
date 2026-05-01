# Identifiability and CRL Loss Landscapes

This repository builds a controlled experimental pipeline for studying how
theorem-backed identifiability assumptions change the empirical optimization
landscape of causal representation learning models.

The experiment compares four regimes:

| Regime | Dataset | Model | Identifiability role |
| --- | --- | --- | --- |
| A | `observational.npz` | Plain VAE | Non-identifiable observational baseline |
| B | `overlap_support.npz` | Sparse VAE | Statistical identifiability via engineered anchor features |
| C | `single_node_interventions.npz` | CausalDiscrepancy VAE | Causal structure from one intervention per latent node |
| D | `two_interventions_per_node.npz` | CausalDiscrepancy VAE | Stronger intervention coverage; optionally anchor-augmented |

The core question is not which architecture wins. The goal is to ask whether
reducing the model's indeterminacy class changes measurable geometry:
local sharpness, curvature, basin width, pairwise interpolation barriers, and
seed-to-seed stability.

## Quick Start

Generate datasets:

```bash
cd src
./build_datasets.sh --overwrite
```

Train one run. The model is inferred from the dataset filename:

```bash
python train.py --data-path data/crl_dsprites/observational.npz
python train.py --data-path data/crl_dsprites/overlap_support.npz
python train.py --data-path data/crl_dsprites/single_node_interventions.npz
python train.py --data-path data/crl_dsprites/two_interventions_per_node.npz
```

Run landscape probes for one regime:

```bash
python landscape.py \
  --regime regimeA \
  --checkpoint-kind best_val \
  --probes endpoint perturbation slice1d pairwise \
  --max-runs 20
```

Analyze saved probe artifacts:

```bash
python analyze_landscapes.py \
  --landscape-root landscape_runs \
  --outdir landscape_analysis \
  --checkpoint-kind best_val \
  --splits val
```

## Workflow

The pipeline has four stages:

1. `make_datasets.py` creates theorem-facing datasets from a synthetic SCM and
   the dSprites renderer table.
2. `train.py` trains forests of models, one seed per run directory.
3. `landscape.py` reloads checkpoints and emits deterministic probe artifacts.
4. `analyze_landscapes.py` turns probe artifacts into wide CSV tables,
   summaries, Regime A comparisons, and figures when matplotlib is available.

## Design Decisions Worth Knowing

The dSprites images are not treated as a causal data-generating process.
Instead, dSprites is used as a renderer table. Continuous SCM latents are
sampled first, then quantized to valid dSprites factors, then rendered by lookup.
This gives us latent causal structure and intervention labels while keeping the
visual data simple.

Anchor features are engineered on purpose. Raw dSprites pixels do not satisfy
the sparse-decoding anchor assumptions, so the generated datasets append scalar
features with two anchors per latent coordinate. This makes Regime B
theorem-facing rather than merely heuristic.

CD-VAE does not consume the true `intervention_target` as model input. It uses
`env_id`, while `intervention_target` remains metadata for auditing and later
analysis. This avoids leaking the object the estimator is supposed to recover.

Landscape evaluation is deterministic. During probes the VAE forward pass uses
`z = mu(x)` instead of sampling from the posterior. This keeps "loss at theta"
well-defined and avoids stochastic roughness from reparameterization noise.

Interpolation is only intra-regime. We interpolate checkpoints only when they
share the same architecture and parameter layout. Cross-regime comparison is
done by comparing distributions of scalar descriptors.

Total loss is not the main cross-regime object. Sparse VAE and CD-VAE add
different constraints and auxiliary losses, so cross-regime plots should
emphasize component metrics such as `recon_img`, Hessian summaries for
`recon_img`, and interpolation barriers for `recon_img`.

Hessian estimates are treated as delicate diagnostics. The probe artifacts keep
status fields, raw restart/sample summaries, and trace standard errors. The
analysis script does not discard high-uncertainty Hessians; it flags them.

## Documentation Map

- [Project overview](docs/00_project_overview.md)
- [Data generation](docs/01_data_generation.md)
- [Training](docs/02_training.md)
- [Landscape probes](docs/03_landscape_probes.md)
- [Landscape analysis](docs/04_landscape_analysis.md)
- [Artifact schema](docs/05_artifact_schema.md)
- [Scientific assumptions](docs/06_scientific_assumptions.md)
- [CLI reference](docs/cli_reference.md)

