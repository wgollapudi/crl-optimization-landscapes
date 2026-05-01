# CLI Reference

This file lists the parameters that must be passed today after the ergonomic
defaults in the code.

## `build_datasets.sh`

Required parameters: none.

Preferred command:

```bash
cd src
./build_datasets.sh --overwrite
```

Important options:

```text
--root DIR            default: data
--shape SHAPE         default: ellipse
--seed INT            default: 0
--generator PATH      default: sibling make_datasets.py
--force-download      default: false
--overwrite           required only when output dir is non-empty
```

## `make_datasets.py`

Required parameters: none, if using the standard paths.

Default command:

```bash
cd src
python make_datasets.py
```

Defaults:

```text
--dsprites data/raw/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
--outdir data/crl_dsprites
--shape ellipse
--seed 0
--n-obs 40000
--n-per-intervention 5000
--overlap-grid-budget 20000
--train-frac 0.8
--val-frac 0.1
--adjacency default SCM
--noise-scales default SCM
--clip 1.0
--anchors-per-latent 2
--anchor-noise-std 0.0
```

## `train.py`

Required parameters:

```text
--data-path
```

The model is inferred from the dataset stem when `--model-name` is omitted.
Sparse VAE also auto-enables anchor features.

Examples:

```bash
python train.py --data-path data/crl_dsprites/observational.npz
python train.py --data-path data/crl_dsprites/overlap_support.npz
python train.py --data-path data/crl_dsprites/single_node_interventions.npz
python train.py --data-path data/crl_dsprites/two_interventions_per_node.npz
```

Important defaults:

```text
--model-name inferred from dataset filename
--latent-dim 4
--hidden-dim 256
--anchor-dim 0, or 2 * latent_dim for sparse/anchored runs
--epochs 100
--batch-size 128
--lr 1e-3
--beta-warmup-epochs 10
--seed 0
--outdir auto
--device cuda if available, else cpu
```

## `landscape.py`

Required parameters:

```text
--regime
```

`--run-root` defaults to `runs/<regime>`.
`--outdir` defaults to `landscape_runs`.
`--data-path` is inferred from selected run `config.log` files when possible.

Example:

```bash
python landscape.py \
  --regime regimeA \
  --checkpoint-kind best_val \
  --probes endpoint perturbation slice1d pairwise \
  --max-runs 20
```

Important defaults:

```text
--checkpoint-kind all
--probes all
--selection all
--use-anchor-features auto
--train-subset-n 5000
--val-subset-n 5000
--subset-seed 0
--radii 1e-4 3e-4 1e-3 3e-3 1e-2
--directions-per-radius 20
--slice-points 41
--interp-points 41
--max-pairs 100
--grad-components total recon_img
--hessian-components total recon_img
--hessian-power-iters 20
--hessian-power-restarts 1
--hessian-trace-samples 20
--hessian-max-batches 3
--slice2d-max-runs 3
```

Checkpoint kinds:

```text
start       start.pt
mid_best    mid_best.pt
final       final.pt, with fallback to last.pt
best_val    best_val_loss.pt, with fallback to best_loss.pt
all         start, mid_best, final, best_val
```

## `analyze_landscapes.py`

Required parameters: none.

Default command:

```bash
python analyze_landscapes.py
```

Important defaults:

```text
--landscape-root landscape_runs
--outdir landscape_analysis
--regimes regimeA regimeB regimeC regimeD
--checkpoint-kind best_val
--splits val
--control-regime regimeA
--bootstrap-samples 2000
--ci-level 0.95
--make-figures
```

## `scripts/smoke_pipeline.sh`

Required parameters: none.

This is a tiny proof-of-concept pipeline that writes to a disposable output
directory and exercises all stages:

```text
tiny dataset generation
two seeds per regime
one or more epochs of training
endpoint / perturbation / slice1d / slice2d / gradnorm / hessian / pairwise probes
landscape analysis tables
```

Default command:

```bash
scripts/smoke_pipeline.sh
```

Useful faster command:

```bash
scripts/smoke_pipeline.sh --outdir /tmp/crl_landscape_poc --epochs 1
```

Important defaults:

```text
--outdir smoke_runs/poc_<timestamp>
--dsprites src/data/raw/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
--seeds 2
--epochs 2
--device cpu
--figures disabled unless requested
```
