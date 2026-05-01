# Landscape Analysis Interface

## Core Abstraction

Run = one trained model checkpoint directory:

```
runs/<regime>/<run_id>/
  config.log
  last.pt
  best_val_loss.pt
```

All analysis is defined over collections of runs.

---

## Command Interface

### Basic Usage

```bash
python landscape.py \
  --regime <name> \
  --run-root runs/<regime> \
  --outdir landscape_runs \
  --data-path data/<dataset>.npz \
  --checkpoint-kind best_val \
  --max-runs 20 \
  --selection random \
  --probes all
```

---

## Run Selection

Runs are automatically discovered from `--run-root`.

### Options

```text
--max-runs <int>           # limit number of runs
--selection <mode>         # selection strategy
```

### Selection Modes

```text
all       use all discovered runs
random    uniform random subset
first     earliest runs (sorted)
last      latest runs (sorted)
```

---

## Probe Groups

Probes are grouped by level.

### Local (single-run)

```text
endpoint metrics
perturbation sharpness
1D loss slices
```

### Pairwise (within regime)

```text
linear interpolation curves
barrier statistics
```

### Advanced (optional / future)

```text
Hessian eigenvalues / trace
2D slices
mode connectivity paths
```

### Usage

```bash
--probes local
--probes pairwise
--probes all
```

---

## Checkpoint Selection

```text
--checkpoint-kind final      # uses last.pt
--checkpoint-kind best_val   # uses best_val_loss.pt
```

---

## Dataset Subsets

Landscape evaluation uses fixed deterministic subsets:

```text
train subset
validation subset
```

Configured via:

```text
--train-subset-n <int>
--val-subset-n <int>
--subset-seed <int>
```

Subsets are:

* saved to disk
* reused across all probes
* identical across runs within a regime

---

## Output Structure

```
landscape_runs/<regime>/
  subsets/
    train_*.npz
    val_*.npz

  seed_<i>/<checkpoint_kind>/
    endpoint.npz
    perturbation_train.npz
    perturbation_val.npz
    slice1d_random_train.npz
    slice1d_random_val.npz

  pairs/<checkpoint_kind>/
    seed_i__seed_j__interpolation_train.npz
    seed_i__seed_j__interpolation_val.npz
```

---

## Manifest

Each run collection produces a manifest:

```
landscape_runs/<regime>/manifest.csv
```

Fields:

```text
run_id, path, seed, timestamp, hash
```

This defines the mapping between filesystem structure and analysis units.

---

## Determinism Guarantees

* fixed dataset subsets
* deterministic latent evaluation (`z = μ(x)`)
* no stochastic sampling during probes
* fixed pair selection (seeded)

---

## Intended Workflow

```text
train runs → run landscape.py → analyze_landscapes.py
```

* `landscape.py` produces probe artifacts
* `analyze_landscapes.py` consumes them

---

## Design Principles

* operate on runs, not seeds
* separate training and analysis
* produce small, structured artifacts
* ensure comparability across regimes
* avoid implicit randomness

---

## Notes

* Interpolation is only valid within identical architectures.
* Cross-regime comparisons are performed on aggregated statistics, not raw parameter geometry.
* Component losses (e.g. reconstruction, KL) should be analyzed separately from total loss.


Cool things we do
- infer anchor use from config.log
- can recompute all probe files using `--overwrite`
- can force anchors on or off with `--use-anchor-features true` or `false`
- we automatically skip existing outputs and dynamically compute missing ones.

---
