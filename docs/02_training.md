# Training

`train.py` trains one model run and writes logs plus checkpoints to a run
directory.

The minimum command is:

```bash
cd src
python train.py --data-path data/crl_dsprites/observational.npz
```

When `--model-name` is omitted, it is inferred from the dataset filename:

| Dataset stem | Inferred model |
| --- | --- |
| `observational` | `plain` |
| `overlap_support` | `sparse` |
| `single_node_interventions` | `causal_discrepancy` |
| `two_interventions_per_node` | `causal_discrepancy` |

Sparse VAE automatically enables anchor features and sets `anchor_dim` to
`2 * latent_dim` if omitted.

CD-VAE can use anchors, but this is not forced because Regime C and Regime D may
be run either as purely causal-intervention estimators or anchor-augmented
estimators depending on the experiment.

## Run Directory

If `--outdir` is omitted, runs are written to:

```text
runs/<regime>/hash-<config_hash>_time-<timestamp>/
```

Each run contains:

```text
config.log
metrics.log
run.log
start.pt
mid_best.pt
last.pt
best_val_loss.pt
```

`config.log` is important for landscape extraction because it lets
`landscape.py` infer data paths and anchor usage.

## Checkpoint Kinds

`start.pt` is saved immediately after initialization and before the first
optimizer step.

`best_val_loss.pt` is saved whenever validation loss improves.

`final.pt` is overwritten after each epoch and therefore contains the final
training state at the end of training.

`mid_best.pt` is created after training. It copies the available checkpoint
closest to half of the global step at which the best validation loss was reached.
The available candidates are `start.pt`, periodic `epoch_*.pt` checkpoints, and
`best_val_loss.pt`. The midpoint is therefore approximate; decrease
`--save-every` to make it more precise.
