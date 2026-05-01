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
last.pt
best_val_loss.pt
```

`config.log` is important for landscape extraction because it lets
`landscape.py` infer data paths and anchor usage.

