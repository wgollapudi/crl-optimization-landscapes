# Landscape Probes

`landscape.py` reloads trained checkpoints and writes deterministic landscape
probe artifacts.

The common command is:

```bash
cd src
python landscape.py \
  --regime regimeA \
  --checkpoint-kind best_val \
  --probes endpoint perturbation slice1d pairwise \
  --max-runs 20
```

`--run-root` defaults to `runs/<regime>`.
`--outdir` defaults to `landscape_runs`.

Supported checkpoint kinds are:

```text
start
mid_best
final
best_val
all
```

## Probe Groups

`endpoint` evaluates train and validation losses at the checkpoint.

`perturbation` samples normalized random directions around a checkpoint and
measures loss increases at several radii.

`slice1d` evaluates a normalized 1D line through the checkpoint.

`slice2d` evaluates a normalized 2D grid through the checkpoint. This is mainly
for visualization and representative examples.

`gradnorm` computes deterministic gradient norms for selected components.

`hessian` estimates top Hessian eigenvalues and Hessian traces for selected
components. It records status and uncertainty fields.

`pairwise` evaluates linear interpolation curves between independently trained
models in the same regime.

## Determinism

Landscape probes use fixed train/validation subsets and deterministic VAE
evaluation with `z = mu(x)`. Subset indices are saved under:

```text
landscape_runs/<regime>/subsets/
```
