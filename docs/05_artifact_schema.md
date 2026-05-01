# Artifact Schema

Landscape artifacts are compressed `.npz` files. They are intended to be stable,
small enough to inspect, and easy to aggregate without reloading models.

## Seed Artifacts

Path pattern:

```text
landscape_runs/<regime>/<run_id>/<checkpoint_kind>/<probe>.npz
```

Common metadata:

```text
regime
run_index
run_id
seed
checkpoint
checkpoint_kind
probe_name
split
```

`endpoint.npz` contains train/val metrics plus:

```text
num_params
param_norm
```

`perturbation_<split>.npz` contains:

```text
radii
mean_delta
median_delta
p90_delta
max_delta
base_loss
auc_mean_delta
```

`slice1d_random_<split>.npz` contains:

```text
alphas
metric_names
metric_values
base_<component>
max_delta_<component>
min_delta_<component>
center_second_diff_<component>
near_base_width_rel0.001_<component>
near_base_width_rel0.01_<component>
```

`slice2d_random_<split>.npz` contains:

```text
alphas
betas
metric_names
metric_values
sublevel_area_frac_rel0.001_<component>
sublevel_area_frac_rel0.01_<component>
```

`gradnorm_<split>.npz` contains:

```text
component_names
grad_norm
grad_norm_sq
num_params
```

`hessian_<split>.npz` contains:

```text
component_names
status
top_eigenvalue
top_eigenvalues_raw
trace
trace_std
trace_stderr
trace_samples_raw
num_params
power_iters
power_restarts
trace_samples
max_batches
```

## Pair Artifacts

Path pattern:

```text
landscape_runs/<regime>/pairs/<checkpoint_kind>/run_<i>__run_<j>__interpolation_<split>.npz
```

Interpolation artifacts contain:

```text
alphas
metric_names
metric_values
distance
endpoint_i_<component>
endpoint_j_<component>
barrier_<component>
area_excess_<component>
max_delta_<component>
num_peaks_<component>
```

## Analysis Column Naming

Analysis tables use:

```text
<probe>_<stat>_<component>[_extra]
```

Examples:

```text
hessian_top_recon_img
hessian_trace_stderr_recon_img
gradnorm_norm_total
perturb_p90_delta_loss_r_0p001
interp_barrier_recon_img
slice2d_sublevel_area_frac_rel0p01_loss
```

