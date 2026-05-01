# Data Generation

`make_datasets.py` generates the four regime datasets:

```text
observational.npz
overlap_support.npz
single_node_interventions.npz
two_interventions_per_node.npz
```

The preferred entrypoint is:

```bash
cd src
./build_datasets.sh --overwrite
```

`build_datasets.sh` downloads the raw dSprites NPZ if needed and then calls
`make_datasets.py`.

## Renderer Table Design

dSprites supplies images and factor grids, but not the causal process we need.
The generator therefore samples continuous latents from a synthetic SCM:

```text
scale -> orientation
scale -> posX
orientation -> posY
```

Those continuous latents are quantized to valid dSprites factor values and
rendered by table lookup. Shape is fixed by default to `ellipse`.

## Anchors

Each generated dataset includes `anchor_features`. The anchors are engineered
features used by Sparse VAE:

```text
a_{j,m} = g_j(z_j) + eps_{j,m}
```

There are two anchors per latent by default. Each anchor depends on exactly one
latent coordinate, and the repeated anchors for a coordinate share the same
conditional mean map. This is stricter than simply adding arbitrary scalar
views.

## Splits

Splits are stratified by `env_id`, and identical rendered dSprites tuples are
kept in the same split. This avoids leakage where the same rendered image
appears in both train and validation.

