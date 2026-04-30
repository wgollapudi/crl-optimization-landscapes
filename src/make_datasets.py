#!/usr/bin/env python3
"""Generate CRL datasets from raw dSprites NPZ.

This script treats dSprites as a renderer table: it samples latent variables from a
small synthetic (continuous) SCM, quantizes them to the nearest valid dSprites
factor levels, and then looks up the corresponding raw image from the NPZ.

It produces datasets for the four regimes:
  1. observational
  2. overlap_support
  3. single_node_interventions
  4. two_interventions_per_node

Outputs are saved as compressed .npz files.
Each output file contains:
  - images
  - class_tuples
  - latents_values
  - latents_true_continuous
  - latents_quantized_continuous
  - orientation_cos
  - orientation_sin
  - anchor_features
  - env_id
  - intervention_target
  - intervention_variant
  - split

Notes
- orientation is treated as a circular variable during quantization, which avoids
  wraparound artifacts near 0 and 2*pi.
- train/val/test splits are stratified by env_id, while identical rendered tuples
  are kept in the same split to avoid leakage across splits.

Ex usage
python make_datasets.py \
  --dsprites /path/to/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
  --outdir ./crl_dsprites \
  --shape ellipse \
  --n-obs 40000 \
  --n-per-intervention 5000 \
  --log-file ./crl_dsprites/generation.log
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


FACTOR_NAMES = ["color", "shape", "scale", "orientation", "posX", "posY"]
ACTIVE_FACTORS = ["scale", "orientation", "posX", "posY"]
ACTIVE_TO_DSPRITES = {
    "scale": 2,
    "orientation": 3,
    "posX": 4,
    "posY": 5,
}
SHAPE_NAME_TO_CLASS = {
    "square": 0,
    "ellipse": 1,
    "heart": 2,
}
TAU = 2.0 * np.pi


def log(msg: str, fh=None) -> None:
    print(msg)
    if fh is not None:
        fh.write(msg + "\n")


@dataclass
class SCMConfig:
    adjacency: List[List[float]]
    noise_scales: List[float]
    clip: float = 2.5


DEFAULT_SCM = SCMConfig(
    # z = [scale, orientation, posX, posY]
    # Edges: scale -> orientation, scale -> posX, orientation -> posY
    adjacency=[
        [0.0, 0.0, 0.0, 0.0],
        [0.35, 0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 0.0],
        [0.0, 0.30, 0.0, 0.0],
    ],
    noise_scales=[0.28, 0.24, 0.24, 0.24],
    clip=1.0
)


class DSpritesTable:
    def __init__(self, npz_path: Path):
        with np.load(npz_path, allow_pickle=True, encoding="latin1") as data:
            self.imgs = data["imgs"]
            self.latents_values = data["latents_values"]
            self.latents_classes = data["latents_classes"]
            metadata = data["metadata"]

        if isinstance(metadata, np.ndarray):
            metadata = metadata.item()
        self.metadata = metadata
        self.latents_sizes = np.array(metadata["latents_sizes"], dtype=np.int64)
        self.latents_possible_values = metadata["latents_possible_values"]
        self.bases = np.concatenate(
            (
                np.cumprod(self.latents_sizes[::-1])[::-1][1:],
                np.array([1], dtype=np.int64),
            )
        )

    def class_tuple_to_index(self, class_tuple: np.ndarray) -> np.ndarray:
        return np.sum(class_tuple * self.bases, axis=1).astype(np.int64)

    def fetch_images(self, class_tuples: np.ndarray) -> np.ndarray:
        flat_idx = self.class_tuple_to_index(class_tuples)
        return self.imgs[flat_idx]

    def factor_values(self, name: str) -> np.ndarray:
        return np.array(self.latents_possible_values[name], dtype=np.float64)



class Quantizer:
    def __init__(self, table: DSpritesTable):
        self.table = table
        self.value_grids = {name: table.factor_values(name) for name in ACTIVE_FACTORS}

    def normalize(self, name: str, raw_values: np.ndarray) -> np.ndarray:
        grid = self.value_grids[name]
        lo, hi = float(grid.min()), float(grid.max())
        if hi == lo:
            return np.zeros_like(raw_values)
        return 2.0 * (raw_values - lo) / (hi - lo) - 1.0

    def denormalize(self, name: str, z: np.ndarray) -> np.ndarray:
        grid = self.value_grids[name]
        lo, hi = float(grid.min()), float(grid.max())
        clipped = np.clip(z, -1.0, 1.0)
        raw = (clipped + 1.0) * 0.5 * (hi - lo) + lo
        if name == "orientation":
            raw = np.mod(raw, TAU)
        return raw

    def quantize(self, name: str, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raw = self.denormalize(name, z)
        grid = self.value_grids[name]
        if name == "orientation":
            diff = np.abs(raw[:, None] - grid[None, :])
            wrapped = np.minimum(diff, TAU - diff)
            idx = wrapped.argmin(axis=1)
        else:
            idx = np.abs(raw[:, None] - grid[None, :]).argmin(axis=1)
        return idx.astype(np.int64), grid[idx]


class SCMGenerator:
    def __init__(self, config: SCMConfig, rng: np.random.Generator):
        self.A = np.array(config.adjacency, dtype=np.float64)
        self.noise_scales = np.array(config.noise_scales, dtype=np.float64)
        self.clip = float(config.clip)
        self.rng = rng
        if self.A.shape != (4, 4):
            raise ValueError("adjacency must be 4x4 for the active dSprites factors")
        if np.any(np.triu(self.A) != 0):
            raise ValueError("adjacency must have zeros on and above the diagonal (i.e., strictly lower triangular in topological order)")

    def sample_observational(self, n: int) -> np.ndarray:
        eps = self.rng.normal(size=(n, 4)) * self.noise_scales
        z = np.zeros_like(eps)
        for j in range(4):
            z[:, j] = eps[:, j] + z @ self.A[j]
        return np.clip(z, -self.clip, self.clip)

    def apply_perfect_intervention(
        self,
        target: int,
        mean: float,
        scale: float,
        n: int,
    ) -> np.ndarray:
        out = np.zeros((n, 4), dtype=np.float64)
        eps = self.rng.normal(size=(n, 4)) * self.noise_scales
        for j in range(4):
            if j == target:
                out[:, j] = self.rng.normal(loc=mean, scale=scale, size=n)
            else:
                out[:, j] = eps[:, j] + out @ self.A[j]
        return np.clip(out, -self.clip, self.clip)

def summarize_latent_ranges(name: str, arrays: Mapping[str, np.ndarray], fh=None) -> None:
    z_true = arrays["latents_true_continuous"]
    z_quant = arrays["latents_quantized_continuous"]
    log(f"{name}:", fh)
    log(f"  n = {len(z_true)}", fh)
    log(f"  true latent min = {z_true.min(axis=0).tolist()}", fh)
    log(f"  true latent max = {z_true.max(axis=0).tolist()}", fh)
    log(f"  quantized latent min = {z_quant.min(axis=0).tolist()}", fh)
    log(f"  quantized latent max = {z_quant.max(axis=0).tolist()}", fh)


def build_class_tuples(
    quantizer: Quantizer,
    z_true_cont: np.ndarray,
    shape_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = z_true_cont.shape[0]
    class_tuples = np.zeros((n, 6), dtype=np.int64)
    values = np.zeros((n, 6), dtype=np.float64)

    class_tuples[:, 0] = 0
    class_tuples[:, 1] = shape_class
    values[:, 0] = quantizer.table.factor_values("color")[0]
    values[:, 1] = quantizer.table.factor_values("shape")[shape_class]

    z_quantized_cont = np.zeros_like(z_true_cont)
    for j, name in enumerate(ACTIVE_FACTORS):
        class_idx, snapped = quantizer.quantize(name, z_true_cont[:, j])
        ds_idx = ACTIVE_TO_DSPRITES[name]
        class_tuples[:, ds_idx] = class_idx
        values[:, ds_idx] = snapped
        z_quantized_cont[:, j] = quantizer.normalize(name, snapped)

    return class_tuples, values, z_quantized_cont


def make_anchor_features(
    z: np.ndarray,
    anchors_per_latent: int = 2,
    noise_std: float = 0.03,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build enginnered anchor featrues. (to satisfy sparse identifibility theorem in Moran et. at. 2022)

    Output order:
      z0_anchor0, z0_anchor1, ..., z1_anchor0, z1_anchor1, ...

    Each anchor depends on exactly one latent coordinate.

    Note that anchor featrues are constructed as determanistic nonlinear functions of the underlying SCM latents prior to quantization. This preserves the identifiability assumptions of Moran et al. 2022, while images are generated from quantized latents via the dSprites renderer.

    Unfortunatly, the model will now technically see two inconsistant views of "the same" latent, as image depends on z_quantized, and anchors depend on z_true.
    """
    if z.ndim != 2:
        raise ValueError(f"z must have shape [N, latent_dim], got {z.shape}")
    if anchors_per_latent < 2:
        raise ValueError("identifiability requires at least two anchors per latent")
    if noise_std < 0.0:
        raise ValueError("noise_std must be >= 0")

    if rng is None:
        rng = np.random.default_rng()

    funcs = [
        lambda x: x,
        lambda x: np.tanh(1.5 * x),
        lambda x: x**2 + 0.5 * x,
        lambda x: np.sin(np.pi * x),
        lambda x: np.sign(x) * np.sqrt(np.abs(x) + 1e-6),
        lambda x: x**3,
    ]

    anchors: list[np.ndarray] = []

    for j in range(z.shape[1]):
        zj = z[:, j].astype(np.float32, copy=False)

        for a in range(anchors_per_latent):
            f = funcs[a % len(funcs)]
            anchor = f(zj).astype(np.float32, copy=False)
            anchors.append(anchor)

    out = np.stack(anchors, axis=1).astype(np.float32, copy=False)

    # Standardize each anchor dimension so loss scales are comparable.
    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True)
    out = (out - mean) / (std + 1e-6)

    if noise_std > 0.0:
        out += rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    return out

def add_anchor_features(
    arrays: MutableMapping[str, np.ndarray],
    anchors_per_latent: int,
    noise_std: float,
    rng: np.random.Generator,
) -> None:
    arrays["anchor_features"] = make_anchor_features(
        arrays["latents_true_continuous"],
        anchors_per_latent=anchors_per_latent,
        noise_std=noise_std,
        rng=rng,
    )


def make_overlap_grid(
    quantizer: Quantizer,
    shape_class: int,
    max_points: int,
) -> Mapping[str, np.ndarray]:
    # Build a rectangular grid in normalized latent space [-1, 1]^4,
    # then quantize/render through dSprites.
    per_dim = int(np.floor(max_points ** 0.25))
    per_dim = max(per_dim, 2)

    grid_1d = np.linspace(-0.9, 0.9, per_dim)
    mesh = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, grid_1d, indexing="ij"))
    z_true = mesh.reshape(4, -1).T

    if z_true.shape[0] > max_points:
        idx = np.linspace(0, z_true.shape[0] - 1, max_points).round().astype(int)
        z_true = z_true[idx]

    classes, values, z_quantized = build_class_tuples(quantizer, z_true, shape_class)

    return {
        "class_tuples": classes,
        "latents_values": values,
        "latents_true_continuous": z_true,
        "latents_quantized_continuous": z_quantized,
        "orientation_cos": np.cos(values[:, 3]),
        "orientation_sin": np.sin(values[:, 3]),
        "env_id": np.zeros(classes.shape[0], dtype=np.int64),
        "intervention_target": np.full(classes.shape[0], -1, dtype=np.int64),
        "intervention_variant": np.full(classes.shape[0], -1, dtype=np.int64),
    }


def add_images(table: DSpritesTable, arrays: MutableMapping[str, np.ndarray]) -> None:
    arrays["images"] = table.fetch_images(arrays["class_tuples"])


def split_indices_stratified(
    env_id: np.ndarray,
    class_tuples: np.ndarray,
    rng: np.random.Generator,
    train_frac: float,
    val_frac: float,
) -> np.ndarray:
    n = len(env_id)
    split = np.empty(n, dtype=np.int64)

    # Group by (env_id, rendered tuple) so identical rendered examples stay in the same split.
    groups = {}
    for i, key in enumerate(zip(env_id.tolist(), map(tuple, class_tuples.tolist()))):
        groups.setdefault(key, []).append(i)

    # Bucket groups by environment for stratification.
    env_to_group_keys = {}
    for (env, cls_key) in groups:
        env_to_group_keys.setdefault(env, []).append((env, cls_key))

    for env, keys in env_to_group_keys.items():
        perm = rng.permutation(len(keys))
        keys = [keys[i] for i in perm]

        m = len(keys)
        n_train = int(train_frac * m)
        n_val = int(val_frac * m)

        train_keys = keys[:n_train]
        val_keys = keys[n_train:n_train + n_val]
        test_keys = keys[n_train + n_val:]

        for key in train_keys:
            split[np.array(groups[key], dtype=np.int64)] = 0
        for key in val_keys:
            split[np.array(groups[key], dtype=np.int64)] = 1
        for key in test_keys:
            split[np.array(groups[key], dtype=np.int64)] = 2

    return split


def parse_adjacency(s: str) -> List[List[float]]:
    rows = s.strip().split(";")
    mat = []
    for r in rows:
        mat.append([float(x) for x in r.strip().split(",")])
    if len(mat) != 4 or any(len(r) != 4 for r in mat):
        raise ValueError("adjacency must be 4x4")
    return mat


def parse_vector(s: str, n: int) -> List[float]:
    vals = [float(x) for x in s.strip().split(",")]
    if len(vals) != n:
        raise ValueError(f"expected {n} values")
    return vals


def save_regime(
    outdir: Path,
    regime_name: str,
    arrays: Mapping[str, np.ndarray],
    split_seed: int,
    train_frac: float,
    val_frac: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(split_seed)
    payload = dict(arrays)
    payload["split"] = split_indices_stratified(
        payload["env_id"],
        payload["class_tuples"],
        rng,
        train_frac,
        val_frac,
    )
    np.savez_compressed(outdir / f"{regime_name}.npz", **payload)


def generate_observational(
    quantizer: Quantizer,
    scm: SCMGenerator,
    shape_class: int,
    n_obs: int,
) -> Mapping[str, np.ndarray]:
    z_true = scm.sample_observational(n_obs)
    classes, values, z_quantized = build_class_tuples(quantizer, z_true, shape_class)
    return {
        "class_tuples": classes,
        "latents_values": values,
        "latents_true_continuous": z_true,
        "latents_quantized_continuous": z_quantized,
        "orientation_cos": np.cos(values[:, 3]),
        "orientation_sin": np.sin(values[:, 3]),
        "env_id": np.zeros(n_obs, dtype=np.int64),
        "intervention_target": np.full(n_obs, -1, dtype=np.int64),
        "intervention_variant": np.full(n_obs, -1, dtype=np.int64),
    }


def generate_single_node_interventions(
    quantizer: Quantizer,
    scm: SCMGenerator,
    shape_class: int,
    n_obs: int,
    n_per_intervention: int,
) -> Mapping[str, np.ndarray]:
    pieces = []
    obs = generate_observational(quantizer, scm, shape_class, n_obs)
    pieces.append(obs)

    means = np.array([-0.75, 0.75, -0.60, 0.60])
    scales = np.array([0.12, 0.12, 0.12, 0.12])

    for target in range(4):
        z_true = scm.apply_perfect_intervention(target, means[target], scales[target], n_per_intervention)
        classes, values, z_quantized = build_class_tuples(quantizer, z_true, shape_class)
        pieces.append({
            "class_tuples": classes,
            "latents_values": values,
            "latents_true_continuous": z_true,
            "latents_quantized_continuous": z_quantized,
            "orientation_cos": np.cos(values[:, 3]),
            "orientation_sin": np.sin(values[:, 3]),
            "env_id": np.full(n_per_intervention, target + 1, dtype=np.int64),
            "intervention_target": np.full(n_per_intervention, target, dtype=np.int64),
            "intervention_variant": np.zeros(n_per_intervention, dtype=np.int64),
        })

    return concat_piece_dicts(pieces)


def generate_two_interventions_per_node(
    quantizer: Quantizer,
    scm: SCMGenerator,
    shape_class: int,
    n_obs: int,
    n_per_intervention: int,
) -> Mapping[str, np.ndarray]:
    pieces = []
    obs = generate_observational(quantizer, scm, shape_class, n_obs)
    pieces.append(obs)

    variants = [
        {
            "mean": np.array([-0.80, 0.80, -0.65, 0.65]),
            "scale": np.array([0.10, 0.10, 0.10, 0.10]),
        },
        {
            "mean": np.array([0.20, -0.70, 0.55, -0.55]),
            "scale": np.array([0.18, 0.18, 0.18, 0.18]),
        },
    ]

    env_counter = 1
    for target in range(4):
        for variant_idx, variant in enumerate(variants):
            z_true = scm.apply_perfect_intervention(
                target,
                float(variant["mean"][target]),
                float(variant["scale"][target]),
                n_per_intervention,
            )
            classes, values, z_quantized = build_class_tuples(quantizer, z_true, shape_class)
            pieces.append({
                "class_tuples": classes,
                "latents_values": values,
                "latents_true_continuous": z_true,
                "latents_quantized_continuous": z_quantized,
                "orientation_cos": np.cos(values[:, 3]),
                "orientation_sin": np.sin(values[:, 3]),
                "env_id": np.full(n_per_intervention, env_counter, dtype=np.int64),
                "intervention_target": np.full(n_per_intervention, target, dtype=np.int64),
                "intervention_variant": np.full(n_per_intervention, variant_idx, dtype=np.int64),
            })
            env_counter += 1

    return concat_piece_dicts(pieces)


def concat_piece_dicts(pieces: Sequence[Mapping[str, np.ndarray]]) -> Mapping[str, np.ndarray]:
    keys = list(pieces[0].keys())
    return {k: np.concatenate([piece[k] for piece in pieces], axis=0) for k in keys}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsprites", type=Path, required=True, help="Path to the raw dSprites NPZ file")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--shape", choices=sorted(SHAPE_NAME_TO_CLASS), default="ellipse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-obs", type=int, default=40000, help="Observational sample count")
    parser.add_argument(
        "--n-per-intervention",
        type=int,
        default=5000,
        help="Samples per intervention environment",
    )
    parser.add_argument(
        "--overlap-grid-budget",
        type=int,
        default=20000,
        help="Approximate maximum number of overlap-support samples",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file (otherwise prints only to stdout)",
    )
    parser.add_argument(
        "--adjacency",
        type=str,
        default=None,
        help="4x4 lower-triangular matrix, rows separated by ';', entries by ','",
    )
    parser.add_argument(
        "--noise-scales",
        type=str,
        default=None,
        help="Comma-separated 4 noise std values",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--anchors-per-latent",
        type=int,
        default=2,
        help="Number of engineered anchor features to repeat for each latent factor",
    )
    parser.add_argument(
        "--anchor-noise-std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise std added to engineered anchor features",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    rng = np.random.default_rng(args.seed)
    table = DSpritesTable(args.dsprites)
    quantizer = Quantizer(table)
    if args.adjacency is not None:
        adjacency = parse_adjacency(args.adjacency)
    else:
        adjacency = DEFAULT_SCM.adjacency

    if args.noise_scales is not None:
        noise_scales = parse_vector(args.noise_scales, 4)
    else:
        noise_scales = DEFAULT_SCM.noise_scales

    config = SCMConfig(
        adjacency=adjacency,
        noise_scales=noise_scales,
        clip=args.clip,
    )

    scm = SCMGenerator(config, rng)
    shape_class = SHAPE_NAME_TO_CLASS[args.shape]

    log_fh = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log_file, "w", encoding="utf-8")

    try:
        observational = generate_observational(quantizer, scm, shape_class, args.n_obs)
        add_anchor_features(
            observational,
            anchors_per_latent=args.anchors_per_latent,
            noise_std=args.anchor_noise_std,
            rng=rng,
        )
        add_images(table, observational)

        overlap_support = make_overlap_grid(
            quantizer,
            shape_class,
            args.overlap_grid_budget,
        )
        add_anchor_features(
            overlap_support,
            anchors_per_latent=args.anchors_per_latent,
            noise_std=args.anchor_noise_std,
            rng=rng,
        )
        add_images(table, overlap_support)

        single_node = generate_single_node_interventions(
            quantizer,
            scm,
            shape_class,
            args.n_obs,
            args.n_per_intervention,
        )
        add_anchor_features(
            single_node,
            anchors_per_latent=args.anchors_per_latent,
            noise_std=args.anchor_noise_std,
            rng=rng,
        )
        add_images(table, single_node)

        two_per_node = generate_two_interventions_per_node(
            quantizer,
            scm,
            shape_class,
            args.n_obs,
            args.n_per_intervention,
        )
        add_anchor_features(
            two_per_node,
            anchors_per_latent=args.anchors_per_latent,
            noise_std=args.anchor_noise_std,
            rng=rng,
        )
        add_images(table, two_per_node)

        save_regime(args.outdir, "observational", observational, args.seed + 11, args.train_frac, args.val_frac)
        save_regime(args.outdir, "overlap_support", overlap_support, args.seed + 17, args.train_frac, args.val_frac)
        save_regime(args.outdir, "single_node_interventions", single_node, args.seed + 23, args.train_frac, args.val_frac)
        save_regime(args.outdir, "two_interventions_per_node", two_per_node, args.seed + 29, args.train_frac, args.val_frac)

        log("=== DATASET GENERATION SUMMARY ===", log_fh)
        log(f"dsprites_path: {args.dsprites}", log_fh)
        log(f"shape_fixed_to: {args.shape}", log_fh)
        log(f"active_factors: {ACTIVE_FACTORS}", log_fh)
        log("", log_fh)

        log("SCM:", log_fh)
        log(f"adjacency: {scm.A.tolist()}", log_fh)
        log(f"noise_scales: {scm.noise_scales.tolist()}", log_fh)
        log(f"clip: {scm.clip}", log_fh)
        log("", log_fh)

        log("ANCHORS:", log_fh)
        log(f"anchors_per_latent: {args.anchors_per_latent}", log_fh)
        log(f"anchor_noise_std: {args.anchor_noise_std}", log_fh)
        log(f"anchor_dim: {observational['anchor_features'].shape[1]}", log_fh)
        log("", log_fh)

        log("REGIMES:", log_fh)
        log("  observational", log_fh)
        log("  overlap_support", log_fh)
        log("  single_node_interventions", log_fh)
        log("  two_interventions_per_node", log_fh)

        log("LATENT RANGES:", log_fh)
        summarize_latent_ranges("observational", observational, log_fh)
        summarize_latent_ranges("overlap_support", overlap_support, log_fh)
        summarize_latent_ranges("single_node_interventions", single_node, log_fh)
        summarize_latent_ranges("two_interventions_per_node", two_per_node, log_fh)
        log("", log_fh)

        cfg_str = str(adjacency) + str(noise_scales) + str(args.seed)
        cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
        log(f"config_hash: {cfg_hash}", log_fh)
        log(f"Wrote datasets to {args.outdir}", log_fh)
        for name in [
            "observational",
            "overlap_support",
            "single_node_interventions",
            "two_interventions_per_node",
        ]:
            log(f"  - {name}.npz", log_fh)
    finally:
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    main()
