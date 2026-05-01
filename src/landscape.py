# landscape.py
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from losses import AnchorLossConfig, LinearWarmupSchedule, VAELoss
from models.build import load_model_from_checkpoint
from landscape.eval import endpoint_metrics
from landscape.io import (
    pair_probe_path,
    save_npz,
    save_summary_table,
    seed_probe_path,
    subset_path,
)
from landscape.params import clone_param_vector
from landscape.probes import (
    curve_result_to_dict,
    gradient_norm,
    gradient_result_to_dict,
    hessian_result_to_dict,
    hessian_summary,
    interpolation_curve,
    interpolation_result_to_dict,
    perturbation_result_to_dict,
    perturbation_sharpness,
    slice2d_result_to_dict,
    slice_1d_random,
    slice_2d_random,
)
from landscape.subsets import ensure_subset_loader


CHECKPOINT_KINDS: dict[str, tuple[str, ...]] = {
    "final": ("last.pt", "final.pt"),
    "best_val": ("best_val_loss.pt", "best_loss.pt"),
    # future:
    # "early": ("early.pt",),
    # "mid": ("mid.pt",),
}


@dataclass(frozen=True)
class Run:
    run_id: str
    path: Path
    seed: int | None
    timestamp: str | None
    config_hash: str | None
    config: dict[str, str]


def parse_config_log(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config log: {path}")

    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_bool_string(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def resolve_checkpoint_kinds(kind: str) -> list[str]:
    if kind == "all":
        return list(CHECKPOINT_KINDS.keys())
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(
            f"Unknown checkpoint kind {kind!r}; expected all or one of {sorted(CHECKPOINT_KINDS)}"
        )
    return [kind]


PROBE_ALIASES: dict[str, tuple[str, ...]] = {
    "local": ("endpoint", "perturbation", "slice1d"),
    "expensive": ("gradnorm", "hessian", "slice2d"),
    "all": ("endpoint", "perturbation", "slice1d", "pairwise"),
}


PROBE_NAMES = {
    "endpoint",
    "perturbation",
    "slice1d",
    "slice2d",
    "gradnorm",
    "hessian",
    "pairwise",
}


def resolve_probes(probes: list[str]) -> set[str]:
    resolved: set[str] = set()
    for probe in probes:
        if probe in PROBE_ALIASES:
            resolved.update(PROBE_ALIASES[probe])
        elif probe in PROBE_NAMES:
            resolved.add(probe)
        else:
            raise ValueError(
                f"Unknown probe {probe!r}; expected aliases "
                f"{sorted(PROBE_ALIASES)} or probes {sorted(PROBE_NAMES)}"
            )
    return resolved


def checkpoint_path_for_run(run: Run, checkpoint_kind: str) -> Path:
    if checkpoint_kind not in CHECKPOINT_KINDS:
        raise ValueError(f"Unknown checkpoint kind: {checkpoint_kind}")

    candidates = [run.path / name for name in CHECKPOINT_KINDS[checkpoint_kind]]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No checkpoint found for kind={checkpoint_kind!r} in {run.path}. "
        f"Tried: {', '.join(str(p.name) for p in candidates)}"
    )


def infer_data_path(runs: list[Run], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit

    candidates: list[str] = []
    keys = ("data.path", "path", "data_path")

    for run in runs:
        value = None
        for key in keys:
            if key in run.config:
                value = run.config[key]
                break
        if value is None:
            raise ValueError(
                f"Could not infer data path from {run.path / 'config.log'}; "
                "pass --data-path explicitly."
            )
        candidates.append(value)

    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(
            "Selected runs disagree on data path; pass --data-path explicitly. "
            f"Found: {unique}"
        )

    return Path(unique[0])


def infer_use_anchor_features(runs: list[Run]) -> bool:
    inferred: list[bool] = []

    for run in runs:
        cfg = run.config
        value: bool | None = None

        for key in ("data.use_anchor_features", "use_anchor_features"):
            if key in cfg:
                value = parse_bool_string(cfg[key])
                break

        if value is None:
            for key in ("model.anchor_dim", "anchor_dim"):
                if key in cfg:
                    value = int(cfg[key]) > 0
                    break

        inferred.append(False if value is None else value)

    if len(set(inferred)) != 1:
        details = ", ".join(f"{run.run_id}:{val}" for run, val in zip(runs, inferred))
        raise ValueError(
            "Selected runs disagree on use_anchor_features. "
            f"Use a more specific --run-filter. Details: {details}"
        )

    return inferred[0]


def infer_run_metadata(run_dir: Path) -> Run:
    cfg = parse_config_log(run_dir / "config.log")

    seed = None
    for key in ("seed", "run.seed"):
        if key in cfg:
            seed = int(cfg[key])
            break

    run_id = run_dir.name
    config_hash = None
    timestamp = None

    m = re.search(r"hash-([^_]+)", run_id)
    if m:
        config_hash = m.group(1)

    m = re.search(r"time-(.+)$", run_id)
    if m:
        timestamp = m.group(1)

    return Run(
        run_id=run_id,
        path=run_dir,
        seed=seed,
        timestamp=timestamp,
        config_hash=config_hash,
        config=cfg,
    )


def discover_runs(run_root: Path) -> list[Run]:
    if not run_root.exists():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    runs: list[Run] = []
    for child in sorted(run_root.iterdir()):
        if child.is_dir() and (child / "config.log").exists():
            runs.append(infer_run_metadata(child))

    if not runs:
        raise ValueError(f"No runs with config.log found in {run_root}")

    return runs


def apply_run_filter(runs: list[Run], filters: list[str]) -> list[Run]:
    out = runs
    for filt in filters:
        if "=" not in filt:
            raise ValueError(f"Bad run filter {filt!r}; expected key=value")

        key, value = [s.strip() for s in filt.split("=", 1)]

        if key == "seed":
            out = [r for r in out if r.seed == int(value)]
        elif key == "hash":
            out = [r for r in out if r.config_hash == value]
        elif key == "id":
            out = [r for r in out if r.run_id == value]
        elif key == "contains":
            out = [r for r in out if value in r.run_id or value in str(r.path)]
        else:
            raise ValueError(f"Unknown run filter key {key!r}")

    return out


def select_runs(
    runs: list[Run],
    selection: str,
    max_runs: int | None,
    selection_seed: int,
) -> list[Run]:
    ordered = sorted(runs, key=lambda r: (r.timestamp or "", r.run_id))

    if selection == "all":
        chosen = ordered
    elif selection == "first":
        chosen = ordered
    elif selection == "last":
        chosen = list(reversed(ordered))
    elif selection == "random":
        rng = np.random.default_rng(selection_seed)
        idx = np.arange(len(ordered))
        rng.shuffle(idx)
        chosen = [ordered[i] for i in idx]
    else:
        raise ValueError(f"unknown selection mode: {selection}")

    if max_runs is not None:
        if max_runs <= 0:
            raise ValueError("--max-runs must be positive")
        chosen = chosen[:max_runs]

    return chosen


def choose_pairs(runs: list[Run], max_pairs: int, pair_seed: int) -> list[tuple[Run, Run]]:
    pairs = list(combinations(runs, 2))
    if len(pairs) <= max_pairs:
        return pairs

    rng = np.random.default_rng(pair_seed)
    idx = rng.choice(len(pairs), size=max_pairs, replace=False)
    return [pairs[i] for i in sorted(idx)]


def save_manifest(path: Path, runs: list[Run]) -> None:
    rows = [
        {
            "run_index": i,
            "run_id": r.run_id,
            "path": r.path,
            "seed": "" if r.seed is None else r.seed,
            "timestamp": "" if r.timestamp is None else r.timestamp,
            "hash": "" if r.config_hash is None else r.config_hash,
        }
        for i, r in enumerate(runs)
    ]
    save_summary_table(path, rows)


def save_probe_config(
    path: Path,
    args: argparse.Namespace,
    *,
    data_path: Path,
    checkpoint_kinds: list[str],
    inferred_use_anchor_features: bool,
) -> None:
    payload: dict[str, Any] = vars(args).copy()
    payload["resolved_data_path"] = data_path
    payload["resolved_checkpoint_kinds"] = checkpoint_kinds
    payload["inferred_use_anchor_features"] = inferred_use_anchor_features

    lines = ["LANDSCAPE PROBE CONFIG", "=" * 100]
    for key in sorted(payload):
        lines.append(f"{key} = {payload[key]}")
    lines.append("=" * 100)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def should_skip(path: Path, overwrite: bool) -> bool:
    return path.exists() and not overwrite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run landscape probes over discovered model runs.")

    p.add_argument("--regime", required=True, help="e.g. regimeA, regimeB, regimeC, regimeD")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--data-path", type=Path, default=None)

    p.add_argument(
        "--checkpoint-kind",
        choices=["all", *CHECKPOINT_KINDS.keys()],
        default="all",
    )

    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--selection", choices=["all", "random", "first", "last"], default="all")
    p.add_argument("--selection-seed", type=int, default=0)
    p.add_argument("--run-filter", action="append", default=[])

    p.add_argument(
        "--probes",
        nargs="+",
        choices=sorted(PROBE_NAMES | set(PROBE_ALIASES)),
        default=["all"],
    )

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--use-anchor-features", choices=["auto", "true", "false"], default="auto")

    p.add_argument("--train-subset-n", type=int, default=5000)
    p.add_argument("--val-subset-n", type=int, default=5000)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--overwrite-subsets", action="store_true")

    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--max-batches", type=int, default=None)

    p.add_argument("--radii", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    p.add_argument("--directions-per-radius", type=int, default=20)

    p.add_argument("--slice-points", type=int, default=41)
    p.add_argument("--slice-alpha-max", type=float, default=1.0)

    p.add_argument("--interp-points", type=int, default=41)
    p.add_argument("--max-pairs", type=int, default=100)
    p.add_argument("--pair-seed", type=int, default=0)

    p.add_argument("--grad-components", nargs="+", default=["total", "recon_img"])

    p.add_argument("--hessian-components", nargs="+", default=["total", "recon_img"])
    p.add_argument("--hessian-power-iters", type=int, default=20)
    p.add_argument("--hessian-power-restarts", type=int, default=1)
    p.add_argument("--hessian-trace-samples", type=int, default=20)
    p.add_argument("--hessian-max-batches", type=int, default=3)

    p.add_argument("--slice2d-points", type=int, default=21)
    p.add_argument("--slice2d-alpha-max", type=float, default=1.0)
    p.add_argument("--slice2d-max-runs", type=int, default=3)

    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def run_one_checkpoint_kind(
    *,
    args: argparse.Namespace,
    checkpoint_kind: str,
    runs: list[Run],
    data_path: Path,
    use_anchor_features: bool,
    train_loader,
    val_loader,
    loss_fn: VAELoss,
    alphas_slice: np.ndarray,
    alphas_interp: np.ndarray,
    alphas_slice2d: np.ndarray,
    device: torch.device,
) -> None:
    probes = resolve_probes(args.probes)
    loaded: dict[int, tuple[Run, Path, torch.Tensor, list]] = {}

    for run_index, run in enumerate(runs):
        ckpt = checkpoint_path_for_run(run, checkpoint_kind)
        model = load_model_from_checkpoint(ckpt, device=device)
        theta, specs = clone_param_vector(model)

        if "endpoint" in probes:
            endpoint_path = seed_probe_path(args.outdir, args.regime, run.run_id, checkpoint_kind, "endpoint")
            if should_skip(endpoint_path, args.overwrite):
                print(f"[skip] {endpoint_path}")
            else:
                endpoints = endpoint_metrics(
                    model=model,
                    loss_fn=loss_fn,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    max_batches=args.max_batches,
                )
                save_npz(
                    endpoint_path,
                    regime=args.regime,
                    run_index=run_index,
                    run_id=run.run_id,
                    seed=-1 if run.seed is None else run.seed,
                    checkpoint=str(ckpt),
                    checkpoint_kind=checkpoint_kind,
                    probe_name="endpoint",
                    num_params=int(theta.numel()),
                    param_norm=float(torch.linalg.vector_norm(theta).item()),
                    **endpoints,
                )
                print(f"[write] {endpoint_path}")

        for split_name, loader in [("train", train_loader), ("val", val_loader)]:
            if "perturbation" in probes:
                pert_path = seed_probe_path(args.outdir, args.regime, run.run_id, checkpoint_kind, f"perturbation_{split_name}")
                if should_skip(pert_path, args.overwrite):
                    print(f"[skip] {pert_path}")
                else:
                    pert = perturbation_sharpness(
                        model=model,
                        theta=theta,
                        specs=specs,
                        loss_fn=loss_fn,
                        loader=loader,
                        device=device,
                        radii=args.radii,
                        directions_per_radius=args.directions_per_radius,
                        direction_seed=run.seed if run.seed is not None else run_index,
                        max_batches=args.max_batches,
                        normalization="layerwise",
                    )
                    save_npz(
                        pert_path,
                        regime=args.regime,
                        run_index=run_index,
                        run_id=run.run_id,
                        seed=-1 if run.seed is None else run.seed,
                        checkpoint=str(ckpt),
                        checkpoint_kind=checkpoint_kind,
                        probe_name="perturbation",
                        split=split_name,
                        **perturbation_result_to_dict(pert),
                    )
                    print(f"[write] {pert_path}")

            if "slice1d" in probes:
                slice_path = seed_probe_path(args.outdir, args.regime, run.run_id, checkpoint_kind, f"slice1d_random_{split_name}")
                if should_skip(slice_path, args.overwrite):
                    print(f"[skip] {slice_path}")
                else:
                    sl = slice_1d_random(
                        model=model,
                        theta=theta,
                        specs=specs,
                        loss_fn=loss_fn,
                        loader=loader,
                        device=device,
                        alphas=alphas_slice,
                        direction_seed=run.seed if run.seed is not None else run_index,
                        max_batches=args.max_batches,
                        normalization="layerwise",
                    )
                    save_npz(
                        slice_path,
                        regime=args.regime,
                        run_index=run_index,
                        run_id=run.run_id,
                        seed=-1 if run.seed is None else run.seed,
                        checkpoint=str(ckpt),
                        checkpoint_kind=checkpoint_kind,
                        probe_name="slice1d",
                        split=split_name,
                        **curve_result_to_dict(sl),
                    )
                    print(f"[write] {slice_path}")

            if "gradnorm" in probes:
                grad_path = seed_probe_path(args.outdir, args.regime, run.run_id, checkpoint_kind, f"gradnorm_{split_name}")
                if should_skip(grad_path, args.overwrite):
                    print(f"[skip] {grad_path}")
                else:
                    grad = gradient_norm(
                        model=model,
                        theta=theta,
                        specs=specs,
                        loss_fn=loss_fn,
                        loader=loader,
                        device=device,
                        components=args.grad_components,
                        max_batches=args.max_batches,
                    )
                    save_npz(
                        grad_path,
                        regime=args.regime,
                        run_index=run_index,
                        run_id=run.run_id,
                        seed=-1 if run.seed is None else run.seed,
                        checkpoint=str(ckpt),
                        checkpoint_kind=checkpoint_kind,
                        probe_name="gradnorm",
                        split=split_name,
                        **gradient_result_to_dict(grad),
                    )
                    print(f"[write] {grad_path}")

            if "hessian" in probes:
                hess_path = seed_probe_path(args.outdir, args.regime, run.run_id, checkpoint_kind, f"hessian_{split_name}")
                if should_skip(hess_path, args.overwrite):
                    print(f"[skip] {hess_path}")
                else:
                    hess = hessian_summary(
                        model=model,
                        theta=theta,
                        specs=specs,
                        loss_fn=loss_fn,
                        loader=loader,
                        device=device,
                        components=args.hessian_components,
                        power_iters=args.hessian_power_iters,
                        power_restarts=args.hessian_power_restarts,
                        trace_samples=args.hessian_trace_samples,
                        max_batches=args.hessian_max_batches,
                        seed=run.seed if run.seed is not None else run_index,
                    )
                    save_npz(
                        hess_path,
                        regime=args.regime,
                        run_index=run_index,
                        run_id=run.run_id,
                        seed=-1 if run.seed is None else run.seed,
                        checkpoint=str(ckpt),
                        checkpoint_kind=checkpoint_kind,
                        probe_name="hessian",
                        split=split_name,
                        **hessian_result_to_dict(hess),
                    )
                    print(f"[write] {hess_path}")

            if "slice2d" in probes and run_index < args.slice2d_max_runs:
                slice2d_path = seed_probe_path(args.outdir, args.regime, run.run_id, checkpoint_kind, f"slice2d_random_{split_name}")
                if should_skip(slice2d_path, args.overwrite):
                    print(f"[skip] {slice2d_path}")
                else:
                    sl2 = slice_2d_random(
                        model=model,
                        theta=theta,
                        specs=specs,
                        loss_fn=loss_fn,
                        loader=loader,
                        device=device,
                        alphas=alphas_slice2d,
                        betas=alphas_slice2d,
                        direction_seed=run.seed if run.seed is not None else run_index,
                        max_batches=args.max_batches,
                        normalization="layerwise",
                    )
                    save_npz(
                        slice2d_path,
                        regime=args.regime,
                        run_index=run_index,
                        run_id=run.run_id,
                        seed=-1 if run.seed is None else run.seed,
                        checkpoint=str(ckpt),
                        checkpoint_kind=checkpoint_kind,
                        probe_name="slice2d",
                        split=split_name,
                        **slice2d_result_to_dict(sl2),
                    )
                    print(f"[write] {slice2d_path}")

        loaded[run_index] = (run, ckpt, theta.detach().cpu(), specs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "pairwise" in probes:
        pairs = choose_pairs(runs=runs, max_pairs=args.max_pairs, pair_seed=args.pair_seed)
        run_to_index = {run.run_id: i for i, run in enumerate(runs)}

        for run_i, run_j in pairs:
            idx_i = run_to_index[run_i.run_id]
            idx_j = run_to_index[run_j.run_id]

            _, ckpt_i, theta_i_cpu, specs_i = loaded[idx_i]
            _, _, theta_j_cpu, specs_j = loaded[idx_j]

            if len(specs_i) != len(specs_j):
                raise ValueError(
                    f"Cannot interpolate runs {run_i.run_id}, {run_j.run_id}: "
                    "different number of trainable tensors"
                )

            model_i: torch.nn.Module | None = None

            for split_name, loader in [("train", train_loader), ("val", val_loader)]:
                interp_path = pair_probe_path(
                    args.outdir,
                    args.regime,
                    idx_i,
                    idx_j,
                    checkpoint_kind,
                    f"interpolation_{split_name}",
                )

                if should_skip(interp_path, args.overwrite):
                    print(f"[skip] {interp_path}")
                    continue

                if model_i is None:
                    model_i = load_model_from_checkpoint(ckpt_i, device=device)

                theta_i = theta_i_cpu.to(device)
                theta_j = theta_j_cpu.to(device)

                if theta_i.shape != theta_j.shape:
                    raise ValueError(
                        f"Cannot interpolate runs {run_i.run_id}, {run_j.run_id}: "
                        f"parameter shapes differ: {theta_i.shape} vs {theta_j.shape}"
                    )

                interp = interpolation_curve(
                    model=model_i,
                    theta_a=theta_i,
                    theta_b=theta_j,
                    specs=specs_i,
                    loss_fn=loss_fn,
                    loader=loader,
                    device=device,
                    alphas=alphas_interp,
                    max_batches=args.max_batches,
                )

                save_npz(
                    interp_path,
                    regime=args.regime,
                    run_index_i=idx_i,
                    run_index_j=idx_j,
                    run_id_i=run_i.run_id,
                    run_id_j=run_j.run_id,
                    seed_i=-1 if run_i.seed is None else run_i.seed,
                    seed_j=-1 if run_j.seed is None else run_j.seed,
                    checkpoint_i=str(ckpt_i),
                    checkpoint_j=str(checkpoint_path_for_run(run_j, checkpoint_kind)),
                    checkpoint_kind=checkpoint_kind,
                    probe_name="interpolation",
                    split=split_name,
                    **interpolation_result_to_dict(interp),
                )
                print(f"[write] {interp_path}")

            del model_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    runs = discover_runs(args.run_root)
    runs = apply_run_filter(runs, args.run_filter)
    runs = select_runs(
        runs,
        selection=args.selection,
        max_runs=args.max_runs,
        selection_seed=args.selection_seed,
    )
    if not runs:
        raise ValueError("No runs selected")

    checkpoint_kinds = resolve_checkpoint_kinds(args.checkpoint_kind)
    data_path = infer_data_path(runs, args.data_path)

    if args.use_anchor_features == "auto":
        use_anchor_features = infer_use_anchor_features(runs)
    else:
        use_anchor_features = parse_bool_string(args.use_anchor_features)

    regime_root = args.outdir / args.regime
    regime_root.mkdir(parents=True, exist_ok=True)

    save_manifest(regime_root / "manifest.csv", runs)
    save_probe_config(
        regime_root / "landscape_config.log",
        args,
        data_path=data_path,
        checkpoint_kinds=checkpoint_kinds,
        inferred_use_anchor_features=use_anchor_features,
    )

    loss_fn = VAELoss(
        beta_schedule=LinearWarmupSchedule(
            start_value=args.beta,
            end_value=args.beta,
            warmup_epochs=0,
        ),
        anchor_cfg=AnchorLossConfig(kind="mse", weight=1.0),
    )

    train_loader = ensure_subset_loader(
        subset_path=subset_path(args.outdir, args.regime, "train", args.train_subset_n, args.subset_seed),
        dataset_path=data_path,
        regime=args.regime,
        split="train",
        n_examples=args.train_subset_n,
        subset_seed=args.subset_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_anchor_features=use_anchor_features,
        overwrite=args.overwrite_subsets,
    )

    val_loader = ensure_subset_loader(
        subset_path=subset_path(args.outdir, args.regime, "val", args.val_subset_n, args.subset_seed),
        dataset_path=data_path,
        regime=args.regime,
        split="val",
        n_examples=args.val_subset_n,
        subset_seed=args.subset_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_anchor_features=use_anchor_features,
        overwrite=args.overwrite_subsets,
    )

    alphas_slice = np.linspace(-args.slice_alpha_max, args.slice_alpha_max, args.slice_points)
    alphas_interp = np.linspace(0.0, 1.0, args.interp_points)
    alphas_slice2d = np.linspace(-args.slice2d_alpha_max, args.slice2d_alpha_max, args.slice2d_points)

    for checkpoint_kind in checkpoint_kinds:
        print(f"[checkpoint-kind] {checkpoint_kind}")
        run_one_checkpoint_kind(
            args=args,
            checkpoint_kind=checkpoint_kind,
            runs=runs,
            data_path=data_path,
            use_anchor_features=use_anchor_features,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            alphas_slice=alphas_slice,
            alphas_interp=alphas_interp,
            alphas_slice2d=alphas_slice2d,
            device=device,
        )


if __name__ == "__main__":
    main()
