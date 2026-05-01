from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


IDENTITY_COLUMNS = [
    "table",
    "regime",
    "run_id",
    "run_index",
    "seed",
    "run_id_i",
    "run_id_j",
    "run_index_i",
    "run_index_j",
    "seed_i",
    "seed_j",
    "checkpoint_kind",
    "split",
]

CHECKPOINT_KINDS = ("final", "best_val")
DEFAULT_REGIMES = ("regimeA", "regimeB", "regimeC", "regimeD")
DEFAULT_SPLITS = ("val",)
ARTIFACT_METADATA_KEYS = {
    "regime",
    "run_index",
    "run_index_i",
    "run_index_j",
    "run_id",
    "run_id_i",
    "run_id_j",
    "seed",
    "seed_i",
    "seed_j",
    "checkpoint",
    "checkpoint_i",
    "checkpoint_j",
    "checkpoint_kind",
    "probe_name",
    "split",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize and visualize saved landscape probe artifacts."
    )
    p.add_argument("--landscape-root", type=Path, default=Path("landscape_runs"))
    p.add_argument("--outdir", type=Path, default=Path("landscape_analysis"))
    p.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    p.add_argument(
        "--checkpoint-kind",
        choices=["all", *CHECKPOINT_KINDS],
        default="best_val",
    )
    p.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    p.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metric-column allowlist for summaries/differences/figures.",
    )
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--control-regime", default="regimeA")
    p.add_argument("--representative-policy", choices=["median"], default="median")
    p.add_argument("--make-figures", dest="make_figures", action="store_true", default=True)
    p.add_argument("--no-figures", dest="make_figures", action="store_false")
    return p.parse_args()


def resolve_checkpoint_kinds(value: str) -> list[str]:
    return list(CHECKPOINT_KINDS) if value == "all" else [value]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return item
    return value


def text_value(value: Any) -> str:
    val = scalar(value)
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def float_value(value: Any) -> float:
    try:
        return float(scalar(value))
    except (TypeError, ValueError):
        return np.nan


def sanitize_token(value: str) -> str:
    value = value.strip()
    value = value.replace("-", "neg")
    value = value.replace("+", "")
    value = value.replace(".", "p")
    value = value.replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")


def float_token(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return sanitize_token(f"{value:g}")


def descriptor_key(prefix: str, raw_key: str) -> str:
    parts = raw_key.split("_")
    out = [sanitize_token(part) if part.startswith("rel") else part for part in parts]
    return f"{prefix}_{'_'.join(out)}"


def add_missing(
    rows: list[dict[str, Any]],
    *,
    regime: str,
    checkpoint_kind: str,
    split: str,
    artifact_kind: str,
    path: Path,
    reason: str,
    run_id: str = "",
    run_index: Any = "",
) -> None:
    rows.append(
        {
            "regime": regime,
            "checkpoint_kind": checkpoint_kind,
            "split": split,
            "artifact_kind": artifact_kind,
            "run_id": run_id,
            "run_index": run_index,
            "path": path,
            "reason": reason,
        }
    )


def try_load_npz(
    path: Path,
    missing_rows: list[dict[str, Any]],
    *,
    regime: str,
    checkpoint_kind: str,
    split: str,
    artifact_kind: str,
    run_id: str = "",
    run_index: Any = "",
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        add_missing(
            missing_rows,
            regime=regime,
            checkpoint_kind=checkpoint_kind,
            split=split,
            artifact_kind=artifact_kind,
            path=path,
            reason="missing",
            run_id=run_id,
            run_index=run_index,
        )
        return None

    try:
        payload = load_npz(path)
    except Exception as exc:
        add_missing(
            missing_rows,
            regime=regime,
            checkpoint_kind=checkpoint_kind,
            split=split,
            artifact_kind=artifact_kind,
            path=path,
            reason=f"malformed:{type(exc).__name__}",
            run_id=run_id,
            run_index=run_index,
        )
        return None

    if not payload:
        add_missing(
            missing_rows,
            regime=regime,
            checkpoint_kind=checkpoint_kind,
            split=split,
            artifact_kind=artifact_kind,
            path=path,
            reason="empty",
            run_id=run_id,
            run_index=run_index,
        )
        return None

    return payload


def read_manifest(regime_root: Path) -> list[dict[str, str]]:
    path = regime_root / "manifest.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def discover_runs(regime_root: Path) -> list[dict[str, str]]:
    manifest_rows = read_manifest(regime_root)
    if manifest_rows:
        return manifest_rows

    rows: list[dict[str, str]] = []
    for idx, child in enumerate(sorted(regime_root.iterdir()) if regime_root.exists() else []):
        if not child.is_dir() or child.name in {"pairs", "subsets"}:
            continue
        rows.append(
            {
                "run_index": str(idx),
                "run_id": child.name,
                "seed": "",
                "path": str(child),
                "timestamp": "",
                "hash": "",
            }
        )
    return rows


def flatten_endpoint(row: dict[str, Any], payload: Mapping[str, np.ndarray], split: str) -> None:
    prefix = f"{split}_"
    for key, value in payload.items():
        if key.startswith(prefix):
            row[f"endpoint_{key[len(prefix):]}"] = float_value(value)

    for key in ("param_norm", "num_params"):
        if key in payload:
            row[key] = float_value(payload[key])


def flatten_perturbation(row: dict[str, Any], payload: Mapping[str, np.ndarray]) -> None:
    radii = np.asarray(payload.get("radii", []), dtype=np.float64)
    for stat in ("mean_delta", "median_delta", "p90_delta", "max_delta"):
        values = np.asarray(payload.get(stat, []), dtype=np.float64)
        for radius, val in zip(radii, values):
            col = f"perturb_{stat}_loss_r_{float_token(float(radius))}"
            row[col] = float(val)

    for key in ("base_loss", "auc_mean_delta"):
        if key in payload:
            row[f"perturb_{key}"] = float_value(payload[key])


def flatten_curve_descriptors(
    row: dict[str, Any],
    payload: Mapping[str, np.ndarray],
    *,
    prefix: str,
    skip: set[str],
) -> None:
    for key, value in payload.items():
        if key in skip or key in ARTIFACT_METADATA_KEYS:
            continue
        arr = np.asarray(value)
        if arr.shape == () and is_numeric_value(scalar(value)):
            row[descriptor_key(prefix, key)] = float_value(value)


def flatten_gradient(row: dict[str, Any], payload: Mapping[str, np.ndarray]) -> None:
    comps = [str(x) for x in np.asarray(payload.get("component_names", []), dtype=str).tolist()]
    norms = np.asarray(payload.get("grad_norm", []), dtype=np.float64)
    norms_sq = np.asarray(payload.get("grad_norm_sq", []), dtype=np.float64)
    for comp, norm, norm_sq in zip(comps, norms, norms_sq):
        safe = sanitize_token(comp)
        row[f"gradnorm_norm_{safe}"] = float(norm)
        row[f"gradnorm_norm_sq_{safe}"] = float(norm_sq)
    if "num_params" in payload:
        row["gradnorm_num_params"] = float_value(payload["num_params"])


def flatten_hessian(row: dict[str, Any], payload: Mapping[str, np.ndarray]) -> None:
    comps = [str(x) for x in np.asarray(payload.get("component_names", []), dtype=str).tolist()]
    statuses = [str(x) for x in np.asarray(payload.get("status", []), dtype=str).tolist()]
    top = np.asarray(payload.get("top_eigenvalue", []), dtype=np.float64)
    trace = np.asarray(payload.get("trace", []), dtype=np.float64)
    trace_std = np.asarray(payload.get("trace_std", []), dtype=np.float64)
    trace_stderr = np.asarray(payload.get("trace_stderr", []), dtype=np.float64)

    for idx, comp in enumerate(comps):
        safe = sanitize_token(comp)
        if idx < len(statuses):
            row[f"hessian_status_{safe}"] = statuses[idx]
        if idx < len(top):
            row[f"hessian_top_{safe}"] = float(top[idx])
        if idx < len(trace):
            row[f"hessian_trace_{safe}"] = float(trace[idx])
        if idx < len(trace_std):
            row[f"hessian_trace_std_{safe}"] = float(trace_std[idx])
        if idx < len(trace_stderr):
            row[f"hessian_trace_stderr_{safe}"] = float(trace_stderr[idx])
            denom = max(abs(float(trace[idx])) if idx < len(trace) else 0.0, 1e-12)
            rel = float(trace_stderr[idx]) / denom
            row[f"hessian_trace_rel_stderr_{safe}"] = rel
            row[f"hessian_trace_unstable_{safe}"] = int(np.isfinite(rel) and rel > 1.0)

    for key in ("num_params", "power_iters", "power_restarts", "trace_samples", "max_batches"):
        if key in payload:
            row[f"hessian_{key}"] = float_value(payload[key])


def flatten_interpolation(row: dict[str, Any], payload: Mapping[str, np.ndarray]) -> None:
    if "distance" in payload:
        row["distance"] = float_value(payload["distance"])
    flatten_curve_descriptors(
        row,
        payload,
        prefix="interp",
        skip={"alphas", "metric_names", "metric_values", "distance"},
    )


def collect_seed_rows(
    *,
    landscape_root: Path,
    regimes: Iterable[str],
    checkpoint_kinds: Iterable[str],
    splits: Iterable[str],
    missing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime in regimes:
        regime_root = landscape_root / regime
        for run in discover_runs(regime_root):
            run_id = run.get("run_id", "")
            run_index = run.get("run_index", "")
            seed = run.get("seed", "")
            for checkpoint_kind in checkpoint_kinds:
                seed_root = regime_root / run_id / checkpoint_kind
                endpoint = try_load_npz(
                    seed_root / "endpoint.npz",
                    missing_rows,
                    regime=regime,
                    checkpoint_kind=checkpoint_kind,
                    split="",
                    artifact_kind="endpoint",
                    run_id=run_id,
                    run_index=run_index,
                )
                for split in splits:
                    row: dict[str, Any] = {
                        "table": "seed",
                        "regime": regime,
                        "run_id": run_id,
                        "run_index": run_index,
                        "seed": seed,
                        "checkpoint_kind": checkpoint_kind,
                        "split": split,
                    }
                    if endpoint is not None:
                        flatten_endpoint(row, endpoint, split)

                    artifact_specs = [
                        ("perturbation", f"perturbation_{split}.npz", flatten_perturbation),
                        (
                            "slice1d",
                            f"slice1d_random_{split}.npz",
                            lambda r, p: flatten_curve_descriptors(
                                r,
                                p,
                                prefix="slice1d",
                                skip={"alphas", "metric_names", "metric_values"},
                            ),
                        ),
                        ("gradnorm", f"gradnorm_{split}.npz", flatten_gradient),
                        ("hessian", f"hessian_{split}.npz", flatten_hessian),
                        (
                            "slice2d",
                            f"slice2d_random_{split}.npz",
                            lambda r, p: flatten_curve_descriptors(
                                r,
                                p,
                                prefix="slice2d",
                                skip={"alphas", "betas", "metric_names", "metric_values"},
                            ),
                        ),
                    ]
                    for artifact_kind, filename, flattener in artifact_specs:
                        payload = try_load_npz(
                            seed_root / filename,
                            missing_rows,
                            regime=regime,
                            checkpoint_kind=checkpoint_kind,
                            split=split,
                            artifact_kind=artifact_kind,
                            run_id=run_id,
                            run_index=run_index,
                        )
                        if payload is not None:
                            flattener(row, payload)
                    rows.append(row)
    return rows


def collect_pair_rows(
    *,
    landscape_root: Path,
    regimes: Iterable[str],
    checkpoint_kinds: Iterable[str],
    splits: Iterable[str],
    missing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"run_(?P<i>\d+)__run_(?P<j>\d+)__interpolation_(?P<split>.+)\.npz$")

    for regime in regimes:
        for checkpoint_kind in checkpoint_kinds:
            pair_root = landscape_root / regime / "pairs" / checkpoint_kind
            if not pair_root.exists():
                add_missing(
                    missing_rows,
                    regime=regime,
                    checkpoint_kind=checkpoint_kind,
                    split="",
                    artifact_kind="pair_dir",
                    path=pair_root,
                    reason="missing",
                )
                continue

            for path in sorted(pair_root.glob("run_*__run_*__interpolation_*.npz")):
                m = pattern.match(path.name)
                if not m:
                    continue
                split = m.group("split")
                if split not in set(splits):
                    continue

                payload = try_load_npz(
                    path,
                    missing_rows,
                    regime=regime,
                    checkpoint_kind=checkpoint_kind,
                    split=split,
                    artifact_kind="interpolation",
                    run_index=f"{m.group('i')}:{m.group('j')}",
                )
                if payload is None:
                    continue

                row: dict[str, Any] = {
                    "table": "pair",
                    "regime": regime,
                    "run_id_i": text_value(payload.get("run_id_i", "")),
                    "run_id_j": text_value(payload.get("run_id_j", "")),
                    "run_index_i": int(m.group("i")),
                    "run_index_j": int(m.group("j")),
                    "seed_i": text_value(payload.get("seed_i", "")),
                    "seed_j": text_value(payload.get("seed_j", "")),
                    "checkpoint_kind": checkpoint_kind,
                    "split": split,
                }
                flatten_interpolation(row, payload)
                rows.append(row)
    return rows


def is_numeric_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.isfinite(float(value))
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    excluded = set(IDENTITY_COLUMNS)
    cols: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in excluded or key.startswith("hessian_status_"):
                continue
            if is_numeric_value(value):
                cols.add(key)
    return sorted(cols)


def value_array(rows: list[dict[str, Any]], column: str) -> np.ndarray:
    values = [float(row[column]) for row in rows if is_numeric_value(row.get(column))]
    return np.asarray(values, dtype=np.float64)


def summarize_rows(rows: list[dict[str, Any]], metric_columns: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("regime", ""), row.get("checkpoint_kind", ""), row.get("split", ""))].append(row)

    out: list[dict[str, Any]] = []
    for (regime, checkpoint_kind, split), group_rows in sorted(groups.items()):
        for metric in metric_columns:
            arr = value_array(group_rows, metric)
            if arr.size == 0:
                continue
            out.append(
                {
                    "regime": regime,
                    "checkpoint_kind": checkpoint_kind,
                    "split": split,
                    "metric": metric,
                    "n": int(arr.size),
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            )
    return out


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    stat: str,
    samples: int,
    ci_level: float,
    seed: int = 0,
) -> tuple[float, float]:
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    if samples <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    diffs = np.empty(samples, dtype=np.float64)
    fn = np.mean if stat == "mean" else np.median
    for i in range(samples):
        aa = rng.choice(a, size=a.size, replace=True)
        bb = rng.choice(b, size=b.size, replace=True)
        diffs[i] = float(fn(aa) - fn(bb))

    alpha = 1.0 - ci_level
    return (
        float(np.percentile(diffs, 100.0 * alpha / 2.0)),
        float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0))),
    )


def regime_differences(
    *,
    seed_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    seed_metrics: list[str],
    pair_metrics: list[str],
    control_regime: str,
    bootstrap_samples: int,
    ci_level: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for table_name, rows, metrics in [
        ("seed", seed_rows, seed_metrics),
        ("pair", pair_rows, pair_metrics),
    ]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row.get("checkpoint_kind", ""), row.get("split", ""))].append(row)

        for (checkpoint_kind, split), group_rows in sorted(groups.items()):
            regimes = sorted({str(row.get("regime", "")) for row in group_rows})
            control_rows = [row for row in group_rows if row.get("regime") == control_regime]
            for regime in regimes:
                if regime == control_regime:
                    continue
                regime_rows = [row for row in group_rows if row.get("regime") == regime]
                for metric in metrics:
                    a = value_array(regime_rows, metric)
                    b = value_array(control_rows, metric)
                    if a.size == 0 or b.size == 0:
                        continue
                    mean_ci = bootstrap_diff_ci(
                        a,
                        b,
                        stat="mean",
                        samples=bootstrap_samples,
                        ci_level=ci_level,
                        seed=stable_seed(table_name, checkpoint_kind, split, regime, metric, "mean"),
                    )
                    median_ci = bootstrap_diff_ci(
                        a,
                        b,
                        stat="median",
                        samples=bootstrap_samples,
                        ci_level=ci_level,
                        seed=stable_seed(table_name, checkpoint_kind, split, regime, metric, "median"),
                    )
                    out.append(
                        {
                            "table": table_name,
                            "checkpoint_kind": checkpoint_kind,
                            "split": split,
                            "metric": metric,
                            "regime": regime,
                            "control_regime": control_regime,
                            "n_regime": int(a.size),
                            "n_control": int(b.size),
                            "mean_regime": float(a.mean()),
                            "mean_control": float(b.mean()),
                            "mean_diff_vs_control": float(a.mean() - b.mean()),
                            "mean_diff_ci_low": mean_ci[0],
                            "mean_diff_ci_high": mean_ci[1],
                            "median_regime": float(np.median(a)),
                            "median_control": float(np.median(b)),
                            "median_diff_vs_control": float(np.median(a) - np.median(b)),
                            "median_diff_ci_low": median_ci[0],
                            "median_diff_ci_high": median_ci[1],
                            "bootstrap_note": "pair rows are not independent formal samples" if table_name == "pair" else "",
                        }
                    )
    return out


def stable_seed(*parts: str) -> int:
    text = "::".join(str(p) for p in parts)
    total = 0
    for ch in text:
        total = (total * 131 + ord(ch)) % (2**32 - 1)
    return total


def hessian_status_counts(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in seed_rows:
        for key, value in row.items():
            if not key.startswith("hessian_status_"):
                continue
            component = key[len("hessian_status_") :]
            group = (
                str(row.get("regime", "")),
                str(row.get("checkpoint_kind", "")),
                str(row.get("split", "")),
                component,
            )
            counters[group][str(value)] += 1

    out: list[dict[str, Any]] = []
    for (regime, checkpoint_kind, split, component), counter in sorted(counters.items()):
        total = sum(counter.values())
        for status, count in sorted(counter.items()):
            out.append(
                {
                    "regime": regime,
                    "checkpoint_kind": checkpoint_kind,
                    "split": split,
                    "component": component,
                    "status": status,
                    "count": count,
                    "fraction": count / total if total else np.nan,
                }
            )
    return out


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns_if_empty: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ordered_columns(rows) if rows else (columns_if_empty or [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in columns})


def ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    all_cols: set[str] = set()
    for row in rows:
        all_cols.update(row.keys())
    ordered = [col for col in IDENTITY_COLUMNS if col in all_cols]
    ordered.extend(sorted(col for col in all_cols if col not in set(ordered)))
    return ordered


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        if math.isnan(val):
            return "nan"
        if math.isinf(val):
            return "inf" if val > 0 else "-inf"
        return f"{val:.12g}"
    if isinstance(value, Path):
        return str(value)
    return value


def filter_metric_columns(columns: list[str], requested: list[str] | None) -> list[str]:
    if requested is None:
        return columns
    requested_set = set(requested)
    return [col for col in columns if col in requested_set]


def plot_distribution(
    rows: list[dict[str, Any]],
    *,
    column: str,
    outpath: Path,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = sorted({str(row.get("regime", "")) for row in rows})
    arrays = [value_array([row for row in rows if row.get("regime") == regime], column) for regime in regimes]
    pairs = [(regime, arr) for regime, arr in zip(regimes, arrays) if arr.size > 0]
    if not pairs:
        return

    labels, values = zip(*pairs)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4.2))
    ax.boxplot(values, labels=labels, showfliers=False)
    rng = np.random.default_rng(0)
    for idx, arr in enumerate(values, start=1):
        jitter = rng.normal(0.0, 0.035, size=arr.size)
        ax.scatter(np.full(arr.size, idx) + jitter, arr, s=18, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def metric_index(payload: Mapping[str, np.ndarray], metric: str) -> int | None:
    names = [str(x) for x in np.asarray(payload.get("metric_names", []), dtype=str).tolist()]
    try:
        return names.index(metric)
    except ValueError:
        return None


def choose_representative(rows: list[dict[str, Any]], column: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if is_numeric_value(row.get(column))]
    if not candidates:
        return sorted(rows, key=lambda r: str(r))[:1][0] if rows else None
    vals = np.asarray([float(row[column]) for row in candidates], dtype=np.float64)
    target = float(np.median(vals))
    idx = int(np.argmin(np.abs(vals - target)))
    return candidates[idx]


def plot_representative_slice1d(
    *,
    seed_rows: list[dict[str, Any]],
    landscape_root: Path,
    outpath: Path,
    metric: str = "loss",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_regime[str(row.get("regime", ""))].append(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    plotted = False
    for regime in sorted(by_regime):
        rep = choose_representative(by_regime[regime], f"slice1d_max_delta_{sanitize_token(metric)}")
        if rep is None:
            continue
        path = (
            landscape_root
            / regime
            / str(rep.get("run_id", ""))
            / str(rep.get("checkpoint_kind", ""))
            / f"slice1d_random_{rep.get('split', '')}.npz"
        )
        if not path.exists():
            continue
        payload = load_npz(path)
        idx = metric_index(payload, metric)
        if idx is None:
            continue
        x = np.asarray(payload["alphas"], dtype=np.float64)
        y = np.asarray(payload["metric_values"], dtype=np.float64)[:, idx]
        ax.plot(x, y, label=regime)
        plotted = True

    if not plotted:
        plt.close(fig)
        return
    ax.set_title(f"Representative 1D Slice ({metric})")
    ax.set_xlabel("alpha")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_representative_interpolation(
    *,
    pair_rows: list[dict[str, Any]],
    landscape_root: Path,
    outpath: Path,
    metric: str = "recon_img",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_regime[str(row.get("regime", ""))].append(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    plotted = False
    for regime in sorted(by_regime):
        rep = choose_representative(by_regime[regime], f"interp_barrier_{sanitize_token(metric)}")
        if rep is None:
            continue
        path = (
            landscape_root
            / regime
            / "pairs"
            / str(rep.get("checkpoint_kind", ""))
            / f"run_{rep.get('run_index_i')}__run_{rep.get('run_index_j')}__interpolation_{rep.get('split', '')}.npz"
        )
        if not path.exists():
            continue
        payload = load_npz(path)
        idx = metric_index(payload, metric)
        if idx is None:
            continue
        x = np.asarray(payload["alphas"], dtype=np.float64)
        y = np.asarray(payload["metric_values"], dtype=np.float64)[:, idx]
        ax.plot(x, y, label=regime)
        plotted = True

    if not plotted:
        plt.close(fig)
        return
    ax.set_title(f"Representative Interpolation ({metric})")
    ax.set_xlabel("alpha")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def make_figures(
    *,
    seed_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    landscape_root: Path,
    figures_dir: Path,
) -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        print("[warn] matplotlib is not installed; skipping figures")
        return

    plot_specs = [
        (
            seed_rows,
            "hessian_top_recon_img",
            "local_hessian_top_recon_img_by_regime.png",
            "Top Hessian Eigenvalue (recon_img)",
            "top eigenvalue",
        ),
        (
            seed_rows,
            "hessian_trace_recon_img",
            "local_hessian_trace_recon_img_by_regime.png",
            "Hessian Trace Estimate (recon_img)",
            "trace",
        ),
        (
            seed_rows,
            "gradnorm_norm_total",
            "local_gradnorm_total_by_regime.png",
            "Gradient Norm (total objective)",
            "gradient norm",
        ),
        (
            seed_rows,
            largest_radius_column(seed_rows, "perturb_p90_delta_loss_r_"),
            "local_perturb_p90_loss_by_regime.png",
            "Perturbation p90 Delta (loss)",
            "loss increase",
        ),
        (
            pair_rows,
            "interp_barrier_recon_img",
            "pair_interp_barrier_recon_img_by_regime.png",
            "Interpolation Barrier (recon_img)",
            "barrier",
        ),
        (
            pair_rows,
            "interp_area_excess_recon_img",
            "pair_interp_area_excess_recon_img_by_regime.png",
            "Interpolation Area Excess (recon_img)",
            "area excess",
        ),
        (
            pair_rows,
            "distance",
            "pair_distance_by_regime.png",
            "Pairwise Parameter Distance",
            "distance",
        ),
    ]
    for rows, column, filename, title, ylabel in plot_specs:
        if column:
            plot_distribution(
                rows,
                column=column,
                outpath=figures_dir / filename,
                title=title,
                ylabel=ylabel,
            )

    plot_representative_slice1d(
        seed_rows=seed_rows,
        landscape_root=landscape_root,
        outpath=figures_dir / "representative_slice1d_loss_by_regime.png",
        metric="loss",
    )
    plot_representative_interpolation(
        pair_rows=pair_rows,
        landscape_root=landscape_root,
        outpath=figures_dir / "representative_interp_recon_img_by_regime.png",
        metric="recon_img",
    )


def largest_radius_column(rows: list[dict[str, Any]], prefix: str) -> str:
    candidates: list[tuple[float, str]] = []
    for col in numeric_columns(rows):
        if not col.startswith(prefix):
            continue
        token = col[len(prefix) :]
        try:
            radius = float(token.replace("neg", "-").replace("p", "."))
        except ValueError:
            radius = -np.inf
        candidates.append((radius, col))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1]


def main() -> None:
    args = parse_args()
    checkpoint_kinds = resolve_checkpoint_kinds(args.checkpoint_kind)
    tables_dir = args.outdir / "tables"
    figures_dir = args.outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    missing_rows: list[dict[str, Any]] = []
    seed_rows = collect_seed_rows(
        landscape_root=args.landscape_root,
        regimes=args.regimes,
        checkpoint_kinds=checkpoint_kinds,
        splits=args.splits,
        missing_rows=missing_rows,
    )
    pair_rows = collect_pair_rows(
        landscape_root=args.landscape_root,
        regimes=args.regimes,
        checkpoint_kinds=checkpoint_kinds,
        splits=args.splits,
        missing_rows=missing_rows,
    )

    seed_metrics = filter_metric_columns(numeric_columns(seed_rows), args.metrics)
    pair_metrics = filter_metric_columns(numeric_columns(pair_rows), args.metrics)

    seed_summary = summarize_rows(seed_rows, seed_metrics)
    pair_summary = summarize_rows(pair_rows, pair_metrics)
    differences = regime_differences(
        seed_rows=seed_rows,
        pair_rows=pair_rows,
        seed_metrics=seed_metrics,
        pair_metrics=pair_metrics,
        control_regime=args.control_regime,
        bootstrap_samples=args.bootstrap_samples,
        ci_level=args.ci_level,
    )
    hessian_counts = hessian_status_counts(seed_rows)

    write_csv(
        tables_dir / "seed_table.csv",
        seed_rows,
        columns_if_empty=["table", "regime", "run_id", "run_index", "seed", "checkpoint_kind", "split"],
    )
    write_csv(
        tables_dir / "pair_table.csv",
        pair_rows,
        columns_if_empty=[
            "table",
            "regime",
            "run_id_i",
            "run_id_j",
            "run_index_i",
            "run_index_j",
            "seed_i",
            "seed_j",
            "checkpoint_kind",
            "split",
        ],
    )
    summary_columns = [
        "regime",
        "checkpoint_kind",
        "split",
        "metric",
        "n",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "min",
        "max",
    ]
    write_csv(tables_dir / "seed_summary.csv", seed_summary, columns_if_empty=summary_columns)
    write_csv(tables_dir / "pair_summary.csv", pair_summary, columns_if_empty=summary_columns)
    write_csv(
        tables_dir / "regime_differences.csv",
        differences,
        columns_if_empty=[
            "table",
            "checkpoint_kind",
            "split",
            "metric",
            "regime",
            "control_regime",
            "n_regime",
            "n_control",
            "mean_regime",
            "mean_control",
            "mean_diff_vs_control",
            "mean_diff_ci_low",
            "mean_diff_ci_high",
            "median_regime",
            "median_control",
            "median_diff_vs_control",
            "median_diff_ci_low",
            "median_diff_ci_high",
            "bootstrap_note",
        ],
    )
    write_csv(
        tables_dir / "missing_artifacts.csv",
        missing_rows,
        columns_if_empty=[
            "regime",
            "checkpoint_kind",
            "split",
            "artifact_kind",
            "run_id",
            "run_index",
            "path",
            "reason",
        ],
    )
    write_csv(
        tables_dir / "hessian_status_counts.csv",
        hessian_counts,
        columns_if_empty=[
            "regime",
            "checkpoint_kind",
            "split",
            "component",
            "status",
            "count",
            "fraction",
        ],
    )

    if args.make_figures:
        make_figures(
            seed_rows=seed_rows,
            pair_rows=pair_rows,
            landscape_root=args.landscape_root,
            figures_dir=figures_dir,
        )

    print(f"[write] {tables_dir / 'seed_table.csv'} ({len(seed_rows)} rows)")
    print(f"[write] {tables_dir / 'pair_table.csv'} ({len(pair_rows)} rows)")
    print(f"[write] {tables_dir / 'regime_differences.csv'} ({len(differences)} rows)")
    if missing_rows:
        print(f"[warn] {len(missing_rows)} missing/malformed artifacts recorded")


if __name__ == "__main__":
    main()
