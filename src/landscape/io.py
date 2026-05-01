# landscape/io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_dir(root: str | Path, regime: str, run_id: str, checkpoint_kind: str) -> Path:
    return ensure_dir(Path(root) / regime / run_id / checkpoint_kind)


def pair_dir(root: str | Path, regime: str, checkpoint_kind: str) -> Path:
    return ensure_dir(Path(root) / regime / "pairs" / checkpoint_kind)


def subset_dir(root: str | Path, regime: str) -> Path:
    return ensure_dir(Path(root) / regime / "subsets")


def save_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    for key, value in arrays.items():
        payload[key] = _to_npz_value(value)

    np.savez_compressed(path, **payload)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def save_summary_row(path: str | Path, row: Mapping[str, Any]) -> None:
    """
    Append one CSV row.

    This is intentionally simple. Header is written only if the file does not exist.
    Values are converted with str(); avoid commas in string fields.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(row.keys())
    exists = path.exists()

    with path.open("a", encoding="utf-8") as fh:
        if not exists:
            fh.write(",".join(keys) + "\n")
        fh.write(",".join(_csv_value(row[k]) for k in keys) + "\n")


def save_summary_table(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must be nonempty")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            if list(row.keys()) != keys:
                raise ValueError("all rows must have identical key order")
            fh.write(",".join(_csv_value(row[k]) for k in keys) + "\n")


def load_summary_table(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        lines = [line.rstrip("\n") for line in fh if line.strip()]

    if not lines:
        return []

    header = lines[0].split(",")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        vals = line.split(",")
        if len(vals) != len(header):
            raise ValueError(f"Malformed CSV row in {path}: {line}")
        rows.append(dict(zip(header, vals)))

    return rows

def checkpoint_path(
    run_dir: str | Path,
    checkpoint_kind: str,
) -> Path:
    """
    Resolve checkpoint path inside a single run directory.

    Expected structure:
        runs/regimeX/hash-.../
            last.pt
            best_val_loss.pt

    checkpoint_kind:
        - "final"    -> last.pt
        - "best_val" -> best_val_loss.pt (fallback: best_loss.pt)
    """
    run_dir = Path(run_dir)

    if checkpoint_kind == "final":
        primary = run_dir / "final.pt"
        if primary.exists():
            return primary
        fallback = run_dir / "last.pt"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"Could not find final checkpoint in {run_dir}; expected final.pt"
        )


    if checkpoint_kind == "best_val":
        primary = run_dir / "best_val_loss.pt"
        if primary.exists():
            return primary

        fallback = run_dir / "best_loss.pt"  # backward compatibility
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            f"No best checkpoint found in {run_dir} "
            "(expected best_val_loss.pt or best_loss.pt)"
        )

    raise ValueError(
        f"Unknown checkpoint_kind={checkpoint_kind!r}; expected 'final' or 'best_val'"
    )

def pair_name(run_index_i: int, run_index_j: int, probe_name: str) -> str:
    a, b = sorted((run_index_i, run_index_j))
    return f"run_{a}__run_{b}__{probe_name}.npz"


def seed_probe_path(
    root: str | Path,
    regime: str,
    run_id: str,
    checkpoint_kind: str,
    probe_name: str,
) -> Path:
    return seed_dir(root, regime, run_id, checkpoint_kind) / f"{probe_name}.npz"


def pair_probe_path(
    root: str | Path,
    regime: str,
    run_index_i: int,
    run_index_j: int,
    checkpoint_kind: str,
    probe_name: str,
) -> Path:
    return pair_dir(root, regime, checkpoint_kind) / pair_name(run_index_i, run_index_j, probe_name)


def subset_path(
    root: str | Path,
    regime: str,
    split: str,
    n_examples: int,
    seed: int,
) -> Path:
    return subset_dir(root, regime) / f"{split}_n{n_examples}_seed{seed}.npz"


def _to_npz_value(value: Any) -> Any:
    if isinstance(value, Path):
        return np.array(str(value))
    if isinstance(value, str):
        return np.array(value)
    if isinstance(value, (int, float, bool)):
        return np.array(value)
    if value is None:
        return np.array("")
    return value


def _csv_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return str(value)
