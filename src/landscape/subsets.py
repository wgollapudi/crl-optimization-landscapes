# landscape/subsets.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import CRLDataset, Batch, _collate_samples


SplitName = Literal["train", "val", "test"]


def make_subset_indices(
    dataset_size: int,
    n_examples: int,
    seed: int,
    sort_indices: bool = True,
) -> np.ndarray:
    """
    Deterministically sample subset indices from [0, dataset_size).

    These indices are relative to the already split-specific dataset.
    """
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")

    if n_examples <= 0:
        raise ValueError("n_examples must be positive")

    n = min(n_examples, dataset_size)
    rng = np.random.default_rng(seed)
    idx = rng.choice(dataset_size, size=n, replace=False).astype(np.int64)

    if sort_indices:
        idx.sort()

    return idx


def save_subset_indices(
    path: str | Path,
    indices: np.ndarray,
    *,
    regime: str,
    split: SplitName,
    dataset_path: str | Path,
    seed: int,
    requested_n: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        indices=np.asarray(indices, dtype=np.int64),
        regime=np.array(regime),
        split=np.array(split),
        dataset_path=np.array(str(dataset_path)),
        seed=np.array(seed, dtype=np.int64),
        requested_n=np.array(requested_n, dtype=np.int64),
        actual_n=np.array(len(indices), dtype=np.int64),
    )


def load_subset_indices(path: str | Path) -> np.ndarray:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["indices"], dtype=np.int64)


def ensure_subset_indices(
    path: str | Path,
    dataset_size: int,
    n_examples: int,
    seed: int,
    *,
    regime: str,
    split: SplitName,
    dataset_path: str | Path,
    overwrite: bool = False,
) -> np.ndarray:
    """
    Load existing subset indices if present; otherwise create and save them.
    """
    path = Path(path)

    if path.exists() and not overwrite:
        idx = load_subset_indices(path)
        if len(idx) == 0:
            raise ValueError(f"Loaded empty subset indices from {path}")
        if idx.min() < 0 or idx.max() >= dataset_size:
            raise ValueError(
                f"Subset indices in {path} are invalid for dataset_size={dataset_size}"
            )
        return idx

    idx = make_subset_indices(
        dataset_size=dataset_size,
        n_examples=n_examples,
        seed=seed,
    )

    save_subset_indices(
        path,
        idx,
        regime=regime,
        split=split,
        dataset_path=dataset_path,
        seed=seed,
        requested_n=n_examples,
    )

    return idx


def build_subset_dataset(
    dataset_path: str | Path,
    split: SplitName,
    indices: np.ndarray,
    *,
    use_anchor_features: bool = False,
    include_metadata: bool = False,
) -> Subset[CRLDataset]:
    base = CRLDataset(
        npz_path=dataset_path,
        split=split,
        use_anchor_features=use_anchor_features,
        include_metadata=include_metadata,
    )

    if len(indices) == 0:
        raise ValueError("indices must be nonempty")
    if indices.min() < 0 or indices.max() >= len(base):
        raise ValueError(
            f"indices out of range for split={split}: "
            f"min={indices.min()}, max={indices.max()}, dataset_size={len(base)}"
        )

    return Subset(base, indices.tolist())


def build_subset_loader(
    dataset_path: str | Path,
    split: SplitName,
    indices: np.ndarray,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_anchor_features: bool = False,
    include_metadata: bool = False,
) -> DataLoader[Batch]:
    subset = build_subset_dataset(
        dataset_path=dataset_path,
        split=split,
        indices=indices,
        use_anchor_features=use_anchor_features,
        include_metadata=include_metadata,
    )

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_samples,
    )


def ensure_subset_loader(
    *,
    subset_path: str | Path,
    dataset_path: str | Path,
    regime: str,
    split: SplitName,
    n_examples: int,
    subset_seed: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_anchor_features: bool = False,
    include_metadata: bool = False,
    overwrite: bool = False,
) -> DataLoader[Batch]:
    """
    Convenience helper:
      - instantiate split dataset once to know its size
      - load/create fixed subset indices
      - return deterministic non-shuffled loader
    """
    base = CRLDataset(
        npz_path=dataset_path,
        split=split,
        use_anchor_features=use_anchor_features,
        include_metadata=False,
    )

    indices = ensure_subset_indices(
        path=subset_path,
        dataset_size=len(base),
        n_examples=n_examples,
        seed=subset_seed,
        regime=regime,
        split=split,
        dataset_path=dataset_path,
        overwrite=overwrite,
    )

    return build_subset_loader(
        dataset_path=dataset_path,
        split=split,
        indices=indices,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_anchor_features=use_anchor_features,
        include_metadata=include_metadata,
    )
