# data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import DataConfig


SPLIT_NAME_TO_ID = {
    "train": 0,
    "val": 1,
    "test": 2,
}


@dataclass
class Batch:
    x_img: Tensor
    x_anchor: Tensor | None
    env_id: Tensor
    intervention_target: Tensor
    intervention_variant: Tensor
    z_true: Tensor | None

    class_tuples: Tensor | None = None
    latents_quantized: Tensor | None = None
    orientation_cos_sin: Tensor | None = None


@dataclass
class Sample:
    x_img: Tensor
    x_anchor: Tensor | None
    env_id: Tensor
    intervention_target: Tensor
    intervention_variant: Tensor
    z_true: Tensor | None

    class_tuples: Tensor | None = None
    latents_quantized: Tensor | None = None
    orientation_cos_sin: Tensor | None = None


def _to_tensor(x: np.ndarray, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(x, dtype=dtype)


def _collate_samples(samples: list[Sample]) -> Batch:
    x_img = torch.stack([s.x_img for s in samples], dim=0)

    first_anchor = samples[0].x_anchor
    x_anchor = None if first_anchor is None else torch.stack(
        [s.x_anchor for s in samples], dim=0
    )

    first_z = samples[0].z_true
    z_true = None if first_z is None else torch.stack(
        [s.z_true for s in samples], dim=0
    )

    first_class = samples[0].class_tuples
    class_tuples = None if first_class is None else torch.stack(
        [s.class_tuples for s in samples], dim=0
    )

    first_quant = samples[0].latents_quantized
    latents_quantized = None if first_quant is None else torch.stack(
        [s.latents_quantized for s in samples], dim=0
    )

    first_orient = samples[0].orientation_cos_sin
    orientation_cos_sin = None if first_orient is None else torch.stack(
        [s.orientation_cos_sin for s in samples], dim=0
    )

    return Batch(
        x_img=x_img,
        x_anchor=x_anchor,
        env_id=torch.stack([s.env_id for s in samples], dim=0),
        intervention_target=torch.stack([s.intervention_target for s in samples], dim=0),
        intervention_variant=torch.stack([s.intervention_variant for s in samples], dim=0),
        z_true=z_true,
        class_tuples=class_tuples,
        latents_quantized=latents_quantized,
        orientation_cos_sin=orientation_cos_sin,
    )


class CRLDataset(Dataset):
    """
    Dataset wrapper around one regime .npz file.

    Expected fields from make_datasets.py:
    - images
    - class_tuples
    - latents_values
    - latents_true_continuous
    - latents_quantized_continuous
    - orientation_cos
    - orientation_sin
    - env_id
    - intervention_target
    - intervention_variant
    - split

    Future:
    - anchor_features -> used for regime B
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        use_anchor_features: bool = False,
        include_metadata: bool = False,
    ) -> None:
        super().__init__()

        if split not in SPLIT_NAME_TO_ID:
            raise ValueError(f"Unknown split '{split}'")

        self.path = Path(npz_path)
        self.split_name = split
        if split not in SPLIT_NAME_TO_ID:
            raise ValueError(f"Unknown split '{split}'. Expected one of {tuple(SPLIT_NAME_TO_ID)}")
        self.split_id = SPLIT_NAME_TO_ID[split]
        self.use_anchor_features = use_anchor_features
        self.include_metadata = include_metadata

        with np.load(self.path, allow_pickle=False) as data:
            split_arr = np.asarray(data["split"], dtype=np.int64)
            mask = split_arr == self.split_id

            images = np.asarray(data["images"])[mask]
            if images.ndim != 3:
                raise ValueError(
                    f"Expected `images` to have shape [N, H, W], got {images.shape}"
                )
            self.images = torch.as_tensor(images[:, None, :, :], dtype=torch.float32)

            self.env_id = _to_tensor(
                np.asarray(data["env_id"], dtype=np.int64)[mask], torch.long
            )
            self.intervention_target = _to_tensor(
                np.asarray(data["intervention_target"], dtype=np.int64)[mask], torch.long
            )
            self.intervention_variant = _to_tensor(
                np.asarray(data["intervention_variant"], dtype=np.int64)[mask], torch.long
            )

            if "latents_true_continuous" in data.files:
                z_true = np.asarray(data["latents_true_continuous"], dtype=np.float32)[mask]
                self.z_true: Tensor | None = _to_tensor(z_true, torch.float32)
            else:
                self.z_true = None

            if use_anchor_features:
                if "anchor_features" not in data.files:
                    raise ValueError(
                        f"use_anchor_features=True, but `{self.path}` has no `anchor_features` array"
                    )
                anchor = np.asarray(data["anchor_features"], dtype=np.float32)[mask]
                self.x_anchor: Tensor | None = _to_tensor(anchor, torch.float32)
            else:
                self.x_anchor = None

            if include_metadata and "class_tuples" in data.files:
                cls = np.asarray(data["class_tuples"], dtype=np.int64)[mask]
                self.class_tuples: Tensor | None = _to_tensor(cls, torch.long)
            else:
                self.class_tuples = None

            if include_metadata and "latents_quantized_continuous" in data.files:
                zq = np.asarray(data["latents_quantized_continuous"], dtype=np.float32)[mask]
                self.latents_quantized: Tensor | None = _to_tensor(zq, torch.float32)
            else:
                self.latents_quantized = None

            if include_metadata and {"orientation_cos", "orientation_sin"}.issubset(data.files):
                oc = np.asarray(data["orientation_cos"], dtype=np.float32)[mask]
                os = np.asarray(data["orientation_sin"], dtype=np.float32)[mask]
                orient = np.stack([oc, os], axis=1)
                self.orientation_cos_sin: Tensor | None = _to_tensor(orient, torch.float32)
            else:
                self.orientation_cos_sin = None

        if len(self.images) == 0:
            raise ValueError(f"No examples found for split='{split}' in {self.path}")

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Sample:
        return Sample(
            x_img=self.images[idx],
            x_anchor=None if self.x_anchor is None else self.x_anchor[idx],
            env_id=self.env_id[idx],
            intervention_target=self.intervention_target[idx],
            intervention_variant=self.intervention_variant[idx],
            z_true=None if self.z_true is None else self.z_true[idx],
            class_tuples=None if self.class_tuples is None else self.class_tuples[idx],
            latents_quantized=None if self.latents_quantized is None else self.latents_quantized[idx],
            orientation_cos_sin=None if self.orientation_cos_sin is None else self.orientation_cos_sin[idx],
        )


class DataModule:
    def __init__(self, cfg: DataConfig, include_metadata: bool = False) -> None:
        self.cfg = cfg
        self.include_metadata = include_metadata

        self.train_ds: CRLDataset | None = None
        self.val_ds: CRLDataset | None = None
        self.test_ds: CRLDataset | None = None

    def setup(self) -> None:
        self.train_ds = CRLDataset(
            self.cfg.path,
            split="train",
            use_anchor_features=self.cfg.use_anchor_features,
            include_metadata=self.include_metadata,
        )
        self.val_ds = CRLDataset(
            self.cfg.path,
            split="val",
            use_anchor_features=self.cfg.use_anchor_features,
            include_metadata=self.include_metadata,
        )
        self.test_ds = CRLDataset(
            self.cfg.path,
            split="test",
            use_anchor_features=self.cfg.use_anchor_features,
            include_metadata=self.include_metadata,
        )

    def train_loader(self) -> DataLoader:
        if self.train_ds is None:
            self.setup()
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=_collate_samples,
        )

    def val_loader(self) -> DataLoader:
        if self.val_ds is None:
            self.setup()
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=_collate_samples,
        )

    def test_loader(self) -> DataLoader:
        if self.test_ds is None:
            self.setup()
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=_collate_samples,
        )

    def summary(self) -> dict[str, Any]:
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            self.setup()
        assert self.train_ds is not None
        assert self.val_ds is not None
        assert self.test_ds is not None

        return {
            "path": str(self.cfg.path),
            "train_size": len(self.train_ds),
            "val_size": len(self.val_ds),
            "test_size": len(self.test_ds),
            "image_shape": tuple(self.train_ds.images.shape[1:]),
            "anchor_dim": None if self.train_ds.x_anchor is None else int(self.train_ds.x_anchor.shape[1]),
            "latent_dim": None if self.train_ds.z_true is None else int(self.train_ds.z_true.shape[1]),
        }
