# models/build.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.causal_discrepancy_vae import (
    CausalDiscrepancyVAE,
    CausalDiscrepancyVAEConfig,
)
from models.plain_vae import PlainVAE, PlainVAEConfig
from models.sparse_vae import SparseVAE, SparseVAEConfig


def build_model_from_checkpoint_payload(payload: dict[str, Any]) -> nn.Module:
    model_name = payload["model_name"]
    model_config = payload["model_config"]

    if model_config is None:
        raise ValueError(f"{model_name} checkpoint is missing model_config")

    if model_name == "PlainVAE":
        return PlainVAE(PlainVAEConfig(**model_config))

    if model_name == "SparseVAE":
        return SparseVAE(SparseVAEConfig(**model_config))

    if model_name == "CausalDiscrepancyVAE":
        return CausalDiscrepancyVAE(CausalDiscrepancyVAEConfig(**model_config))

    raise ValueError(f"Unknown model_name in checkpoint: {model_name}")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> nn.Module:
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model_from_checkpoint_payload(payload)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model
