# models/build.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.plain_vae import PlainVAE, PlainVAEConfig


def build_model_from_checkpoint_payload(payload: dict[str, Any]) -> nn.Module:
    model_name = payload["model_name"]
    model_config = payload["model_config"]

    if model_name == "PlainVAE":
        if model_config is None:
            raise ValueError("PlainVAE checkpoint is missing model_config")
        cfg = PlainVAEConfig(**model_config)
        return PlainVAE(cfg)

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
