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
from models.common import Encoder
from models.plain_vae import PlainVAE, PlainVAEConfig
from models.sparse_vae import SparseVAE, SparseVAEConfig


def build_plain_from_config(model_config: dict[str, Any]) -> PlainVAE:
    return PlainVAE(PlainVAEConfig(**model_config))


def build_sparse_from_config(model_config: dict[str, Any]) -> SparseVAE:
    cfg = SparseVAEConfig(**model_config)
    encoder = Encoder(
        image_shape=tuple(cfg.image_shape),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        anchor_dim=cfg.anchor_dim,
    )

    return SparseVAE(
        encoder=encoder,
        latent_dim=cfg.latent_dim,
        image_shape=tuple(cfg.image_shape),
        hidden_dim=cfg.hidden_dim,
        anchor_dim=cfg.anchor_dim,
        sparse_lambda=cfg.sparse_lambda,
        feature_hidden_dim=cfg.feature_hidden_dim,
        anchor_hidden_dim=cfg.anchor_hidden_dim,
        gate_temperature=cfg.gate_temperature,
        cfg=cfg,
    )


def build_causal_discrepancy_from_config(
    model_config: dict[str, Any],
) -> CausalDiscrepancyVAE:
    cfg = CausalDiscrepancyVAEConfig(**model_config)
    if cfg.num_intervention_envs <= 0:
        raise ValueError(
            "CausalDiscrepancyVAE checkpoint config must include "
            "num_intervention_envs > 0; cannot infer intervention embedding "
            "shape during checkpoint loading."
        )

    encoder = Encoder(
        image_shape=tuple(cfg.image_shape),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        anchor_dim=cfg.anchor_dim,
    )

    return CausalDiscrepancyVAE(
        encoder=encoder,
        latent_dim=cfg.latent_dim,
        image_shape=tuple(cfg.image_shape),
        hidden_dim=cfg.hidden_dim,
        anchor_dim=cfg.anchor_dim,
        num_intervention_variants=cfg.num_intervention_variants,
        num_intervention_envs=cfg.num_intervention_envs,
        mmd_weight=cfg.mmd_weight,
        graph_l1_weight=cfg.graph_l1_weight,
        decoder_num_hidden_layers=cfg.decoder_num_hidden_layers,
        anchor_hidden_dim=cfg.anchor_hidden_dim,
        mmd_bandwidths=cfg.mmd_bandwidths,
        cfg=cfg,
    )


def build_model_from_checkpoint_payload(payload: dict[str, Any]) -> nn.Module:
    model_name = payload["model_name"]
    model_config = payload["model_config"]

    if model_config is None:
        raise ValueError(f"{model_name} checkpoint is missing model_config")

    if model_name == "PlainVAE":
        return build_plain_from_config(model_config)

    if model_name == "SparseVAE":
        return build_sparse_from_config(model_config)

    if model_name == "CausalDiscrepancyVAE":
        return build_causal_discrepancy_from_config(model_config)

    raise ValueError(f"Unknown model_name in checkpoint: {model_name}")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> nn.Module:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model_from_checkpoint_payload(payload)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model
