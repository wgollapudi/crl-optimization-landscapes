from __future__ import annotations

import torch

from data import Batch
from models.causal_discrepancy_vae import CausalDiscrepancyVAE
from models.common import Encoder
from models.plain_vae import PlainVAE
from models.sparse_vae import SparseVAE


def make_dummy_batch(batch_size: int = 6, anchor_dim: int | None = None) -> Batch:
    x_anchor = None
    if anchor_dim is not None:
        x_anchor = torch.randn(batch_size, anchor_dim)

    env_id = torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.long)[:batch_size]
    intervention_target = torch.tensor([-1, 0, 0, 1, 2, 3], dtype=torch.long)[:batch_size]
    intervention_variant = torch.tensor([-1, 0, 1, 0, 1, 0], dtype=torch.long)[:batch_size]

    return Batch(
        x_img=torch.bernoulli(torch.full((batch_size, 1, 64, 64), 0.5)),
        x_anchor=x_anchor,
        env_id=env_id,
        intervention_target=intervention_target,
        intervention_variant=intervention_variant,
        z_true=None,
    )


def make_encoder(anchor_dim: int = 0) -> Encoder:
    return Encoder(
        image_shape=(1, 64, 64),
        latent_dim=4,
        hidden_dim=16,
        anchor_dim=anchor_dim,
    )


def assert_scalar(value: torch.Tensor, name: str) -> None:
    assert value.ndim == 0, f"{name} should be scalar, got {tuple(value.shape)}"
    assert torch.isfinite(value).all(), f"{name} should be finite"


def check_plain_vae() -> None:
    batch = make_dummy_batch(anchor_dim=None)
    model = PlainVAE(
        encoder=make_encoder(anchor_dim=0),
        latent_dim=4,
        image_shape=(1, 64, 64),
        hidden_dim=16,
        anchor_dim=0,
    )
    out = model(batch)
    assert out.recon.img_logits.shape == (6, 1, 64, 64)
    assert out.recon.anchor_mean is None
    print("PlainVAE forward OK")


def check_sparse_vae() -> None:
    batch = make_dummy_batch(anchor_dim=8)
    model = SparseVAE(
        encoder=make_encoder(anchor_dim=8),
        latent_dim=4,
        image_shape=(1, 64, 64),
        hidden_dim=16,
        anchor_dim=8,
        sparse_lambda=1e-3,
        feature_hidden_dim=8,
    )
    out = model(batch)
    assert out.recon.img_logits.shape == (6, 1, 64, 64)
    assert out.recon.anchor_mean is not None
    assert out.recon.anchor_mean.shape == (6, 8)
    assert "sparsity" in out.extras
    assert_scalar(out.extras["sparsity"], "sparsity")
    print("SparseVAE forward OK")


def check_causal_discrepancy_vae() -> None:
    batch = make_dummy_batch(anchor_dim=None)
    model = CausalDiscrepancyVAE(
        encoder=make_encoder(anchor_dim=0),
        latent_dim=4,
        image_shape=(1, 64, 64),
        hidden_dim=16,
        num_intervention_variants=2,
        mmd_weight=1.0,
        graph_l1_weight=1e-3,
    )
    out = model(batch)
    assert out.recon.img_logits.shape == (6, 1, 64, 64)
    assert out.recon.anchor_mean is None
    assert "mmd" in out.extras
    assert "graph_l1" in out.extras
    assert_scalar(out.extras["mmd"], "mmd")
    assert_scalar(out.extras["graph_l1"], "graph_l1")
    print("CausalDiscrepancyVAE forward OK")


def main() -> None:
    torch.manual_seed(0)
    check_plain_vae()
    check_sparse_vae()
    check_causal_discrepancy_vae()
    print("All model smoke checks passed.")


if __name__ == "__main__":
    main()
