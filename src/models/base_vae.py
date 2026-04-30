# models/base_vae.py
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from data import Batch
from models.common import Encoder, PosteriorParams


@dataclass
class Reconstruction:
    img_logits: Tensor
    anchor_mean: Tensor | None = None


@dataclass
class ModelOutput:
    posterior: PosteriorParams
    z: Tensor
    decoder_latent: Tensor
    recon: Reconstruction
    extras: dict[str, Tensor] = field(default_factory=dict)


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Per-example KL(q(z|x) || N(0, I)).

    Returns:
        kl: [B]
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=1)


class BaseVAE(nn.Module, ABC):
    """
    Shared VAE scaffold.

    The intended extension points are:
      - latent_to_decoder_input(): for causal layers / transformed latents
      - decode(): for regime-specific decoder structure
      - compute_aux_losses(): for sparsity, MMD, graph penalties, etc.

    This keeps the trainer and loss code generic.
    """

    def __init__(
        self,
        encoder: Encoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder

    def encode(self, batch: Batch) -> PosteriorParams:
        return self.encoder(batch.x_img, batch.x_anchor)

    def sample_latent(self, posterior: PosteriorParams, sample: bool = True) -> Tensor:
        if sample:
            return reparameterize(posterior.mu, posterior.logvar)
        return posterior.mu

    def latent_to_decoder_input(self, z: Tensor, batch: Batch) -> Tensor:
        """
        Identity by default.

        Sparse-VAE will probably keep this unchanged.
        CausalDiscrepancy-VAE will override this to map exogenous/noise variables
        through a latent causal layer before decoding.
        """
        return z

    def decode(self, decoder_latent: Tensor, batch: Batch) -> Reconstruction:
        raise NotImplementedError

    def compute_aux_losses(
        self,
        batch: Batch,
        out: ModelOutput,
    ) -> dict[str, Tensor]:
        """
        Optional regime-specific loss terms.

        Expected contract:
          - returned tensors should be scalar tensors
          - keys are descriptive names like "mmd", "graph_l1", "sparsity"
        """
        return {}

    def forward(self, batch: Batch, sample: bool = True) -> ModelOutput:
        posterior = self.encode(batch)
        z = self.sample_latent(posterior, sample=sample)
        decoder_latent = self.latent_to_decoder_input(z, batch)
        recon = self.decode(decoder_latent, batch)

        out = ModelOutput(
            posterior=posterior,
            z=z,
            decoder_latent=decoder_latent,
            recon=recon,
        )
        out.extras = self.compute_aux_losses(batch, out)
        return out
