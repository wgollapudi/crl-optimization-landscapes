from __future__ import annotations

import torch
from torch import Tensor, nn

from data import Batch
from models.base_vae import BaseVAE, ModelOutput, Reconstruction
from models.common import Encoder


class SparseFeatureDecoder(nn.Module):
    """
    Sparse per-feature decoder for dSprites images plus engineered anchors.

    This is a compact, project-compatible version of the sparse decoder idea in
    Moran et al.: each observed feature receives a feature-specific gate over
    latent dimensions before a shallow decoder produces that feature.  We use
    learned gates for image pixels and fixed one-hot gates for anchor features.
    """

    def __init__(
        self,
        latent_dim: int,
        image_shape: tuple[int, int, int],
        anchor_dim: int,
        feature_hidden_dim: int = 16,
        gate_temperature: float = 1.0,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if anchor_dim <= 0:
            raise ValueError("SparseVAE requires anchor_dim > 0")
        if anchor_dim % latent_dim != 0:
            raise ValueError(
                f"anchor_dim must be divisible by latent_dim, got anchor_dim={anchor_dim} "
                f"and latent_dim={latent_dim}"
            )
        if feature_hidden_dim <= 0:
            raise ValueError("feature_hidden_dim must be > 0")
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be > 0")

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.anchor_dim = anchor_dim
        self.feature_hidden_dim = feature_hidden_dim
        self.gate_temperature = float(gate_temperature)

        c, h, w = image_shape
        self.image_num_pixels = c * h * w

        self.gate_logits = nn.Parameter(
            torch.empty(self.image_num_pixels, latent_dim)
        )
        self.hidden_weight = nn.Parameter(
            torch.empty(feature_hidden_dim, latent_dim)
        )
        self.hidden_bias = nn.Parameter(torch.zeros(feature_hidden_dim))
        self.image_output_weight = nn.Parameter(
            torch.empty(self.image_num_pixels, feature_hidden_dim)
        )
        self.image_output_bias = nn.Parameter(torch.zeros(self.image_num_pixels))
        self.anchor_output_weight = nn.Parameter(
            torch.empty(anchor_dim, feature_hidden_dim)
        )
        self.anchor_output_bias = nn.Parameter(torch.zeros(anchor_dim))

        anchors_per_latent = anchor_dim // latent_dim
        anchor_latent_idx = torch.arange(anchor_dim) // anchors_per_latent
        anchor_gates = torch.zeros(anchor_dim, latent_dim)
        anchor_gates[torch.arange(anchor_dim), anchor_latent_idx] = 1.0
        self.register_buffer("anchor_gates", anchor_gates)

        self.activation = activation()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gate_logits, mean=-1.0, std=0.01)
        nn.init.xavier_uniform_(self.hidden_weight)
        nn.init.xavier_uniform_(self.image_output_weight)
        nn.init.xavier_uniform_(self.anchor_output_weight)

    def image_gates(self) -> Tensor:
        return torch.sigmoid(self.gate_logits / self.gate_temperature)

    def _decode_features(
        self,
        z: Tensor,
        gates: Tensor,
        output_weight: Tensor,
        output_bias: Tensor,
    ) -> Tensor:
        # Equivalent to applying each feature's gate to z before the shallow
        # decoder, without materializing [B, F, K] masked latents.
        effective_hidden = gates[:, None, :] * self.hidden_weight[None, :, :]
        hidden = torch.einsum("bk,fhk->bfh", z, effective_hidden)
        hidden = self.activation(hidden + self.hidden_bias[None, None, :])
        return torch.einsum("bfh,fh->bf", hidden, output_weight) + output_bias

    def forward(self, z: Tensor) -> Reconstruction:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must have shape [B, {self.latent_dim}], got {tuple(z.shape)}"
            )

        b = z.shape[0]
        img_flat = self._decode_features(
            z=z,
            gates=self.image_gates(),
            output_weight=self.image_output_weight,
            output_bias=self.image_output_bias,
        )
        anchor_mean = self._decode_features(
            z=z,
            gates=self.anchor_gates,
            output_weight=self.anchor_output_weight,
            output_bias=self.anchor_output_bias,
        )

        return Reconstruction(
            img_logits=img_flat.reshape(b, *self.image_shape),
            anchor_mean=anchor_mean,
        )


class SparseVAE(BaseVAE):
    """
    Regime B model: sparse decoder VAE with engineered anchor features.

    The full SparseVAE paper uses a Spike-and-Slab Lasso procedure.  Here we use
    differentiable sigmoid gates plus an L1-style gate penalty so the model fits
    the repo's shared VAE/loss abstractions and remains CPU-friendly.
    """

    def __init__(
        self,
        encoder: Encoder,
        latent_dim: int,
        image_shape: tuple[int, int, int],
        hidden_dim: int,
        anchor_dim: int,
        sparse_lambda: float = 1e-3,
        feature_hidden_dim: int = 16,
        gate_temperature: float = 1.0,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(encoder=encoder)

        if anchor_dim <= 0:
            raise ValueError("SparseVAE requires anchor_dim > 0")
        if anchor_dim % latent_dim != 0:
            raise ValueError(
                f"anchor_dim must be divisible by latent_dim, got anchor_dim={anchor_dim} "
                f"and latent_dim={latent_dim}"
            )
        if sparse_lambda < 0.0:
            raise ValueError("sparse_lambda must be >= 0")

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.hidden_dim = hidden_dim
        self.anchor_dim = anchor_dim
        self.sparse_lambda = float(sparse_lambda)

        self.decoder = SparseFeatureDecoder(
            latent_dim=latent_dim,
            image_shape=image_shape,
            anchor_dim=anchor_dim,
            feature_hidden_dim=feature_hidden_dim,
            gate_temperature=gate_temperature,
            activation=activation,
        )

    def decode(self, decoder_latent: Tensor, batch: Batch) -> Reconstruction:
        return self.decoder(decoder_latent)

    def compute_aux_losses(
        self,
        batch: Batch,
        out: ModelOutput,
    ) -> dict[str, Tensor]:
        del batch, out
        gate_penalty = self.decoder.image_gates().abs().mean()
        return {
            "sparsity": gate_penalty * self.sparse_lambda,
        }
