# models/sparse_vae.py
from __future__ import annotations

import torch
from torch import Tensor, nn

from data import Batch
from models.base_vae import BaseVAE, ModelOutput, Reconstruction
from models.common import Encoder


class SparseFeatureDecoder(nn.Module):
    """
    Sparse per-feature image decoder.

    Each image pixel has a learned gate over latent dimensions. The pixel-specific
    gate is applied before a shallow shared hidden map, giving the sparse-decoder
    structure used by identifiable sparse DGMs, but with differentiable sigmoid
    gates instead of full spike-and-slab inference.
    """

    def __init__(
        self,
        latent_dim: int,
        image_shape: tuple[int, int, int],
        feature_hidden_dim: int = 16,
        gate_temperature: float = 1.0,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if len(image_shape) != 3:
            raise ValueError(f"image_shape must be (C, H, W), got {image_shape}")
        if any(dim <= 0 for dim in image_shape):
            raise ValueError(f"all image dimensions must be > 0, got {image_shape}")
        if feature_hidden_dim <= 0:
            raise ValueError("feature_hidden_dim must be > 0")
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be > 0")

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.feature_hidden_dim = feature_hidden_dim
        self.gate_temperature = float(gate_temperature)

        c, h, w = image_shape
        self.num_pixels = c * h * w

        # One gate vector per pixel: gates[p, k] controls whether latent k
        # contributes to pixel p.
        self.gate_logits = nn.Parameter(torch.empty(self.num_pixels, latent_dim))

        # Shared hidden map applied after feature-specific gating.
        self.hidden_weight = nn.Parameter(torch.empty(feature_hidden_dim, latent_dim))
        self.hidden_bias = nn.Parameter(torch.zeros(feature_hidden_dim))

        # Pixel-specific final readout from hidden activations.
        self.output_weight = nn.Parameter(torch.empty(self.num_pixels, feature_hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(self.num_pixels))

        self.activation = activation()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Start sparse: sigmoid(-3) ≈ 0.047. The model can open gates if useful.
        nn.init.normal_(self.gate_logits, mean=-3.0, std=0.01)
        nn.init.xavier_uniform_(self.hidden_weight)
        nn.init.xavier_uniform_(self.output_weight)

    def image_gates(self) -> Tensor:
        return torch.sigmoid(self.gate_logits / self.gate_temperature)

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, K], got {tuple(z.shape)}")
        if z.shape[1] != self.latent_dim:
            raise ValueError(f"expected latent dim {self.latent_dim}, got {z.shape[1]}")

        b = z.shape[0]
        gates = self.image_gates()  # [F, K]

        # For each pixel f, use hidden weights gated by gates[f].
        # This avoids materializing the full [B, F, K] masked latent tensor.
        effective_hidden = gates[:, None, :] * self.hidden_weight[None, :, :]  # [F, H, K]
        hidden = torch.einsum("bk,fhk->bfh", z, effective_hidden)             # [B, F, H]
        hidden = self.activation(hidden + self.hidden_bias[None, None, :])

        img_flat = torch.einsum("bfh,fh->bf", hidden, self.output_weight)
        img_flat = img_flat + self.output_bias[None, :]

        return img_flat.reshape(b, *self.image_shape)


class HardAnchorDecoder(nn.Module):
    """
    Decoder for engineered scalar anchors.

    Anchor order is:
        [z0_a0, z0_a1, z1_a0, z1_a1, ...]

    Each anchor group sees exactly one latent coordinate, giving two or more
    observed anchor features per latent factor.
    """

    def __init__(
        self,
        latent_dim: int,
        anchor_dim: int,
        hidden_dim: int = 16,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if anchor_dim <= 0:
            raise ValueError("anchor_dim must be > 0")
        if anchor_dim % latent_dim != 0:
            raise ValueError(
                f"anchor_dim must be divisible by latent_dim, got "
                f"anchor_dim={anchor_dim}, latent_dim={latent_dim}"
            )
        if anchor_dim // latent_dim < 2:
            raise ValueError("sparse identifiability requires at least two anchors per latent")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.latent_dim = latent_dim
        self.anchor_dim = anchor_dim
        self.anchors_per_latent = anchor_dim // latent_dim

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, self.anchors_per_latent),
                )
                for _ in range(latent_dim)
            ]
        )

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, K], got {tuple(z.shape)}")
        if z.shape[1] != self.latent_dim:
            raise ValueError(f"expected latent dim {self.latent_dim}, got {z.shape[1]}")

        return torch.cat(
            [decoder(z[:, k : k + 1]) for k, decoder in enumerate(self.decoders)],
            dim=1,
        )


class SparseVAE(BaseVAE):
    """
    Regime B model: sparse identifiable VAE with engineered anchors.

    This follows the sparse-DGM idea: observed features are generated from
    feature-specific subsets of the latent coordinates. We use continuous gates
    plus an L1-style penalty as a smooth approximation to the paper's exact
    spike-and-slab sparsity machinery.
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
        anchor_hidden_dim: int = 16,
        gate_temperature: float = 1.0,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(encoder=encoder)

        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if anchor_dim <= 0:
            raise ValueError("SparseVAE requires anchor_dim > 0")
        if anchor_dim % latent_dim != 0:
            raise ValueError(
                f"anchor_dim must be divisible by latent_dim, got "
                f"anchor_dim={anchor_dim}, latent_dim={latent_dim}"
            )
        if anchor_dim // latent_dim < 2:
            raise ValueError("SparseVAE requires at least two anchors per latent")
        if sparse_lambda < 0.0:
            raise ValueError("sparse_lambda must be >= 0")

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.hidden_dim = hidden_dim
        self.anchor_dim = anchor_dim
        self.sparse_lambda = float(sparse_lambda)

        self.image_decoder = SparseFeatureDecoder(
            latent_dim=latent_dim,
            image_shape=image_shape,
            feature_hidden_dim=feature_hidden_dim,
            gate_temperature=gate_temperature,
            activation=activation,
        )

        self.anchor_decoder = HardAnchorDecoder(
            latent_dim=latent_dim,
            anchor_dim=anchor_dim,
            hidden_dim=anchor_hidden_dim,
            activation=activation,
        )

    def decode(self, decoder_latent: Tensor, batch: Batch) -> Reconstruction:
        del batch

        img_logits = self.image_decoder(decoder_latent)
        anchor_mean = self.anchor_decoder(decoder_latent)

        return Reconstruction(
            img_logits=img_logits,
            anchor_mean=anchor_mean,
        )

    def compute_aux_losses(
        self,
        batch: Batch,
        out: ModelOutput,
    ) -> dict[str, Tensor]:
        del batch, out

        gates = self.image_decoder.image_gates()

        # Penalize open image gates. Anchor sparsity is hard-coded by construction,
        # so it does not need a learned sparsity penalty.
        sparsity = self.sparse_lambda * gates.mean()

        return {"sparsity": sparsity}

    def current_image_gates(self) -> Tensor:
        return self.image_decoder.image_gates().detach()
