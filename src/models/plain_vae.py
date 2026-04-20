# models/plain_vae.py
from __future__ import annotations

from torch import Tensor, nn

from data import Batch
from models.base_vae import BaseVAE, Reconstruction
from models.common import Encoder, MLP


class ImageDecoder(nn.Module):
    """
    Image decoder for binary dSprites images.

    Input:
        z: [B, latent_dim]
    Output:
        logits: [B, C, H, W]

    For now this is an MLP over flattened pixels, which is simple and consistent
    with our choice to keep Regime A small and easy to analyze.
    """

    def __init__(
        self,
        latent_dim: int,
        image_shape: tuple[int, int, int],
        hidden_dim: int,
        num_hidden_layers: int = 2,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        c, h, w = image_shape
        self.output_dim = c * h * w

        self.backbone = MLP(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=self.output_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
        )

    def forward(self, z: Tensor) -> Tensor:
        b = z.shape[0]
        logits = self.backbone(z)
        return logits.reshape(b, *self.image_shape)


class AnchorDecoder(nn.Module):
    """
    Optional scalar reconstruction head.

    Used later for regimes with appended anchor features.
    For Regime A this will usually be disabled via anchor_dim=0.
    """

    def __init__(
        self,
        latent_dim: int,
        anchor_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.anchor_dim = anchor_dim
        self.backbone = MLP(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=anchor_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.backbone(z)


class PlainVAE(BaseVAE):
    """
    Regime A model: plain VAE.

    Shared pieces:
      - Encoder from models.common
      - MLP image decoder
      - optional anchor decoder head

    No sparsity, no causal layer, no discrepancy terms.
    """

    def __init__(
        self,
        encoder: Encoder,
        latent_dim: int,
        image_shape: tuple[int, int, int],
        hidden_dim: int,
        anchor_dim: int = 0,
        decoder_num_hidden_layers: int = 2,
        anchor_decoder_num_hidden_layers: int = 1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(encoder=encoder)

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.hidden_dim = hidden_dim
        self.anchor_dim = anchor_dim

        self.image_decoder = ImageDecoder(
            latent_dim=latent_dim,
            image_shape=image_shape,
            hidden_dim=hidden_dim,
            num_hidden_layers=decoder_num_hidden_layers,
            activation=activation,
        )

        self.anchor_decoder: AnchorDecoder | None
        if anchor_dim > 0:
            self.anchor_decoder = AnchorDecoder(
                latent_dim=latent_dim,
                anchor_dim=anchor_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=anchor_decoder_num_hidden_layers,
                activation=activation,
            )
        else:
            self.anchor_decoder = None

    def decode(self, decoder_latent: Tensor, batch: Batch) -> Reconstruction:
        img_logits = self.image_decoder(decoder_latent)

        anchor_mean = None
        if self.anchor_decoder is not None:
            anchor_mean = self.anchor_decoder(decoder_latent)

        return Reconstruction(
            img_logits=img_logits,
            anchor_mean=anchor_mean,
        )
