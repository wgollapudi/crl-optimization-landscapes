# models/plain_vae.py
from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import Tensor, nn

from data import Batch
from models.base_vae import BaseVAE, Reconstruction
from models.common import Encoder, MLP


@dataclass
class PlainVAEConfig:
    image_shape: tuple[int, int, int]
    latent_dim: int
    hidden_dim: int
    anchor_dim: int = 0
    encoder_hidden_layers: int = 2
    decoder_hidden_layers: int = 2
    fusion_hidden_layers: int = 1
    anchor_decoder_hidden_layers: int = 1

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
    model_name = "PlainVAE"

    def __init__(self, cfg: PlainVAEConfig) -> None:
        encoder = Encoder(
            image_shape=cfg.image_shape,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            anchor_dim=cfg.anchor_dim,
            image_num_hidden_layers=cfg.encoder_hidden_layers,
            fusion_num_hidden_layers=cfg.fusion_hidden_layers,
        )
        super().__init__(encoder=encoder)

        self.cfg = cfg

        self.image_decoder = ImageDecoder(
            latent_dim=cfg.latent_dim,
            image_shape=cfg.image_shape,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.decoder_hidden_layers,
        )

        self.anchor_decoder: AnchorDecoder | None
        if cfg.anchor_dim > 0:
            self.anchor_decoder = AnchorDecoder(
                latent_dim=cfg.latent_dim,
                anchor_dim=cfg.anchor_dim,
                hidden_dim=cfg.hidden_dim,
                num_hidden_layers=cfg.anchor_decoder_hidden_layers,
            )
        else:
            self.anchor_decoder = None

    @classmethod
    def from_model_config(cls, model_cfg) -> "PlainVAE":
        """
        Build from your project-level ModelConfig.
        """
        return cls(
            PlainVAEConfig(
                image_shape=tuple(model_cfg.image_shape),
                latent_dim=model_cfg.latent_dim,
                hidden_dim=model_cfg.hidden_dim,
                anchor_dim=model_cfg.anchor_dim,
            )
        )

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def decode(self, decoder_latent: Tensor, batch: Batch) -> Reconstruction:
        img_logits = self.image_decoder(decoder_latent)

        anchor_mean = None
        if self.anchor_decoder is not None:
            anchor_mean = self.anchor_decoder(decoder_latent)

        return Reconstruction(
            img_logits=img_logits,
            anchor_mean=anchor_mean,
        )
