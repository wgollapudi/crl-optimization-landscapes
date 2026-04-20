# models/common.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PosteriorParams:
    mu: Tensor
    logvar: Tensor


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_layers: int = 2,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """
    Image-only encoder backbone.

    Input:
        x_img: [B, C, H, W]
    Output:
        h_img: [B, hidden_dim]

    For now, this is an MLP over flattened images. That keeps Regime A simple
    and makes later loss-landscape analysis easier.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        hidden_dim: int,
        num_hidden_layers: int = 2,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.image_shape = image_shape
        c, h, w = image_shape
        self.input_dim = c * h * w
        self.hidden_dim = hidden_dim

        self.backbone = MLP(
            in_dim=self.input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
        )

    def forward(self, x_img: Tensor) -> Tensor:
        if x_img.ndim != 4:
            raise ValueError(
                f"x_img must have shape [B, C, H, W], got {tuple(x_img.shape)}"
            )

        b = x_img.shape[0]
        x_flat = x_img.reshape(b, -1)
        return self.backbone(x_flat)


class Encoder(nn.Module):
    """
    Full observation encoder.

    Uses:
      - ImageEncoder for the image branch
      - optional anchor-feature branch
      - fusion MLP
      - posterior heads for mu and logvar

    Inputs:
        x_img:    [B, C, H, W]
        x_anchor: [B, A] or None

    Outputs:
        PosteriorParams(mu, logvar), each [B, latent_dim]
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        latent_dim: int,
        hidden_dim: int,
        anchor_dim: int = 0,
        image_num_hidden_layers: int = 2,
        fusion_num_hidden_layers: int = 1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.anchor_dim = anchor_dim

        self.image_encoder = ImageEncoder(
            image_shape=image_shape,
            hidden_dim=hidden_dim,
            num_hidden_layers=image_num_hidden_layers,
            activation=activation,
        )

        if anchor_dim > 0:
            self.anchor_encoder = MLP(
                in_dim=anchor_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                num_hidden_layers=1,
                activation=activation,
            )
            fusion_in_dim = 2 * hidden_dim
        else:
            self.anchor_encoder = None
            fusion_in_dim = hidden_dim

        self.fusion = MLP(
            in_dim=fusion_in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_hidden_layers=fusion_num_hidden_layers,
            activation=activation,
        )

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x_img: Tensor,
        x_anchor: Tensor | None = None,
    ) -> PosteriorParams:
        h_img = self.image_encoder(x_img)

        if self.anchor_dim > 0:
            if x_anchor is None:
                raise ValueError(
                    "Encoder was configured with anchor_dim > 0, but x_anchor is None"
                )
            if self.anchor_encoder is None:
                raise RuntimeError("anchor_encoder is unexpectedly None")

            h_anchor = self.anchor_encoder(x_anchor)
            h = torch.cat([h_img, h_anchor], dim=1)
        else:
            if x_anchor is not None:
                # Ignore extra anchor input if this encoder is image-only.
                # This keeps the harness flexible.
                pass
            h = h_img

        h = self.fusion(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        return PosteriorParams(mu=mu, logvar=logvar)

if __name__ == "__main__":
    pass
