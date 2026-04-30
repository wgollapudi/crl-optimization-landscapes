from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from data import Batch
from models.base_vae import BaseVAE, ModelOutput, Reconstruction
from models.common import Encoder
from models.plain_vae import ImageDecoder

@dataclass
class CausalDiscrepancyVAEConfig:
    image_shape: tuple[int, int, int]
    latent_dim: int
    hidden_dim: int
    num_intervention_variants: int = 2
    mmd_weight: float = 1.0
    graph_l1_weight: float = 1e-3
    decoder_num_hidden_layers: int = 2
    mmd_bandwidths: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

def _normalized_sq_dists(x: Tensor, y: Tensor) -> Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("MMD inputs must be rank-2 tensors")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"MMD feature dimensions differ: {x.shape[1]} vs {y.shape[1]}"
        )

    x_norm = x.pow(2).sum(dim=1, keepdim=True)
    y_norm = y.pow(2).sum(dim=1, keepdim=True).transpose(0, 1)
    dists = x_norm + y_norm - 2.0 * x @ y.transpose(0, 1)
    return dists.clamp_min(0.0) / max(int(x.shape[1]), 1)


def rbf_mmd(
    x: Tensor,
    y: Tensor,
    bandwidths: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> Tensor:
    """
    Biased multi-kernel RBF MMD over flattened samples.

    Distances are normalized by feature dimension, which keeps the kernel scale
    usable for 64x64 binary images in small CPU smoke tests.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("MMD requires non-empty sample sets")

    d_xx = _normalized_sq_dists(x, x)
    d_yy = _normalized_sq_dists(y, y)
    d_xy = _normalized_sq_dists(x, y)

    k_xx = x.new_zeros(d_xx.shape)
    k_yy = x.new_zeros(d_yy.shape)
    k_xy = x.new_zeros(d_xy.shape)

    for bandwidth in bandwidths:
        if bandwidth <= 0.0:
            raise ValueError("MMD bandwidths must be > 0")
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        k_xx = k_xx + torch.exp(-gamma * d_xx)
        k_yy = k_yy + torch.exp(-gamma * d_yy)
        k_xy = k_xy + torch.exp(-gamma * d_xy)

    scale = 1.0 / float(len(bandwidths))
    mmd = scale * (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return mmd.clamp_min(0.0)


class LinearCausalSCMLayer(nn.Module):
    """
    Lower-triangular linear SCM from exogenous z to causal latents u.

    The topological order is fixed to latent dimensions 0..K-1.  Interventions
    replace the targeted structural equation before descendants are computed.
    """

    def __init__(
        self,
        latent_dim: int,
        num_intervention_variants: int = 2,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if num_intervention_variants <= 0:
            raise ValueError("num_intervention_variants must be > 0")

        self.latent_dim = latent_dim
        self.num_intervention_variants = num_intervention_variants

        self.raw_adjacency = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        lower_mask = torch.tril(torch.ones(latent_dim, latent_dim), diagonal=-1)
        self.register_buffer("lower_mask", lower_mask)

        self.intervention_means = nn.Parameter(
            torch.zeros(latent_dim, num_intervention_variants)
        )
        self.intervention_log_scales = nn.Parameter(
            torch.zeros(latent_dim, num_intervention_variants)
        )

    def adjacency(self) -> Tensor:
        return self.raw_adjacency * self.lower_mask

    def _prepare_labels(
        self,
        z: Tensor,
        intervention_target: Tensor | None,
        intervention_variant: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        b = z.shape[0]
        device = z.device

        if intervention_target is None:
            target = torch.full((b,), -1, device=device, dtype=torch.long)
        else:
            target = intervention_target.to(device=device, dtype=torch.long)

        if intervention_variant is None:
            variant = torch.zeros(b, device=device, dtype=torch.long)
        else:
            variant = intervention_variant.to(device=device, dtype=torch.long)

        if target.shape != (b,):
            raise ValueError(
                f"intervention_target must have shape [{b}], got {tuple(target.shape)}"
            )
        if variant.shape != (b,):
            raise ValueError(
                f"intervention_variant must have shape [{b}], got {tuple(variant.shape)}"
            )

        invalid_target = (target < -1) | (target >= self.latent_dim)
        if invalid_target.any().item():
            raise ValueError(
                "intervention_target values must be -1 or in [0, latent_dim)"
            )

        intervened = target >= 0
        invalid_variant = (
            intervened
            & ((variant < 0) | (variant >= self.num_intervention_variants))
        )
        if invalid_variant.any().item():
            raise ValueError(
                "intervention_variant values for intervened samples must be in "
                "[0, num_intervention_variants)"
            )

        return target, variant

    def forward(
        self,
        z: Tensor,
        intervention_target: Tensor | None = None,
        intervention_variant: Tensor | None = None,
    ) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must have shape [B, {self.latent_dim}], got {tuple(z.shape)}"
            )

        target, variant = self._prepare_labels(
            z=z,
            intervention_target=intervention_target,
            intervention_variant=intervention_variant,
        )

        adjacency = self.adjacency()
        u = z.new_zeros(z.shape)

        for j in range(self.latent_dim):
            if j == 0:
                value = z[:, j]
            else:
                parent_effect = u[:, :j] @ adjacency[j, :j]
                value = z[:, j] + parent_effect

            intervened_j = target == j
            if intervened_j.any().item():
                value = value.clone()
                variants_j = variant[intervened_j]
                means = self.intervention_means[j, variants_j]
                scales = torch.exp(self.intervention_log_scales[j, variants_j])
                value[intervened_j] = means + scales * z[intervened_j, j]

            u[:, j] = value

        return u


class CausalDiscrepancyVAE(BaseVAE):
    """
    Regimes C/D model: simplified discrepancy-based causal VAE.

    Zhang et al.'s full method uses a deep SCM and intervention encoder.  This
    project-compatible version uses known intervention labels, a linear
    lower-triangular SCM, and an MMD discrepancy over generated vs. real images.
    """

    model_name = "CausalDiscrepancyVAE"

    def __init__(
        self,
        encoder: Encoder,
        latent_dim: int,
        image_shape: tuple[int, int, int],
        hidden_dim: int,
        num_intervention_variants: int = 2,
        mmd_weight: float = 1.0,
        graph_l1_weight: float = 1e-3,
        decoder_num_hidden_layers: int = 2,
        mmd_bandwidths: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        activation: type[nn.Module] = nn.ReLU,
        cfg: CausalDiscrepancyVAEConfig | None = None,
    ) -> None:
        super().__init__(encoder=encoder)

        if mmd_weight < 0.0:
            raise ValueError("mmd_weight must be >= 0")
        if graph_l1_weight < 0.0:
            raise ValueError("graph_l1_weight must be >= 0")

        self.cfg = cfg or CausalDiscrepancyVAEConfig(
            image_shape=tuple(image_shape),
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_intervention_variants=num_intervention_variants,
            mmd_weight=mmd_weight,
            graph_l1_weight=graph_l1_weight,
            decoder_num_hidden_layers=decoder_num_hidden_layers,
            mmd_bandwidths=tuple(float(v) for v in mmd_bandwidths),
        )

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.hidden_dim = hidden_dim
        self.mmd_weight = float(mmd_weight)
        self.graph_l1_weight = float(graph_l1_weight)
        self.mmd_bandwidths = tuple(float(v) for v in mmd_bandwidths)

        self.scm = LinearCausalSCMLayer(
            latent_dim=latent_dim,
            num_intervention_variants=num_intervention_variants,
        )
        self.image_decoder = ImageDecoder(
            latent_dim=latent_dim,
            image_shape=image_shape,
            hidden_dim=hidden_dim,
            num_hidden_layers=decoder_num_hidden_layers,
            activation=activation,
        )

    @classmethod
    def from_model_config(cls, model_cfg) -> "CausalDiscrepancyVAE":
        cfg = CausalDiscrepancyVAEConfig(
            image_shape=tuple(model_cfg.image_shape),
            latent_dim=model_cfg.latent_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_intervention_variants=model_cfg.num_intervention_variants,
            mmd_weight=model_cfg.mmd_weight,
            graph_l1_weight=model_cfg.graph_l1_weight,
        )

        encoder = Encoder(
            image_shape=cfg.image_shape,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            anchor_dim=0,
        )

        return cls(
            encoder=encoder,
            latent_dim=cfg.latent_dim,
            image_shape=cfg.image_shape,
            hidden_dim=cfg.hidden_dim,
            num_intervention_variants=cfg.num_intervention_variants,
            mmd_weight=cfg.mmd_weight,
            graph_l1_weight=cfg.graph_l1_weight,
            decoder_num_hidden_layers=cfg.decoder_num_hidden_layers,
            mmd_bandwidths=cfg.mmd_bandwidths,
            cfg=cfg,
        )

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def latent_to_decoder_input(self, z: Tensor, batch: Batch) -> Tensor:
        return self.scm(
            z,
            intervention_target=batch.intervention_target,
            intervention_variant=batch.intervention_variant,
        )

    def decode(self, decoder_latent: Tensor, batch: Batch) -> Reconstruction:
        del batch
        return Reconstruction(img_logits=self.image_decoder(decoder_latent))

    def _grouped_mmd(self, generated_flat: Tensor, real_flat: Tensor, env_id: Tensor) -> Tensor:
        mmd_values: list[Tensor] = []
        for env in torch.unique(env_id):
            mask = env_id == env
            if int(mask.sum().item()) >= 2:
                mmd_values.append(
                    rbf_mmd(
                        generated_flat[mask],
                        real_flat[mask],
                        bandwidths=self.mmd_bandwidths,
                    )
                )

        if mmd_values:
            return torch.stack(mmd_values).mean()
        if generated_flat.shape[0] >= 1:
            return rbf_mmd(
                generated_flat,
                real_flat,
                bandwidths=self.mmd_bandwidths,
            )
        return generated_flat.new_zeros(())

    def compute_aux_losses(
        self,
        batch: Batch,
        out: ModelOutput,
    ) -> dict[str, Tensor]:
        adjacency = self.scm.adjacency()
        graph_l1 = self.graph_l1_weight * adjacency.abs().mean()

        if self.mmd_weight == 0.0:
            mmd = out.z.new_zeros(())
        else:
            z_prior = torch.randn_like(out.z)
            u_prior = self.scm(
                z_prior,
                intervention_target=batch.intervention_target,
                intervention_variant=batch.intervention_variant,
            )
            generated_logits = self.image_decoder(u_prior)
            generated_flat = torch.sigmoid(generated_logits).flatten(start_dim=1)
            real_flat = batch.x_img.flatten(start_dim=1)
            mmd = self._grouped_mmd(generated_flat, real_flat, batch.env_id)

        return {
            "graph_l1": graph_l1,
            "mmd": self.mmd_weight * mmd,
        }
