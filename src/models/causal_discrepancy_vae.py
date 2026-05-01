from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from data import Batch
from models.base_vae import BaseVAE, ModelOutput, Reconstruction
from models.common import Encoder
from models.plain_vae import ImageDecoder
from models.sparse_vae import HardAnchorDecoder

@dataclass
class CausalDiscrepancyVAEConfig:
    image_shape: tuple[int, int, int]
    latent_dim: int
    hidden_dim: int
    anchor_dim: int = 0
    num_intervention_variants: int = 2
    num_intervention_envs: int = 0
    mmd_weight: float = 1.0
    graph_l1_weight: float = 1e-3
    decoder_num_hidden_layers: int = 2
    anchor_hidden_dim: int = 16
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

    The triangular order is a practical DAG parameterization. Node labels are
    arbitrary: intervention labels are mapped into this learned order by the
    InterventionEncoder, and true intervention targets are diagnostics-only.
    """

    def __init__(
        self,
        latent_dim: int,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")

        self.latent_dim = latent_dim

        self.raw_adjacency = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        lower_mask = torch.tril(torch.ones(latent_dim, latent_dim), diagonal=-1)
        self.register_buffer("lower_mask", lower_mask)

    def adjacency(self) -> Tensor:
        return self.raw_adjacency * self.lower_mask

    def forward(
        self,
        z: Tensor,
        target_probs: Tensor | None = None,
        intervention_means: Tensor | None = None,
        intervention_log_scales: Tensor | None = None,
    ) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must have shape [B, {self.latent_dim}], got {tuple(z.shape)}"
            )

        b = z.shape[0]
        if target_probs is None:
            target_probs = z.new_zeros((b, self.latent_dim))
        if intervention_means is None:
            intervention_means = z.new_zeros((b, self.latent_dim))
        if intervention_log_scales is None:
            intervention_log_scales = z.new_zeros((b, self.latent_dim))

        expected = (b, self.latent_dim)
        for name, value in [
            ("target_probs", target_probs),
            ("intervention_means", intervention_means),
            ("intervention_log_scales", intervention_log_scales),
        ]:
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

        adjacency = self.adjacency()
        u_cols: list[Tensor] = []

        for j in range(self.latent_dim):
            if j == 0:
                value = z[:, j]
            else:
                parents = torch.stack(u_cols, dim=1)
                parent_effect = parents @ adjacency[j, :j]
                value = z[:, j] + parent_effect

            p_target = target_probs[:, j]
            intervention_value = (
                intervention_means[:, j]
                + torch.exp(intervention_log_scales[:, j]) * z[:, j]
            )
            value = (1.0 - p_target) * value + p_target * intervention_value

            u_cols.append(value)

        return torch.stack(u_cols, dim=1)


class InterventionEncoder(nn.Module):
    """
    Learn intervention target and shift/scale from observed environment labels.

    env_id=0 is reserved for the observational environment and returns no
    intervention. True intervention targets from the dataset are not consumed by
    this module; they are only diagnostics metadata.
    """

    def __init__(self, num_envs: int, latent_dim: int) -> None:
        super().__init__()

        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")

        self.num_envs = num_envs
        self.latent_dim = latent_dim
        self.target_logits = nn.Embedding(num_envs, latent_dim)
        self.means = nn.Embedding(num_envs, latent_dim)
        self.log_scales = nn.Embedding(num_envs, latent_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.target_logits.weight)
        nn.init.zeros_(self.means.weight)
        nn.init.zeros_(self.log_scales.weight)

    def forward(self, env_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        env = env_id.to(dtype=torch.long)
        if env.ndim != 1:
            raise ValueError(f"env_id must have shape [B], got {tuple(env.shape)}")
        if ((env < 0) | (env >= self.num_envs)).any().item():
            raise ValueError(
                f"env_id values must be in [0, {self.num_envs}); got "
                f"min={int(env.min().item())}, max={int(env.max().item())}"
            )

        logits = self.target_logits(env)
        target_probs = torch.softmax(logits, dim=1)
        means = self.means(env)
        log_scales = self.log_scales(env)

        observational = env == 0
        if observational.any().item():
            target_probs = target_probs.clone()
            means = means.clone()
            log_scales = log_scales.clone()
            target_probs[observational] = 0.0
            means[observational] = 0.0
            log_scales[observational] = 0.0

        return target_probs, means, log_scales

    @torch.no_grad()
    def target_probabilities(self) -> Tensor:
        probs = torch.softmax(self.target_logits.weight, dim=1)
        if self.num_envs > 0:
            probs = probs.clone()
            probs[0] = 0.0
        return probs


class CausalDiscrepancyVAE(BaseVAE):
    """
    Regimes C/D model: discrepancy-based causal VAE.

    The model uses env_id labels to learn an intervention-to-target map and
    never consumes true intervention_target during training. The true targets in
    the dataset are diagnostics metadata for post-hoc permutation/equivalence
    checks.
    """

    model_name = "CausalDiscrepancyVAE"

    def __init__(
        self,
        encoder: Encoder,
        latent_dim: int,
        image_shape: tuple[int, int, int],
        hidden_dim: int,
        anchor_dim: int = 0,
        num_intervention_variants: int = 2,
        num_intervention_envs: int = 0,
        mmd_weight: float = 1.0,
        graph_l1_weight: float = 1e-3,
        decoder_num_hidden_layers: int = 2,
        anchor_hidden_dim: int = 16,
        mmd_bandwidths: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        activation: type[nn.Module] = nn.ReLU,
        cfg: CausalDiscrepancyVAEConfig | None = None,
    ) -> None:
        super().__init__(encoder=encoder)

        if mmd_weight < 0.0:
            raise ValueError("mmd_weight must be >= 0")
        if graph_l1_weight < 0.0:
            raise ValueError("graph_l1_weight must be >= 0")
        if anchor_dim < 0:
            raise ValueError("anchor_dim must be >= 0")
        if num_intervention_envs <= 0:
            raise ValueError("num_intervention_envs must be > 0")

        self.cfg = cfg or CausalDiscrepancyVAEConfig(
            image_shape=tuple(image_shape),
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            anchor_dim=anchor_dim,
            num_intervention_variants=num_intervention_variants,
            num_intervention_envs=num_intervention_envs,
            mmd_weight=mmd_weight,
            graph_l1_weight=graph_l1_weight,
            decoder_num_hidden_layers=decoder_num_hidden_layers,
            anchor_hidden_dim=anchor_hidden_dim,
            mmd_bandwidths=tuple(float(v) for v in mmd_bandwidths),
        )

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.hidden_dim = hidden_dim
        self.anchor_dim = anchor_dim
        self.num_intervention_envs = num_intervention_envs
        self.mmd_weight = float(mmd_weight)
        self.graph_l1_weight = float(graph_l1_weight)
        self.mmd_bandwidths = tuple(float(v) for v in mmd_bandwidths)

        self.scm = LinearCausalSCMLayer(latent_dim=latent_dim)
        self.intervention_encoder = InterventionEncoder(
            num_envs=num_intervention_envs,
            latent_dim=latent_dim,
        )
        self.image_decoder = ImageDecoder(
            latent_dim=latent_dim,
            image_shape=image_shape,
            hidden_dim=hidden_dim,
            num_hidden_layers=decoder_num_hidden_layers,
            activation=activation,
        )
        self.anchor_decoder: HardAnchorDecoder | None
        if anchor_dim > 0:
            self.anchor_decoder = HardAnchorDecoder(
                latent_dim=latent_dim,
                anchor_dim=anchor_dim,
                hidden_dim=anchor_hidden_dim,
                activation=activation,
            )
        else:
            self.anchor_decoder = None

    @classmethod
    def from_model_config(cls, model_cfg) -> "CausalDiscrepancyVAE":
        cfg = CausalDiscrepancyVAEConfig(
            image_shape=tuple(model_cfg.image_shape),
            latent_dim=model_cfg.latent_dim,
            hidden_dim=model_cfg.hidden_dim,
            anchor_dim=model_cfg.anchor_dim,
            num_intervention_variants=model_cfg.num_intervention_variants,
            num_intervention_envs=model_cfg.num_intervention_envs,
            mmd_weight=model_cfg.mmd_weight,
            graph_l1_weight=model_cfg.graph_l1_weight,
        )

        encoder = Encoder(
            image_shape=cfg.image_shape,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            anchor_dim=cfg.anchor_dim,
        )

        return cls(
            encoder=encoder,
            latent_dim=cfg.latent_dim,
            image_shape=cfg.image_shape,
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

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def latent_to_decoder_input(self, z: Tensor, batch: Batch) -> Tensor:
        target_probs, means, log_scales = self.intervention_encoder(batch.env_id)
        return self.scm(
            z,
            target_probs=target_probs,
            intervention_means=means,
            intervention_log_scales=log_scales,
        )

    def decode(self, decoder_latent: Tensor, batch: Batch) -> Reconstruction:
        del batch
        anchor_mean = None
        if self.anchor_decoder is not None:
            anchor_mean = self.anchor_decoder(decoder_latent)
        return Reconstruction(
            img_logits=self.image_decoder(decoder_latent),
            anchor_mean=anchor_mean,
        )

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
            mmd = self._virtual_intervention_mmd(batch=batch, out=out)

        return {
            "graph_l1": graph_l1,
            "mmd": self.mmd_weight * mmd,
        }

    def _virtual_intervention_mmd(
        self,
        batch: Batch,
        out: ModelOutput,
    ) -> Tensor:
        obs_mask = batch.env_id == 0
        if int(obs_mask.sum().item()) < 2:
            return out.z.new_zeros(())

        obs_z = out.z[obs_mask]
        mmd_values: list[Tensor] = []

        for env in torch.unique(batch.env_id):
            env_int = int(env.item())
            if env_int == 0:
                continue

            real_mask = batch.env_id == env
            if int(real_mask.sum().item()) < 2:
                continue

            virtual_env = torch.full(
                (obs_z.shape[0],),
                env_int,
                device=batch.env_id.device,
                dtype=torch.long,
            )
            target_probs, means, log_scales = self.intervention_encoder(virtual_env)
            virtual_u = self.scm(
                obs_z,
                target_probs=target_probs,
                intervention_means=means,
                intervention_log_scales=log_scales,
            )
            virtual_logits = self.image_decoder(virtual_u)
            virtual_flat = torch.sigmoid(virtual_logits).flatten(start_dim=1)
            real_flat = batch.x_img[real_mask].flatten(start_dim=1)
            mmd_values.append(
                rbf_mmd(
                    virtual_flat,
                    real_flat,
                    bandwidths=self.mmd_bandwidths,
                )
            )

        if not mmd_values:
            return out.z.new_zeros(())
        return torch.stack(mmd_values).mean()

    @torch.no_grad()
    def learned_intervention_targets(self) -> Tensor:
        return self.intervention_encoder.target_probabilities()
