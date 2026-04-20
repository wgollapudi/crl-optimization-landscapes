# losses.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from data import Batch
from models.base_vae import ModelOutput, kl_standard_normal


class Schedule(Protocol):
    def value(self, epoch: int, global_step: int) -> float: ...


@dataclass
class ConstantSchedule:
    constant: float

    def value(self, epoch: int, global_step: int) -> float:
        return self.constant


@dataclass
class LinearWarmupSchedule:
    start_value: float
    end_value: float
    warmup_epochs: int

    def value(self, epoch: int, global_step: int) -> float:
        if self.warmup_epochs <= 0:
            return self.end_value
        t = min(max(epoch, 0), self.warmup_epochs)
        alpha = t / self.warmup_epochs
        return (1.0 - alpha) * self.start_value + alpha * self.end_value


@dataclass
class StepInfo:
    epoch: int
    global_step: int


@dataclass
class LossBreakdown:
    total: Tensor
    recon_img: Tensor
    kl: Tensor
    beta: float
    recon_anchor: Tensor | None = None
    aux: dict[str, Tensor] = field(default_factory=dict)

    def scalar_metrics(self) -> dict[str, float]:
        metrics = {
            "loss": float(self.total.detach().item()),
            "recon_img": float(self.recon_img.detach().item()),
            "kl": float(self.kl.detach().item()),
            "beta": float(self.beta),
        }
        if self.recon_anchor is not None:
            metrics["recon_anchor"] = float(self.recon_anchor.detach().item())
        for key, value in self.aux.items():
            metrics[key] = float(value.detach().item())
        return metrics


@dataclass
class AnchorLossConfig:
    kind: str = "mse"  # "mse" or "gaussian_nll"
    weight: float = 1.0
    fixed_variance: float = 1.0


class VAELoss:
    """
    Standard VAE loss:
      total = recon_img + beta * kl + optional recon_anchor + optional aux losses

    Conventions:
      - image observations are binary dSprites images, so recon_img uses BCE-with-logits
      - kl is averaged across the batch
      - if anchor scalars are present and predicted, they are reconstructed with MSE by default
      - any model-specific extra losses are read from `out.extras`
    """

    def __init__(
        self,
        beta_schedule: Schedule,
        anchor_cfg: AnchorLossConfig | None = None,
    ) -> None:
        self.beta_schedule = beta_schedule
        self.anchor_cfg = anchor_cfg or AnchorLossConfig()

    def __call__(
        self,
        batch: Batch,
        out: ModelOutput,
        step_info: StepInfo,
    ) -> LossBreakdown:
        beta = self.beta_schedule.value(
            epoch=step_info.epoch,
            global_step=step_info.global_step,
        )

        recon_img = self._image_reconstruction_loss(
            logits=out.recon.img_logits,
            target=batch.x_img,
        )

        kl = kl_standard_normal(out.posterior.mu, out.posterior.logvar).mean()

        recon_anchor = self._anchor_reconstruction_loss(
            pred=out.recon.anchor_mean,
            target=batch.x_anchor,
        )

        aux = self._normalize_aux_losses(out.extras)

        total = recon_img + beta * kl
        if recon_anchor is not None:
            total = total + recon_anchor
        for value in aux.values():
            total = total + value

        return LossBreakdown(
            total=total,
            recon_img=recon_img,
            kl=kl,
            beta=beta,
            recon_anchor=recon_anchor,
            aux=aux,
        )

    def _image_reconstruction_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"Image logits/target shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}"
            )

        # Sum over pixels per example, then average across batch.
        per_example = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        ).flatten(start_dim=1).sum(dim=1)
        return per_example.mean()

    def _anchor_reconstruction_loss(
        self,
        pred: Tensor | None,
        target: Tensor | None,
    ) -> Tensor | None:
        if pred is None and target is None:
            return None

        if pred is None or target is None:
            raise ValueError(
                "Anchor reconstruction requires both predicted anchors and target anchors"
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"Anchor pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
            )

        kind = self.anchor_cfg.kind.lower()
        weight = self.anchor_cfg.weight

        if kind == "mse":
            per_example = F.mse_loss(pred, target, reduction="none").sum(dim=1)
            return weight * per_example.mean()

        if kind == "gaussian_nll":
            var = self.anchor_cfg.fixed_variance
            if var <= 0.0:
                raise ValueError("fixed_variance must be > 0 for gaussian_nll")
            # Constant terms retained for a proper NLL; still fine for optimization.
            per_dim = 0.5 * (
                ((target - pred) ** 2) / var + torch.log(torch.tensor(2.0 * torch.pi * var, device=pred.device))
            )
            per_example = per_dim.sum(dim=1)
            return weight * per_example.mean()

        raise ValueError(f"Unknown anchor loss kind: {self.anchor_cfg.kind}")

    def _normalize_aux_losses(
        self,
        extras: dict[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        if extras is None:
            return {}

        out: dict[str, Tensor] = {}
        for key, value in extras.items():
            if value.ndim != 0:
                raise ValueError(
                    f"Aux loss '{key}' must be a scalar tensor, got shape {tuple(value.shape)}"
                )
            out[key] = value
        return out
