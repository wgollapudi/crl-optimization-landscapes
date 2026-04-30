# landscape/eval.py
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Iterator

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data import Batch
from engine import move_batch_to_device
from losses import LossBreakdown, StepInfo
from models.base_vae import ModelOutput
from landscape.params import ParamSpec, set_params_from_vector, trainable_named_params

@contextmanager
def temporary_params(
    model: nn.Module,
    theta: Tensor,
    specs: list[ParamSpec],
) -> Iterator[None]:
    """
    Temporarily set model parameters to theta, then restore original params.
    """
    original = torch.cat([
        p.detach().reshape(-1).clone()
        for _, p in trainable_named_params(model)
    ])

    device = next(model.parameters()).device
    set_params_from_vector(model, theta, specs)
    try:
        yield
    finally:
        set_params_from_vector(model, original.to(theta.device), specs)


def forward_deterministic(model: nn.Module, batch: Batch) -> ModelOutput:
    """
    Landscape evaluation should use z = mu_phi(x), not stochastic sampling.

    Requires models to support model(batch, sample=False).
    """
    try:
        return model(batch, sample=False)
    except TypeError as exc:
        raise TypeError(
            "model must support forward(batch, sample=False) for deterministic "
            "landscape evaluation"
        ) from exc


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    *,
    max_batches: int | None = None,
    epoch: int = 0,
    global_step: int = 0,
) -> dict[str, float]:
    """
    Deterministically evaluate loss and components on a loader.

    Uses model(batch, sample=False).
    """
    model.eval()
    device = torch.device(device)

    sums: dict[str, float] = defaultdict(float)
    count = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        out = forward_deterministic(model, batch)
        loss: LossBreakdown = loss_fn(
            batch=batch,
            out=out,
            step_info=StepInfo(epoch=epoch, global_step=global_step),
        )

        bsz = int(batch.x_img.shape[0])
        metrics = loss.scalar_metrics()

        for key, value in metrics.items():
            sums[key] += float(value) * bsz
        count += bsz

    if count == 0:
        raise ValueError("evaluate_loss saw zero examples")

    return {key: value / count for key, value in sums.items()}


@torch.no_grad()
def evaluate_loss_at_vector(
    model: nn.Module,
    theta: Tensor,
    specs: list[ParamSpec],
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    *,
    max_batches: int | None = None,
    epoch: int = 0,
    global_step: int = 0,
) -> dict[str, float]:
    """
    Temporarily set model params to theta, evaluate deterministic loss,
    then restore original params.
    """
    with temporary_params(model, theta, specs):
        return evaluate_loss(
            model=model,
            loss_fn=loss_fn,
            loader=loader,
            device=device,
            max_batches=max_batches,
            epoch=epoch,
            global_step=global_step,
        )


def evaluate_loss_for_grad(
    model: nn.Module,
    loss_fn: Callable,
    batch: Batch,
    device: str | torch.device,
    *,
    epoch: int = 0,
    global_step: int = 0,
    component: str = "total",
) -> Tensor:
    """
    Differentiable deterministic loss on one batch.

    Used later for gradients / Hessian-vector products.

    component:
      - "total"
      - "recon_img"
      - "kl"
      - "recon_anchor"
      - any key in loss.aux
    """
    model.train(False)
    batch = move_batch_to_device(batch, torch.device(device))

    out = forward_deterministic(model, batch)
    loss: LossBreakdown = loss_fn(
        batch=batch,
        out=out,
        step_info=StepInfo(epoch=epoch, global_step=global_step),
    )

    if component == "total":
        return loss.total
    if component == "recon_img":
        return loss.recon_img
    if component == "kl":
        return loss.kl
    if component == "recon_anchor":
        if loss.recon_anchor is None:
            raise ValueError("requested recon_anchor, but it is None")
        return loss.recon_anchor
    if component in loss.aux:
        return loss.aux[component]

    raise KeyError(
        f"Unknown loss component {component!r}; available aux keys: {sorted(loss.aux)}"
    )


def endpoint_metrics(
    model: nn.Module,
    loss_fn: Callable,
    train_loader: DataLoader[Batch],
    val_loader: DataLoader[Batch],
    device: str | torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Convenience helper for one checkpoint.
    """
    train = evaluate_loss(
        model=model,
        loss_fn=loss_fn,
        loader=train_loader,
        device=device,
        max_batches=max_batches,
    )
    val = evaluate_loss(
        model=model,
        loss_fn=loss_fn,
        loader=val_loader,
        device=device,
        max_batches=max_batches,
    )

    out: dict[str, float] = {}
    for k, v in train.items():
        out[f"train_{k}"] = v
    for k, v in val.items():
        out[f"val_{k}"] = v
    return out
