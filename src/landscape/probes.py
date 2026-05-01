# landscape/probes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data import Batch
from landscape.eval import evaluate_loss_at_vector
from landscape.params import (
    ParamSpec,
    global_normalized_direction,
    interpolate,
    layerwise_normalized_direction,
    normalize_direction_layerwise,
    param_distance,
    set_params_from_vector,
)


@dataclass
class PerturbationResult:
    radii: np.ndarray
    mean_delta: np.ndarray
    median_delta: np.ndarray
    p90_delta: np.ndarray
    max_delta: np.ndarray
    base_loss: float


@dataclass
class Slice1DResult:
    alphas: np.ndarray
    losses: np.ndarray
    recon_img: np.ndarray
    kl: np.ndarray
    base_loss: float
    max_delta: float
    min_delta: float


@dataclass
class InterpolationResult:
    alphas: np.ndarray
    losses: np.ndarray
    recon_img: np.ndarray
    kl: np.ndarray
    endpoint_loss_a: float
    endpoint_loss_b: float
    barrier: float
    area_excess: float
    distance: float


def perturbation_sharpness(
    *,
    model: nn.Module,
    theta: Tensor,
    specs: list[ParamSpec],
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    radii: Sequence[float],
    directions_per_radius: int = 20,
    direction_seed: int = 0,
    max_batches: int | None = None,
    normalization: str = "layerwise",
) -> PerturbationResult:
    """
    Sample random perturbations around theta and measure deterministic loss increase.

    radii are relative multipliers applied to the normalized direction.
    With layerwise normalization, radius=1e-3 means:
        theta + 1e-3 * d
    where each parameter block of d has the same norm as the corresponding
    parameter block of theta.
    """
    if directions_per_radius <= 0:
        raise ValueError("directions_per_radius must be positive")

    base_metrics = evaluate_loss_at_vector(
        model=model,
        theta=theta,
        specs=specs,
        loss_fn=loss_fn,
        loader=loader,
        device=device,
        max_batches=max_batches,
    )
    base_loss = float(base_metrics["loss"])

    mean_delta: list[float] = []
    median_delta: list[float] = []
    p90_delta: list[float] = []
    max_delta: list[float] = []

    for r_idx, radius in enumerate(radii):
        if radius < 0:
            raise ValueError("radii must be nonnegative")

        deltas: list[float] = []

        for j in range(directions_per_radius):
            seed = direction_seed + 100_000 * r_idx + j

            d = _make_direction(
                model=model,
                theta=theta,
                specs=specs,
                seed=seed,
                normalization=normalization,
            )

            theta_perturbed = theta + float(radius) * d
            metrics = evaluate_loss_at_vector(
                model=model,
                theta=theta_perturbed,
                specs=specs,
                loss_fn=loss_fn,
                loader=loader,
                device=device,
                max_batches=max_batches,
            )
            deltas.append(float(metrics["loss"]) - base_loss)

        arr = np.asarray(deltas, dtype=np.float64)
        mean_delta.append(float(arr.mean()))
        median_delta.append(float(np.median(arr)))
        p90_delta.append(float(np.percentile(arr, 90)))
        max_delta.append(float(arr.max()))

    # Restore model to theta for caller sanity.
    set_params_from_vector(model, theta, specs)

    return PerturbationResult(
        radii=np.asarray(radii, dtype=np.float64),
        mean_delta=np.asarray(mean_delta, dtype=np.float64),
        median_delta=np.asarray(median_delta, dtype=np.float64),
        p90_delta=np.asarray(p90_delta, dtype=np.float64),
        max_delta=np.asarray(max_delta, dtype=np.float64),
        base_loss=base_loss,
    )


def slice_1d_random(
    *,
    model: nn.Module,
    theta: Tensor,
    specs: list[ParamSpec],
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    alphas: Sequence[float],
    direction_seed: int = 0,
    max_batches: int | None = None,
    normalization: str = "layerwise",
) -> Slice1DResult:
    d = _make_direction(
        model=model,
        theta=theta,
        specs=specs,
        seed=direction_seed,
        normalization=normalization,
    )

    return slice_1d_direction(
        model=model,
        theta=theta,
        direction=d,
        specs=specs,
        loss_fn=loss_fn,
        loader=loader,
        device=device,
        alphas=alphas,
        max_batches=max_batches,
    )


def slice_1d_direction(
    *,
    model: nn.Module,
    theta: Tensor,
    direction: Tensor,
    specs: list[ParamSpec],
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    alphas: Sequence[float],
    max_batches: int | None = None,
) -> Slice1DResult:
    """
    Evaluate loss along theta + alpha * direction.
    """
    losses: list[float] = []
    recons: list[float] = []
    kls: list[float] = []

    for alpha in alphas:
        theta_alpha = theta + float(alpha) * direction
        metrics = evaluate_loss_at_vector(
            model=model,
            theta=theta_alpha,
            specs=specs,
            loss_fn=loss_fn,
            loader=loader,
            device=device,
            max_batches=max_batches,
        )
        losses.append(float(metrics["loss"]))
        recons.append(float(metrics.get("recon_img", np.nan)))
        kls.append(float(metrics.get("kl", np.nan)))

    # Restore model to theta for caller sanity.
    set_params_from_vector(model, theta, specs)

    losses_arr = np.asarray(losses, dtype=np.float64)
    base_idx = _closest_zero_index(alphas)
    base_loss = float(losses_arr[base_idx])

    return Slice1DResult(
        alphas=np.asarray(alphas, dtype=np.float64),
        losses=losses_arr,
        recon_img=np.asarray(recons, dtype=np.float64),
        kl=np.asarray(kls, dtype=np.float64),
        base_loss=base_loss,
        max_delta=float(losses_arr.max() - base_loss),
        min_delta=float(losses_arr.min() - base_loss),
    )


def slice_1d_between(
    *,
    model: nn.Module,
    theta_a: Tensor,
    theta_b: Tensor,
    specs: list[ParamSpec],
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    alphas: Sequence[float],
    max_batches: int | None = None,
    normalize_direction: bool = True,
) -> Slice1DResult:
    """
    Slice from theta_a in the direction of theta_b - theta_a.

    This is not interpolation unless alphas go from 0 to 1 and normalize_direction=False.
    """
    d = theta_b - theta_a
    if normalize_direction:
        set_params_from_vector(model, theta_a, specs)
        d = normalize_direction_layerwise(model, d, specs)

    return slice_1d_direction(
        model=model,
        theta=theta_a,
        direction=d,
        specs=specs,
        loss_fn=loss_fn,
        loader=loader,
        device=device,
        alphas=alphas,
        max_batches=max_batches,
    )


def interpolation_curve(
    *,
    model: nn.Module,
    theta_a: Tensor,
    theta_b: Tensor,
    specs: list[ParamSpec],
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    alphas: Sequence[float],
    max_batches: int | None = None,
) -> InterpolationResult:
    """
    Evaluate deterministic loss along the straight line:

        theta(alpha) = (1 - alpha) theta_a + alpha theta_b

    Usually alphas should include 0 and 1.
    """
    if theta_a.shape != theta_b.shape:
        raise ValueError(f"theta shape mismatch: {theta_a.shape} vs {theta_b.shape}")

    losses: list[float] = []
    recons: list[float] = []
    kls: list[float] = []

    for alpha in alphas:
        theta_alpha = interpolate(theta_a, theta_b, float(alpha))
        metrics = evaluate_loss_at_vector(
            model=model,
            theta=theta_alpha,
            specs=specs,
            loss_fn=loss_fn,
            loader=loader,
            device=device,
            max_batches=max_batches,
        )
        losses.append(float(metrics["loss"]))
        recons.append(float(metrics.get("recon_img", np.nan)))
        kls.append(float(metrics.get("kl", np.nan)))

    # Restore model to theta_a for caller sanity.
    set_params_from_vector(model, theta_a, specs)

    alphas_arr = np.asarray(alphas, dtype=np.float64)
    losses_arr = np.asarray(losses, dtype=np.float64)

    endpoint_loss_a = _loss_at_alpha(alphas_arr, losses_arr, target_alpha=0.0)
    endpoint_loss_b = _loss_at_alpha(alphas_arr, losses_arr, target_alpha=1.0)

    baseline = 0.5 * (endpoint_loss_a + endpoint_loss_b)
    excess = np.maximum(losses_arr - baseline, 0.0)

    barrier = float(losses_arr.max() - baseline)
    area_excess = float(np.trapz(excess, alphas_arr) / (alphas_arr.max() - alphas_arr.min()))

    return InterpolationResult(
        alphas=alphas_arr,
        losses=losses_arr,
        recon_img=np.asarray(recons, dtype=np.float64),
        kl=np.asarray(kls, dtype=np.float64),
        endpoint_loss_a=float(endpoint_loss_a),
        endpoint_loss_b=float(endpoint_loss_b),
        barrier=barrier,
        area_excess=area_excess,
        distance=param_distance(theta_a, theta_b),
    )


def perturbation_result_to_dict(result: PerturbationResult) -> dict[str, np.ndarray | float]:
    return {
        "radii": result.radii,
        "mean_delta": result.mean_delta,
        "median_delta": result.median_delta,
        "p90_delta": result.p90_delta,
        "max_delta": result.max_delta,
        "base_loss": result.base_loss,
    }


def slice_1d_result_to_dict(result: Slice1DResult) -> dict[str, np.ndarray | float]:
    return {
        "alphas": result.alphas,
        "losses": result.losses,
        "recon_img": result.recon_img,
        "kl": result.kl,
        "base_loss": result.base_loss,
        "max_delta": result.max_delta,
        "min_delta": result.min_delta,
    }


def interpolation_result_to_dict(result: InterpolationResult) -> dict[str, np.ndarray | float]:
    return {
        "alphas": result.alphas,
        "losses": result.losses,
        "recon_img": result.recon_img,
        "kl": result.kl,
        "endpoint_loss_a": result.endpoint_loss_a,
        "endpoint_loss_b": result.endpoint_loss_b,
        "barrier": result.barrier,
        "area_excess": result.area_excess,
        "distance": result.distance,
    }


def _make_direction(
    *,
    model: nn.Module,
    theta: Tensor,
    specs: list[ParamSpec],
    seed: int,
    normalization: str,
) -> Tensor:
    if normalization == "layerwise":
        set_params_from_vector(model, theta, specs)
        return layerwise_normalized_direction(model, specs, seed=seed)

    if normalization == "global":
        return global_normalized_direction(theta, seed=seed)

    raise ValueError(
        f"Unknown normalization={normalization!r}; expected 'layerwise' or 'global'"
    )


def _closest_zero_index(xs: Sequence[float]) -> int:
    arr = np.asarray(xs, dtype=np.float64)
    return int(np.argmin(np.abs(arr)))


def _loss_at_alpha(
    alphas: np.ndarray,
    losses: np.ndarray,
    target_alpha: float,
) -> float:
    idx = int(np.argmin(np.abs(alphas - target_alpha)))
    if abs(float(alphas[idx]) - target_alpha) > 1e-8:
        raise ValueError(
            f"alphas must include {target_alpha}; closest value is {alphas[idx]}"
        )
    return float(losses[idx])
