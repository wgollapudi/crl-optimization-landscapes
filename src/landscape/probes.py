# landscape/probes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data import Batch
from engine import move_batch_to_device
from landscape.eval import evaluate_loss_at_vector, evaluate_loss_for_grad
from landscape.metric_keys import is_loss_key
from landscape.params import (
    ParamSpec,
    global_normalized_direction,
    interpolate,
    layerwise_normalized_direction,
    normalize_direction_layerwise,
    param_distance,
    set_params_from_vector,
    trainable_named_params,
)


@dataclass
class PerturbationResult:
    radii: np.ndarray
    mean_delta: np.ndarray
    median_delta: np.ndarray
    p90_delta: np.ndarray
    max_delta: np.ndarray
    base_loss: float
    auc_mean_delta: float


@dataclass
class CurveResult:
    alphas: np.ndarray
    metric_names: np.ndarray
    metric_values: np.ndarray
    descriptors: dict[str, float]


@dataclass
class InterpolationResult:
    alphas: np.ndarray
    metric_names: np.ndarray
    metric_values: np.ndarray
    descriptors: dict[str, float]
    distance: float


@dataclass
class GradientResult:
    component_names: np.ndarray
    grad_norm: np.ndarray
    grad_norm_sq: np.ndarray
    num_params: int


@dataclass
class HessianResult:
    component_names: np.ndarray
    status: np.ndarray
    top_eigenvalue: np.ndarray
    top_eigenvalues_raw: np.ndarray
    trace: np.ndarray
    trace_std: np.ndarray
    trace_stderr: np.ndarray
    trace_samples_raw: np.ndarray
    num_params: int
    power_iters: int
    power_restarts: int
    trace_samples: int
    max_batches: int


@dataclass
class Slice2DResult:
    alphas: np.ndarray
    betas: np.ndarray
    metric_names: np.ndarray
    metric_values: np.ndarray
    descriptors: dict[str, float]


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

    set_params_from_vector(model, theta, specs)
    radii_arr = np.asarray(radii, dtype=np.float64)
    mean_arr = np.asarray(mean_delta, dtype=np.float64)
    denom = float(radii_arr.max() - radii_arr.min()) if len(radii_arr) > 1 else 1.0
    auc_mean_delta = float(np.trapezoid(mean_arr, radii_arr) / denom) if denom > 0 else float(mean_arr.mean())

    return PerturbationResult(
        radii=radii_arr,
        mean_delta=mean_arr,
        median_delta=np.asarray(median_delta, dtype=np.float64),
        p90_delta=np.asarray(p90_delta, dtype=np.float64),
        max_delta=np.asarray(max_delta, dtype=np.float64),
        base_loss=base_loss,
        auc_mean_delta=auc_mean_delta,
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
) -> CurveResult:
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
) -> CurveResult:
    metrics_by_point: list[dict[str, float]] = []
    for alpha in alphas:
        theta_alpha = theta + float(alpha) * direction
        metrics_by_point.append(
            evaluate_loss_at_vector(
                model=model,
                theta=theta_alpha,
                specs=specs,
                loss_fn=loss_fn,
                loader=loader,
                device=device,
                max_batches=max_batches,
            )
        )

    set_params_from_vector(model, theta, specs)
    alphas_arr = np.asarray(alphas, dtype=np.float64)
    metric_names, metric_values = _metrics_sequence_to_arrays(metrics_by_point)
    base_idx = _closest_zero_index(alphas_arr)
    descriptors = curve_descriptors(
        x=alphas_arr,
        metric_names=metric_names,
        metric_values=metric_values,
        base_index=base_idx,
        prefix="",
    )
    return CurveResult(
        alphas=alphas_arr,
        metric_names=metric_names,
        metric_values=metric_values,
        descriptors=descriptors,
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
) -> CurveResult:
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
    if theta_a.shape != theta_b.shape:
        raise ValueError(f"theta shape mismatch: {theta_a.shape} vs {theta_b.shape}")

    metrics_by_point: list[dict[str, float]] = []
    for alpha in alphas:
        theta_alpha = interpolate(theta_a, theta_b, float(alpha))
        metrics_by_point.append(
            evaluate_loss_at_vector(
                model=model,
                theta=theta_alpha,
                specs=specs,
                loss_fn=loss_fn,
                loader=loader,
                device=device,
                max_batches=max_batches,
            )
        )

    set_params_from_vector(model, theta_a, specs)
    alphas_arr = np.asarray(alphas, dtype=np.float64)
    metric_names, metric_values = _metrics_sequence_to_arrays(metrics_by_point)
    idx_a = _alpha_index(alphas_arr, 0.0)
    idx_b = _alpha_index(alphas_arr, 1.0)
    descriptors = interpolation_descriptors(
        x=alphas_arr,
        metric_names=metric_names,
        metric_values=metric_values,
        endpoint_a_index=idx_a,
        endpoint_b_index=idx_b,
    )
    return InterpolationResult(
        alphas=alphas_arr,
        metric_names=metric_names,
        metric_values=metric_values,
        descriptors=descriptors,
        distance=param_distance(theta_a, theta_b),
    )


def gradient_norm(
    *,
    model: nn.Module,
    theta: Tensor,
    specs: list[ParamSpec],
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    components: Sequence[str],
    max_batches: int | None = None,
) -> GradientResult:
    set_params_from_vector(model, theta, specs)
    params = [p for _, p in trainable_named_params(model)]
    norms: list[float] = []
    norms_sq: list[float] = []

    for component in components:
        model.zero_grad(set_to_none=True)
        total_loss: Tensor | None = None
        count = 0
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            loss = evaluate_loss_for_grad(
                model=model,
                loss_fn=loss_fn,
                batch=batch,
                device=device,
                component=component,
            )
            total_loss = loss if total_loss is None else total_loss + loss
            count += 1
        if count == 0 or total_loss is None:
            raise ValueError("gradient_norm saw zero batches")
        avg_loss = total_loss / count
        grads = torch.autograd.grad(avg_loss, params, allow_unused=True)
        sq = sum(float(g.detach().pow(2).sum().item()) for g in grads if g is not None)
        norms_sq.append(sq)
        norms.append(float(np.sqrt(sq)))

    set_params_from_vector(model, theta, specs)
    return GradientResult(
        component_names=np.asarray(list(components), dtype=str),
        grad_norm=np.asarray(norms, dtype=np.float64),
        grad_norm_sq=np.asarray(norms_sq, dtype=np.float64),
        num_params=int(theta.numel()),
    )


def hessian_summary(
    *,
    model: nn.Module,
    theta: Tensor,
    specs: list[ParamSpec],
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    components: Sequence[str],
    power_iters: int = 20,
    power_restarts: int = 1,
    trace_samples: int = 20,
    max_batches: int = 3,
    seed: int = 0,
) -> HessianResult:
    if power_iters <= 0:
        raise ValueError("power_iters must be positive")
    if power_restarts <= 0:
        raise ValueError("power_restarts must be positive")
    if trace_samples <= 0:
        raise ValueError("trace_samples must be positive")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")

    set_params_from_vector(model, theta, specs)
    params = [p for _, p in trainable_named_params(model)]
    gen = torch.Generator(device=theta.device)
    gen.manual_seed(seed)

    statuses: list[str] = []
    tops: list[float] = []
    top_raw: list[list[float]] = []
    traces: list[float] = []
    trace_stds: list[float] = []
    trace_stderrs: list[float] = []
    trace_raw: list[list[float]] = []
    for c_idx, component in enumerate(components):
        def hvp(v_flat: Tensor) -> Tensor:
            set_params_from_vector(model, theta, specs)
            model.zero_grad(set_to_none=True)
            loss = _component_loss_over_loader(
                model=model,
                loss_fn=loss_fn,
                loader=loader,
                device=device,
                component=component,
                max_batches=max_batches,
            )
            if not torch.isfinite(loss).all().item():
                raise FloatingPointError("nonfinite_loss")
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            grad_flat = _flatten_grads(grads, params)
            if not torch.isfinite(grad_flat).all().item():
                raise FloatingPointError("nonfinite_grad")
            dot = _dot_grads_with_vector(grads, v_flat, specs)
            hv = torch.autograd.grad(dot, params, allow_unused=True)
            hv_flat = _flatten_grads(hv, params).detach()
            if not torch.isfinite(hv_flat).all().item():
                raise FloatingPointError("nonfinite_hvp")
            return hv_flat

        status = "ok"
        eig_vals: list[float] = []
        trace_vals: list[float] = []
        try:
            for _ in range(power_restarts):
                v = torch.randn(theta.shape, generator=gen, device=theta.device, dtype=theta.dtype)
                v_norm = torch.linalg.vector_norm(v)
                if float(v_norm.item()) == 0.0:
                    raise FloatingPointError("zero_power_vector")
                v = v / v_norm
                eig = 0.0
                for _ in range(power_iters):
                    hv = hvp(v)
                    hv_norm = torch.linalg.vector_norm(hv)
                    if not torch.isfinite(hv_norm).all().item():
                        raise FloatingPointError("nonfinite_hvp_norm")
                    if float(hv_norm.item()) == 0.0:
                        raise FloatingPointError("zero_hvp")
                    v = hv / hv_norm
                    eig = float(torch.dot(v, hv).item())
                    if not np.isfinite(eig):
                        raise FloatingPointError("nonfinite_top_eigenvalue")
                eig_vals.append(eig)

            for _ in range(trace_samples):
                r = torch.randint(0, 2, theta.shape, generator=gen, device=theta.device, dtype=torch.int64)
                r = (r.to(dtype=theta.dtype) * 2.0) - 1.0
                hv = hvp(r)
                trace_val = float(torch.dot(r, hv).item())
                if not np.isfinite(trace_val):
                    raise FloatingPointError("nonfinite_trace")
                trace_vals.append(trace_val)
        except FloatingPointError as exc:
            status = str(exc)

        statuses.append(status)
        if eig_vals:
            top_raw.append(eig_vals)
            tops.append(float(np.max(eig_vals)))
        else:
            top_raw.append([np.nan] * power_restarts)
            tops.append(np.nan)

        if trace_vals:
            trace_arr = np.asarray(trace_vals, dtype=np.float64)
            traces.append(float(trace_arr.mean()))
            trace_stds.append(float(trace_arr.std(ddof=1)) if len(trace_arr) > 1 else 0.0)
            trace_stderrs.append(float(trace_arr.std(ddof=1) / np.sqrt(len(trace_arr))) if len(trace_arr) > 1 else 0.0)
            trace_raw.append(trace_vals)
        else:
            traces.append(np.nan)
            trace_stds.append(np.nan)
            trace_stderrs.append(np.nan)
            trace_raw.append([np.nan] * trace_samples)

        gen.manual_seed(seed + 10_000 * (c_idx + 1))

    set_params_from_vector(model, theta, specs)
    return HessianResult(
        component_names=np.asarray(list(components), dtype=str),
        status=np.asarray(statuses, dtype=str),
        top_eigenvalue=np.asarray(tops, dtype=np.float64),
        top_eigenvalues_raw=np.asarray(top_raw, dtype=np.float64),
        trace=np.asarray(traces, dtype=np.float64),
        trace_std=np.asarray(trace_stds, dtype=np.float64),
        trace_stderr=np.asarray(trace_stderrs, dtype=np.float64),
        trace_samples_raw=np.asarray(trace_raw, dtype=np.float64),
        num_params=int(theta.numel()),
        power_iters=int(power_iters),
        power_restarts=int(power_restarts),
        trace_samples=int(trace_samples),
        max_batches=int(max_batches),
    )


def slice_2d_random(
    *,
    model: nn.Module,
    theta: Tensor,
    specs: list[ParamSpec],
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    alphas: Sequence[float],
    betas: Sequence[float],
    direction_seed: int = 0,
    max_batches: int | None = None,
    normalization: str = "layerwise",
) -> Slice2DResult:
    d1 = _make_direction(
        model=model,
        theta=theta,
        specs=specs,
        seed=direction_seed,
        normalization=normalization,
    )
    d2 = _make_direction(
        model=model,
        theta=theta,
        specs=specs,
        seed=direction_seed + 1,
        normalization=normalization,
    )
    d2 = torch.as_tensor(d2 - torch.dot(d2, d1) / torch.dot(d1, d1).clamp_min(1e-12) * d1)
    set_params_from_vector(model, theta, specs)
    d2 = normalize_direction_layerwise(model, d2, specs)

    alphas_arr = np.asarray(alphas, dtype=np.float64)
    betas_arr = np.asarray(betas, dtype=np.float64)
    flat_metrics: list[dict[str, float]] = []
    for alpha in alphas_arr:
        for beta in betas_arr:
            theta_grid = theta + float(alpha) * d1 + float(beta) * d2
            flat_metrics.append(
                evaluate_loss_at_vector(
                    model=model,
                    theta=theta_grid,
                    specs=specs,
                    loss_fn=loss_fn,
                    loader=loader,
                    device=device,
                    max_batches=max_batches,
                )
            )

    set_params_from_vector(model, theta, specs)
    metric_names, flat_values = _metrics_sequence_to_arrays(flat_metrics)
    metric_values = flat_values.reshape(len(alphas_arr), len(betas_arr), len(metric_names))
    descriptors = slice2d_descriptors(
        alphas=alphas_arr,
        betas=betas_arr,
        metric_names=metric_names,
        metric_values=metric_values,
    )
    return Slice2DResult(
        alphas=alphas_arr,
        betas=betas_arr,
        metric_names=metric_names,
        metric_values=metric_values,
        descriptors=descriptors,
    )


def curve_descriptors(
    *,
    x: np.ndarray,
    metric_names: np.ndarray,
    metric_values: np.ndarray,
    base_index: int,
    prefix: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for m_idx, name in enumerate(metric_names.tolist()):
        y = metric_values[:, m_idx]
        base = float(y[base_index])
        safe = _safe_metric_name(name)
        key = f"{prefix}{safe}"
        out[f"base_{key}"] = base
        out[f"max_delta_{key}"] = float(y.max() - base)
        out[f"min_delta_{key}"] = float(y.min() - base)
        out[f"argmax_alpha_{key}"] = float(x[int(np.argmax(y))])
        out[f"argmin_alpha_{key}"] = float(x[int(np.argmin(y))])
        if 0 < base_index < len(x) - 1:
            dx = float(x[base_index + 1] - x[base_index])
            out[f"center_second_diff_{key}"] = float((y[base_index + 1] - 2.0 * y[base_index] + y[base_index - 1]) / (dx * dx)) if dx != 0 else np.nan
        else:
            out[f"center_second_diff_{key}"] = np.nan
        for eps in (1e-3, 1e-2):
            threshold = base + abs(base) * eps
            mask = y <= threshold
            out[f"near_base_width_rel{eps:g}_{key}"] = _mask_width(x, mask)
    return out


def interpolation_descriptors(
    *,
    x: np.ndarray,
    metric_names: np.ndarray,
    metric_values: np.ndarray,
    endpoint_a_index: int,
    endpoint_b_index: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for m_idx, name in enumerate(metric_names.tolist()):
        y = metric_values[:, m_idx]
        a = float(y[endpoint_a_index])
        b = float(y[endpoint_b_index])
        baseline = 0.5 * (a + b)
        excess = np.maximum(y - baseline, 0.0)
        safe = _safe_metric_name(name)
        out[f"endpoint_i_{safe}"] = a
        out[f"endpoint_j_{safe}"] = b
        out[f"barrier_{safe}"] = float(y.max() - baseline)
        denom = float(x.max() - x.min())
        out[f"area_excess_{safe}"] = float(np.trapezoid(excess, x) / denom) if denom > 0 else float(excess.mean())
        out[f"max_delta_{safe}"] = float(y.max() - min(a, b))
        out[f"num_peaks_{safe}"] = float(_count_peaks(y, threshold=baseline))
    return out


def slice2d_descriptors(
    *,
    alphas: np.ndarray,
    betas: np.ndarray,
    metric_names: np.ndarray,
    metric_values: np.ndarray,
) -> dict[str, float]:
    del alphas, betas
    out: dict[str, float] = {}
    center_a = metric_values.shape[0] // 2
    center_b = metric_values.shape[1] // 2
    total_area = float(metric_values.shape[0] * metric_values.shape[1])
    for m_idx, name in enumerate(metric_names.tolist()):
        z = metric_values[:, :, m_idx]
        base = float(z[center_a, center_b])
        safe = _safe_metric_name(name)
        out[f"base_{safe}"] = base
        out[f"max_delta_{safe}"] = float(z.max() - base)
        out[f"min_delta_{safe}"] = float(z.min() - base)
        for eps in (1e-3, 1e-2):
            threshold = base + abs(base) * eps
            out[f"sublevel_area_frac_rel{eps:g}_{safe}"] = float((z <= threshold).sum() / total_area)
    return out


def perturbation_result_to_dict(result: PerturbationResult) -> dict[str, np.ndarray | float]:
    return {
        "radii": result.radii,
        "mean_delta": result.mean_delta,
        "median_delta": result.median_delta,
        "p90_delta": result.p90_delta,
        "max_delta": result.max_delta,
        "base_loss": result.base_loss,
        "auc_mean_delta": result.auc_mean_delta,
    }


def curve_result_to_dict(result: CurveResult) -> dict[str, np.ndarray | float]:
    return {
        "alphas": result.alphas,
        "metric_names": result.metric_names,
        "metric_values": result.metric_values,
        **result.descriptors,
    }


def interpolation_result_to_dict(result: InterpolationResult) -> dict[str, np.ndarray | float]:
    return {
        "alphas": result.alphas,
        "metric_names": result.metric_names,
        "metric_values": result.metric_values,
        "distance": result.distance,
        **result.descriptors,
    }


def gradient_result_to_dict(result: GradientResult) -> dict[str, np.ndarray | float]:
    return {
        "component_names": result.component_names,
        "grad_norm": result.grad_norm,
        "grad_norm_sq": result.grad_norm_sq,
        "num_params": result.num_params,
    }


def hessian_result_to_dict(result: HessianResult) -> dict[str, np.ndarray | float]:
    return {
        "component_names": result.component_names,
        "status": result.status,
        "top_eigenvalue": result.top_eigenvalue,
        "top_eigenvalues_raw": result.top_eigenvalues_raw,
        "trace": result.trace,
        "trace_std": result.trace_std,
        "trace_stderr": result.trace_stderr,
        "trace_samples_raw": result.trace_samples_raw,
        "num_params": result.num_params,
        "power_iters": result.power_iters,
        "power_restarts": result.power_restarts,
        "trace_samples": result.trace_samples,
        "max_batches": result.max_batches,
    }


def slice2d_result_to_dict(result: Slice2DResult) -> dict[str, np.ndarray | float]:
    return {
        "alphas": result.alphas,
        "betas": result.betas,
        "metric_names": result.metric_names,
        "metric_values": result.metric_values,
        **result.descriptors,
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
    raise ValueError(f"Unknown normalization={normalization!r}; expected 'layerwise' or 'global'")


def _metrics_sequence_to_arrays(metrics_by_point: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not metrics_by_point:
        raise ValueError("metrics_by_point must be nonempty")
    keys = [k for k in metrics_by_point[0] if is_loss_key(k)]
    if "loss" in keys:
        keys = ["loss"] + sorted(k for k in keys if k != "loss")
    else:
        keys = sorted(keys)
    values = np.asarray([[float(m.get(k, np.nan)) for k in keys] for m in metrics_by_point], dtype=np.float64)
    return np.asarray(keys, dtype=str), values


def _component_loss_over_loader(
    *,
    model: nn.Module,
    loss_fn: Callable,
    loader: DataLoader[Batch],
    device: str | torch.device,
    component: str,
    max_batches: int,
) -> Tensor:
    total: Tensor | None = None
    count = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        loss = evaluate_loss_for_grad(
            model=model,
            loss_fn=loss_fn,
            batch=batch,
            device=device,
            component=component,
        )
        total = loss if total is None else total + loss
        count += 1
    if total is None or count == 0:
        raise ValueError("component loss saw zero batches")
    return total / count


def _dot_grads_with_vector(grads: Sequence[Tensor | None], vector: Tensor, specs: list[ParamSpec]) -> Tensor:
    offset = 0
    dot: Tensor | None = None
    for grad, spec in zip(grads, specs):
        chunk = vector[offset : offset + spec.numel].view(spec.shape)
        offset += spec.numel
        if grad is None:
            continue
        term = (grad * chunk.to(device=grad.device, dtype=grad.dtype)).sum()
        dot = term if dot is None else dot + term
    if dot is None:
        return vector.new_zeros(())
    return dot


def _flatten_grads(grads: Sequence[Tensor | None], params: Sequence[nn.Parameter]) -> Tensor:
    pieces: list[Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(grad.reshape(-1))
    return torch.cat(pieces)


def _closest_zero_index(xs: Sequence[float]) -> int:
    arr = np.asarray(xs, dtype=np.float64)
    return int(np.argmin(np.abs(arr)))


def _alpha_index(alphas: np.ndarray, target_alpha: float) -> int:
    idx = int(np.argmin(np.abs(alphas - target_alpha)))
    if abs(float(alphas[idx]) - target_alpha) > 1e-8:
        raise ValueError(f"alphas must include {target_alpha}; closest value is {alphas[idx]}")
    return idx


def _safe_metric_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_").replace(".", "_")


def _mask_width(x: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0.0
    return float(x[idx[-1]] - x[idx[0]])


def _count_peaks(y: np.ndarray, threshold: float) -> int:
    if len(y) < 3:
        return int(np.any(y > threshold))
    count = 0
    for i in range(1, len(y) - 1):
        if y[i] > threshold and y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            count += 1
    return count
