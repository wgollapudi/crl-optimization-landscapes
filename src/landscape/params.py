# landscape/params.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: torch.Size
    numel: int
    requires_grad: bool


def trainable_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]


def flatten_params(model: nn.Module) -> tuple[Tensor, list[ParamSpec]]:
    """
    Flatten all trainable parameters into one vector.

    Returns:
        theta: [P]
        specs: enough metadata to restore the vector into the model
    """
    pieces: list[Tensor] = []
    specs: list[ParamSpec] = []

    for name, p in trainable_named_params(model):
        specs.append(
            ParamSpec(
                name=name,
                shape=p.shape,
                numel=p.numel(),
                requires_grad=p.requires_grad,
            )
        )
        pieces.append(p.detach().reshape(-1))

    if not pieces:
        raise ValueError("model has no trainable parameters")

    return torch.cat(pieces), specs


def clone_param_vector(model: nn.Module) -> tuple[Tensor, list[ParamSpec]]:
    theta, specs = flatten_params(model)
    return theta.clone(), specs


@torch.no_grad()
def set_params_from_vector(
    model: nn.Module,
    theta: Tensor,
    specs: list[ParamSpec],
) -> None:
    """
    Restore trainable model parameters from a flat vector.

    Assumes model parameter order/name/shape matches `specs`.
    """
    named = trainable_named_params(model)

    if len(named) != len(specs):
        raise ValueError(
            f"Parameter count mismatch: model has {len(named)}, specs has {len(specs)}"
        )

    expected_numel = sum(spec.numel for spec in specs)
    if theta.numel() != expected_numel:
        raise ValueError(
            f"theta has {theta.numel()} elements, expected {expected_numel}"
        )

    offset = 0
    for (name, p), spec in zip(named, specs):
        if name != spec.name:
            raise ValueError(f"Parameter name mismatch: got {name}, expected {spec.name}")
        if p.shape != spec.shape:
            raise ValueError(
                f"Parameter shape mismatch for {name}: got {p.shape}, expected {spec.shape}"
            )

        chunk = theta[offset : offset + spec.numel].view(spec.shape)
        p.copy_(chunk.to(device=p.device, dtype=p.dtype))
        offset += spec.numel


def param_distance(theta_a: Tensor, theta_b: Tensor) -> float:
    if theta_a.shape != theta_b.shape:
        raise ValueError(f"shape mismatch: {theta_a.shape} vs {theta_b.shape}")
    return float(torch.linalg.vector_norm(theta_a - theta_b).item())


def global_normalized_direction(
    theta: Tensor,
    seed: int,
    scale: float = 1.0,
) -> Tensor:
    """
    Random direction with norm = scale * ||theta||.
    """
    gen = torch.Generator(device=theta.device)
    gen.manual_seed(seed)

    d = torch.randn(theta.shape, generator=gen, device=theta.device, dtype=theta.dtype)
    d_norm = torch.linalg.vector_norm(d)
    theta_norm = torch.linalg.vector_norm(theta)

    if d_norm == 0:
        raise RuntimeError("sampled zero random direction")

    return d / d_norm * (scale * theta_norm)


def layerwise_normalized_direction(
    model: nn.Module,
    specs: list[ParamSpec],
    seed: int,
    include_bias: bool = True,
) -> Tensor:
    """
    Random direction normalized parameter-block-wise.

    For each trainable parameter tensor p, sample d_p ~ N(0,I), then rescale
    d_p so ||d_p|| = ||p||. This is the simple layer-wise analogue of the
    normalization used in loss-landscape visualization.

    Biases can optionally be set to zero.
    """
    gen = torch.Generator(device=next(model.parameters()).device)
    gen.manual_seed(seed)

    chunks: list[Tensor] = []
    param_map = dict(trainable_named_params(model))

    for spec in specs:
        p = param_map[spec.name].detach()
        d = torch.randn(
            p.shape,
            generator=gen,
            device=p.device,
            dtype=p.dtype,
        )

        if (not include_bias) and _looks_like_bias(spec.name, p):
            chunks.append(torch.zeros_like(p).reshape(-1))
            continue

        p_norm = torch.linalg.vector_norm(p)
        d_norm = torch.linalg.vector_norm(d)

        if d_norm == 0 or p_norm == 0:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append((d / d_norm * p_norm).reshape(-1))

    return torch.cat(chunks)


def normalize_direction_layerwise(
    model: nn.Module,
    direction: Tensor,
    specs: list[ParamSpec],
    include_bias: bool = True,
) -> Tensor:
    """
    Rescale an existing flat direction block-wise so each parameter block has
    the same norm as the corresponding model parameter.

    Useful for interpolation/slice directions derived from another checkpoint.
    """
    param_map = dict(trainable_named_params(model))

    out_chunks: list[Tensor] = []
    offset = 0

    for spec in specs:
        d = direction[offset : offset + spec.numel].view(spec.shape)
        p = param_map[spec.name].detach()

        if (not include_bias) and _looks_like_bias(spec.name, p):
            out_chunks.append(torch.zeros_like(d).reshape(-1))
            offset += spec.numel
            continue

        p_norm = torch.linalg.vector_norm(p)
        d_norm = torch.linalg.vector_norm(d)

        if d_norm == 0 or p_norm == 0:
            out_chunks.append(torch.zeros_like(d).reshape(-1))
        else:
            out_chunks.append((d / d_norm * p_norm).reshape(-1))

        offset += spec.numel

    if offset != direction.numel():
        raise ValueError("direction has extra entries beyond specs")

    return torch.cat(out_chunks)


def orthogonalize_direction(d: Tensor, against: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Remove the component of d along `against`.
    """
    denom = torch.dot(against, against)
    if float(denom.item()) < eps:
        return d
    return d - torch.dot(d, against) / denom * against


def interpolate(theta_a: Tensor, theta_b: Tensor, alpha: float) -> Tensor:
    if theta_a.shape != theta_b.shape:
        raise ValueError(f"shape mismatch: {theta_a.shape} vs {theta_b.shape}")
    return (1.0 - alpha) * theta_a + alpha * theta_b


def _looks_like_bias(name: str, p: Tensor) -> bool:
    return name.endswith(".bias") or p.ndim == 1
