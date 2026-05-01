# engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any
import random
import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import Batch
from losses import LossBreakdown, StepInfo
from experiment_logging import ExperimentLogger


@dataclass
class EpochMetrics:
    loss: float
    metrics: dict[str, float]


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    def move(x: Tensor | None) -> Tensor | None:
        if x is None:
            return None
        return x.to(device, non_blocking=True)

    return Batch(
        x_img=move(batch.x_img),
        x_anchor=move(batch.x_anchor),
        env_id=move(batch.env_id),
        intervention_target=move(batch.intervention_target),
        intervention_variant=move(batch.intervention_variant),
        z_true=move(batch.z_true),
        class_tuples=move(batch.class_tuples),
        latents_quantized=move(batch.latents_quantized),
        orientation_cos_sin=move(batch.orientation_cos_sin),
    )

def posterior_stats(mu: Tensor, logvar: Tensor) -> dict[str, float]:
    stats: dict[str, float] = {}

    mu_mean_dim = mu.mean(dim=0)  # [latent_dim]
    mu_std_dim = mu.std(dim=0)    # [latent_dim]
    logvar_mean_dim = logvar.mean(dim=0)

    for i in range(mu.shape[1]):
        stats[f"mu_mean_{i}"] = float(mu_mean_dim[i].detach().item())
        stats[f"mu_std_{i}"] = float(mu_std_dim[i].detach().item())
        stats[f"logvar_mean_{i}"] = float(logvar_mean_dim[i].detach().item())

    stats["mu_abs_mean"] = float(mu.abs().mean().detach().item())
    stats["logvar_mean"] = float(logvar.mean().detach().item())
    stats["logvar_std"] = float(logvar.std().detach().item())

    return stats

class MetricTracker:
    def __init__(self) -> None:
        self._weighted_sums: dict[str, float] = {}
        self._count: int = 0

    def update(self, metrics: Mapping[str, float], batch_size: int) -> None:
        for key, value in metrics.items():
            self._weighted_sums[key] = self._weighted_sums.get(key, 0.0) + value * batch_size
        self._count += batch_size

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {}
        return {
            key: value / self._count
            for key, value in self._weighted_sums.items()
        }


class CheckpointManager:
    def __init__(
        self,
        outdir: str | Path,
        monitor: str = "loss",
        mode: str = "min",
        save_every: int = 0,
    ) -> None:
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.monitor = monitor
        self.mode = mode
        self.save_every = save_every
        self.best_value: float | None = None
        self.best_path: Path | None = None
        self.best_epoch: int | None = None
        self.best_global_step: int | None = None

    def is_best(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def save(
        self,
        name: str,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        global_step: int,
        extra_state: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.outdir / name
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "extra_state": dict(extra_state or {}),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "model_name": getattr(model, "model_name", model.__class__.__name__),
            "model_config": model.config_dict() if hasattr(model, "config_dict") else None,
        }
        torch.save(payload, path)
        return path

    def maybe_save_best(
        self,
        value: float,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        global_step: int,
        extra_state: Mapping[str, Any] | None = None,
    ) -> Path | None:
        if not self.is_best(value):
            return None

        self.best_value = value
        self.best_epoch = epoch
        self.best_global_step = global_step
        self.best_path = self.save(
            name=f"best_val_{self.monitor}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            extra_state={
                **dict(extra_state or {}),
                "best_value": value,
                "monitor": self.monitor,
                "mode": self.mode,
            },
        )
        return self.best_path

    def save_mid_best_from_available_checkpoints(self) -> Path | None:
        """
        Save mid_best.pt from the available checkpoint closest to half the best step.

        The true "halfway to best validation" model is only known retrospectively.
        We avoid saving every epoch by selecting from the bounded checkpoints that
        already exist: start.pt, periodic epoch_*.pt files, and best_val_*.pt.
        Use a smaller --save-every for a more precise midpoint.
        """
        if self.best_global_step is None:
            return None

        target_step = 0.5 * float(self.best_global_step)
        candidates = self._available_midpoint_candidates()
        if not candidates:
            return None

        before_or_at_best = [
            item for item in candidates
            if item[1] <= self.best_global_step
        ]
        if before_or_at_best:
            candidates = before_or_at_best

        source_path, source_step, source_epoch = min(
            candidates,
            key=lambda item: (abs(float(item[1]) - target_step), item[1]),
        )

        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        extra = dict(payload.get("extra_state", {}))
        extra.update(
            {
                "checkpoint_kind": "mid_best",
                "source_checkpoint": str(source_path),
                "source_epoch": int(source_epoch),
                "source_global_step": int(source_step),
                "best_epoch": int(self.best_epoch) if self.best_epoch is not None else -1,
                "best_global_step": int(self.best_global_step),
                "target_global_step": target_step,
                "selection_rule": "available checkpoint closest to 0.5 * best_global_step",
            }
        )
        payload["extra_state"] = extra

        out = self.outdir / "mid_best.pt"
        torch.save(payload, out)
        return out

    def _available_midpoint_candidates(self) -> list[tuple[Path, int, int]]:
        paths = [self.outdir / "start.pt"]
        paths.extend(sorted(self.outdir.glob("epoch_*.pt")))
        if self.best_path is not None:
            paths.append(self.best_path)

        out: list[tuple[Path, int, int]] = []
        seen: set[Path] = set()
        for path in paths:
            if path in seen or not path.exists():
                continue
            seen.add(path)
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            out.append(
                (
                    path,
                    int(payload.get("global_step", 0)),
                    int(payload.get("epoch", -1)),
                )
            )
        return out


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer: Optimizer,
        train_loader: DataLoader[Batch],
        val_loader: DataLoader[Batch],
        logger: ExperimentLogger,
        checkpoint_manager: CheckpointManager,
        device: str | torch.device,
        epochs: int,
        grad_clip_norm: float | None = None,
        log_every: int | None = None,
        eval_every: int = 1,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.checkpoints = checkpoint_manager
        self.device = torch.device(device)
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.log_every = log_every
        self.eval_every = eval_every
        self.global_step = 0

        self.model.to(self.device)

    def fit(self) -> None:
        start_time = time.time()
        self.logger.log_message("starting training")
        self.logger.init_epoch_table()

        start_path = self.checkpoints.save(
            name="start.pt",
            model=self.model,
            optimizer=self.optimizer,
            epoch=-1,
            global_step=self.global_step,
            extra_state={"checkpoint_kind": "start"},
        )
        self.logger.log_run_message_file_only(
            f"saved checkpoint epoch=-1 kind=start path={start_path}"
        )

        for epoch in range(self.epochs):
            train_metrics = self._run_epoch(
                loader=self.train_loader,
                epoch=epoch,
                train=True,
            )
            self.logger.log_metrics(
                split="train",
                epoch=epoch,
                metrics=train_metrics.metrics,
            )

            should_eval = (epoch % self.eval_every == 0) or (epoch == self.epochs - 1)
            if should_eval:
                val_metrics = self._run_epoch(
                    loader=self.val_loader,
                    epoch=epoch,
                    train=False,
                )

                self.logger.log_metrics(
                    split="val",
                    epoch=epoch,
                    metrics=val_metrics.metrics,
                )

                self.logger.log_epoch_row(
                    epoch=epoch,
                    train_metrics=train_metrics.metrics,
                    val_metrics=val_metrics.metrics,
                )

                monitored_value = self._extract_monitored_value(val_metrics.metrics)
                best_path = self.checkpoints.maybe_save_best(
                    value=monitored_value,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    global_step=self.global_step,
                    extra_state={"val_metrics": val_metrics.metrics},
                )
                if best_path is not None:
                    self.logger.log_best_metric(
                        epoch=epoch,
                        metric_name=f"val_{self.checkpoints.monitor}",
                        value=monitored_value,
                    )
                    self.logger.log_run_message_file_only(
                        f"saved checkpoint epoch={epoch} kind=best "
                        f"metric=val_{self.checkpoints.monitor} value={monitored_value:.6f} "
                        f"path={best_path}"
                    )

            last_path = self.checkpoints.save(
                name="final.pt",
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                global_step=self.global_step,
                extra_state={"epoch": epoch},
            )
            self.logger.log_run_message_file_only(
                f"saved checkpoint epoch={epoch} kind=last path={last_path}"
            )

            if self.checkpoints.save_every > 0 and (epoch + 1) % self.checkpoints.save_every == 0:
                epoch_path = self.checkpoints.save(
                    name=f"epoch_{epoch:04d}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    global_step=self.global_step,
                    extra_state={"epoch": epoch},
                )
                self.logger.log_run_message_file_only(
                    f"saved checkpoint epoch={epoch} kind=periodic path={epoch_path}"
                )

        mid_best_path = self.checkpoints.save_mid_best_from_available_checkpoints()
        if mid_best_path is not None:
            self.logger.log_run_message_file_only(
                f"saved checkpoint kind=mid_best path={mid_best_path}"
            )

        elapsed = time.time() - start_time
        self.logger.log_message(f"finished training elapsed_seconds={elapsed:.2f}")

    def evaluate(
        self,
        loader: DataLoader[Batch],
        split: str = "test",
        epoch: int = -1,
    ) -> EpochMetrics:
        metrics = self._run_epoch(loader=loader, epoch=epoch, train=False)
        self.logger.log_metrics(
            split=split,
            epoch=epoch,
            metrics=metrics.metrics,
        )
        self.logger.log_message(
            f"{split}: loss={metrics.metrics['loss']:.4f} "
            f"recon_img={metrics.metrics['recon_img']:.4f} "
            f"kl={metrics.metrics['kl']:.4f}"
        )
        return metrics


    def _run_epoch(
        self,
        loader: DataLoader[Batch],
        epoch: int,
        train: bool,
    ) -> EpochMetrics:
        if train:
            self.model.train()
        else:
            self.model.eval()

        tracker = MetricTracker()

        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, self.device)

            with torch.set_grad_enabled(train):
                loss, post_stats = self._step(batch=batch, epoch=epoch, train=train)

            batch_size = int(batch.x_img.shape[0])
            metrics = loss.scalar_metrics()
            metrics.update(post_stats)
            metrics["lr"] = self._current_lr()

            tracker.update(metrics, batch_size=batch_size)

            if train and self.log_every is not None and batch_idx % self.log_every == 0:
                self.logger.log_message(
                    self._format_step_message(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        num_batches=len(loader),
                        metrics=metrics,
                    )
                )

        epoch_metrics = tracker.compute()
        if "loss" not in epoch_metrics:
            raise RuntimeError("epoch metrics did not include 'loss'")

        return EpochMetrics(
            loss=epoch_metrics["loss"],
            metrics=epoch_metrics,
        )

    def _step(
        self,
        batch: Batch,
        epoch: int,
        train: bool,
    ) -> tuple[LossBreakdown, dict[str, float]]: # loss, stats
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        out = self.model(batch)
        loss = self.loss_fn(
            batch=batch,
            out=out,
            step_info=StepInfo(
                epoch=epoch,
                global_step=self.global_step,
            ),
        )

        stats = posterior_stats(out.posterior.mu, out.posterior.logvar)

        if train:
            loss.total.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_norm,
                )

            self.optimizer.step()
            self.global_step += 1

        return loss, stats

    def _current_lr(self) -> float:
        if len(self.optimizer.param_groups) == 0:
            raise RuntimeError("optimizer has no param groups")
        return float(self.optimizer.param_groups[0]["lr"])

    def _extract_monitored_value(self, metrics: Mapping[str, float]) -> float:
        key = self.checkpoints.monitor
        if key not in metrics:
            raise KeyError(
                f"Monitored metric '{key}' not found in metrics: {sorted(metrics.keys())}"
            )
        return float(metrics[key])

    def _format_step_message(
        self,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        metrics: Mapping[str, float],
    ) -> str:
        pieces = [
            f"epoch={epoch}",
            f"step={batch_idx + 1}/{num_batches}",
        ]
        for key, value in metrics.items():
            pieces.append(f"{key}={value:.6f}")
        return " ".join(pieces)
