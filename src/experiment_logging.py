# experiment_logging.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
import pprint


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _stringify_value(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _flatten_mapping(
    mapping: Mapping[str, Any],
    prefix: str = "",
) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if is_dataclass(value):
            value = asdict(value)

        if isinstance(value, Mapping):
            items.extend(_flatten_mapping(value, prefix=full_key))
        else:
            items.append((full_key, value))
    return items


class ExperimentLogger:
    def __init__(
        self,
        outdir: str | Path,
        run_log_name: str = "run.log",
        metrics_log_name: str = "metrics.log",
        config_log_name: str = "config.log",
        echo: bool = True,
    ) -> None:
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.run_log_path = self.outdir / run_log_name
        self.metrics_log_path = self.outdir / metrics_log_name
        self.config_log_path = self.outdir / config_log_name

        self.echo = echo

        self._summary_rows: list[str] = []
        self._detail_lines: list[str] = []
        self._metrics_initialized = False

        # Track previous metrics per split for warning generation.
        self._prev_metrics_by_split: dict[str, dict[str, float]] = {}

        # Heuristic thresholds
        self._collapsed_dim_threshold = 0.05
        self._tiny_all_dims_threshold = 0.10
        self._posterior_collapse_kl_threshold = 0.50
        self._meaningful_beta_threshold = 0.25
        self._variance_explosion_threshold = 2.0
        self._variance_very_high_threshold = 5.0
        self._mu_mean_abs_large_threshold = 10.0
        self._mu_mean_jump_threshold = 2.5
        self._logvar_mean_jump_threshold = 1.5
        self._kl_explosion_threshold = 500.0
        self._beta_high_threshold = 1.5

    def log_message(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        self._append(self.run_log_path, line)

    def write_config(self, cfgs: Mapping[str, Any]) -> None:
        lines: list[str] = []
        lines.append(f"[{_timestamp()}] CONFIG")
        lines.append("=" * 100)

        for key, value in cfgs.items():
            if is_dataclass(value):
                value = asdict(value)

            lines.append(f"{key}:")
            if isinstance(value, Mapping):
                flat = _flatten_mapping(value)
                for subkey, subval in flat:
                    lines.append(f"  {subkey} = {_stringify_value(subval)}")
            else:
                pretty = pprint.pformat(value, sort_dicts=False)
                for ln in pretty.splitlines():
                    lines.append(f"  {ln}")
            lines.append("")

        lines.append("=" * 100)

        with self.config_log_path.open("w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")

    def init_epoch_table(self) -> None:
        header = (
            f"{'timestamp':<19}  "
            f"{'epoch':>5}  "
            f"{'train_loss':>11}  {'val_loss':>11}  "
            f"{'train_recon':>11}  {'val_recon':>11}  "
            f"{'train_kl':>10}  {'val_kl':>10}  "
            f"{'beta':>7}"
        )

        with self.metrics_log_path.open("w", encoding="utf-8") as fh:
            fh.write("")

        self._summary_rows.clear()
        self._detail_lines.clear()
        self._prev_metrics_by_split.clear()
        self._metrics_initialized = True

        if self.echo:
            print(header)

    def log_epoch_row(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float],
    ) -> None:
        file_line = (
            f"{epoch:<5d} | "
            f"{train_metrics['loss']:10.4f} | {val_metrics['loss']:10.4f} | "
            f"{train_metrics['recon_img']:11.4f} | {val_metrics['recon_img']:11.4f} | "
            f"{train_metrics['kl']:9.4f} | {val_metrics['kl']:9.4f} | "
            f"{train_metrics['beta']:6.4f}"
        )
        self._summary_rows.append(file_line)

        console_line = (
            f"{_timestamp():<19}  "
            f"{epoch:5d}  "
            f"{train_metrics['loss']:11.4f}  {val_metrics['loss']:11.4f}  "
            f"{train_metrics['recon_img']:11.4f}  {val_metrics['recon_img']:11.4f}  "
            f"{train_metrics['kl']:10.4f}  {val_metrics['kl']:10.4f}  "
            f"{train_metrics['beta']:7.4f}"
        )
        if self.echo:
            print(console_line)

    def log_best_metric(
        self,
        epoch: int,
        metric_name: str,
        value: float,
    ) -> None:
        self._detail_lines.append(
            f"[{_timestamp()}] best epoch={epoch} {metric_name}={value:.6f}"
        )

    def close(self) -> None:
        if not self._metrics_initialized:
            return

        with self.metrics_log_path.open("w", encoding="utf-8") as fh:
            fh.write("=" * 100 + "\n")
            fh.write("EPOCH SUMMARY\n")
            fh.write("=" * 100 + "\n")
            fh.write(
                f"{'epoch':<5} | "
                f"{'train_loss':>10} | {'val_loss':>10} | "
                f"{'train_recon':>11} | {'val_recon':>11} | "
                f"{'train_kl':>9} | {'val_kl':>9} | "
                f"{'beta':>6}\n"
            )
            fh.write("-" * 100 + "\n")
            for row in self._summary_rows:
                fh.write(row + "\n")

            fh.write("\n")
            fh.write("=" * 100 + "\n")
            fh.write("DETAILED METRICS\n")
            fh.write("=" * 100 + "\n\n")
            for line in self._detail_lines:
                fh.write(line + "\n")

    def _append(self, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            print(line)

    def _append_silent(self, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def log_run_message_file_only(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        self._append_silent(self.run_log_path, line)

    def _format_float(self, value: float | int | None, digits: int = 4) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"

    def _extract_latent_vector(
        self,
        metrics: Mapping[str, float | int],
        prefix: str,
    ) -> list[float]:
        values: list[float] = []
        i = 0
        while f"{prefix}_{i}" in metrics:
            values.append(float(metrics[f"{prefix}_{i}"]))
            i += 1
        return values

    def _format_vector(self, values: list[float], digits: int = 2) -> str:
        if not values:
            return "[]"
        return "[" + ", ".join(f"{v:.{digits}f}" for v in values) + "]"

    def _warning_lines_for_metrics(
        self,
        split: str,
        epoch: int,
        metrics: Mapping[str, float | int],
    ) -> list[str]:
        warnings: list[str] = []

        mu_std = self._extract_latent_vector(metrics, "mu_std")
        mu_mean = self._extract_latent_vector(metrics, "mu_mean")

        beta = float(metrics["beta"]) if "beta" in metrics else None
        kl = float(metrics["kl"]) if "kl" in metrics else None
        logvar_mean = float(metrics["logvar_mean"]) if "logvar_mean" in metrics else None

        prev = self._prev_metrics_by_split.get(split)

        # Collapsed latent dimensions
        if mu_std:
            collapsed_dims = [i for i, v in enumerate(mu_std) if v < self._collapsed_dim_threshold]
            if collapsed_dims:
                warnings.append(
                    f"WARNING: possible collapsed latent dimensions on {split} epoch={epoch}: "
                    f"indices={collapsed_dims} threshold={self._collapsed_dim_threshold:.3f}"
                )

        # Posterior collapse overall
        if mu_std and beta is not None and kl is not None:
            if all(v < self._tiny_all_dims_threshold for v in mu_std) and beta >= self._meaningful_beta_threshold and kl < self._posterior_collapse_kl_threshold:
                warnings.append(
                    f"WARNING: possible posterior collapse on {split} epoch={epoch}: "
                    f"all mu_std < {self._tiny_all_dims_threshold:.3f}, beta={beta:.4f}, kl={kl:.4f}"
                )

        # Variance explosion / very high variance
        if logvar_mean is not None:
            if logvar_mean > self._variance_very_high_threshold:
                warnings.append(
                    f"WARNING: severe variance explosion on {split} epoch={epoch}: "
                    f"logvar_mean={logvar_mean:.4f} > {self._variance_very_high_threshold:.2f}"
                )
            elif logvar_mean > self._variance_explosion_threshold:
                warnings.append(
                    f"WARNING: possible variance explosion on {split} epoch={epoch}: "
                    f"logvar_mean={logvar_mean:.4f} > {self._variance_explosion_threshold:.2f}"
                )

        # Large latent means
        if mu_mean:
            large_dims = [i for i, v in enumerate(mu_mean) if abs(v) > self._mu_mean_abs_large_threshold]
            if large_dims:
                warnings.append(
                    f"WARNING: unusually large latent means on {split} epoch={epoch}: "
                    f"indices={large_dims} threshold={self._mu_mean_abs_large_threshold:.2f}"
                )

        # KL behavior
        if kl is not None:
            if kl < 0.0:
                warnings.append(
                    f"WARNING: invalid KL on {split} epoch={epoch}: kl={kl:.6f} < 0"
                )
            elif kl > self._kl_explosion_threshold:
                warnings.append(
                    f"WARNING: unusually large KL on {split} epoch={epoch}: "
                    f"kl={kl:.4f} > {self._kl_explosion_threshold:.1f}"
                )
            elif beta is not None and beta >= self._meaningful_beta_threshold and kl < self._posterior_collapse_kl_threshold:
                warnings.append(
                    f"WARNING: KL is very small despite nontrivial beta on {split} epoch={epoch}: "
                    f"beta={beta:.4f}, kl={kl:.4f}"
                )

        # Beta behavior
        if beta is not None:
            if beta < 0.0:
                warnings.append(
                    f"WARNING: beta is negative on {split} epoch={epoch}: beta={beta:.4f}"
                )
            elif beta > self._beta_high_threshold:
                warnings.append(
                    f"WARNING: beta is unusually large on {split} epoch={epoch}: "
                    f"beta={beta:.4f} > {self._beta_high_threshold:.2f}"
                )

            if prev is not None and "beta" in prev:
                prev_beta = float(prev["beta"])
                if beta + 1e-12 < prev_beta:
                    warnings.append(
                        f"WARNING: beta decreased on {split} epoch={epoch}: "
                        f"prev_beta={prev_beta:.4f}, beta={beta:.4f}"
                    )

        # Unstable epoch-to-epoch drift
        if prev is not None:
            prev_mu_mean = self._extract_latent_vector(prev, "mu_mean")
            if mu_mean and prev_mu_mean and len(mu_mean) == len(prev_mu_mean):
                jump_dims = [
                    i for i, (a, b) in enumerate(zip(mu_mean, prev_mu_mean))
                    if abs(a - b) > self._mu_mean_jump_threshold
                ]
                if jump_dims:
                    warnings.append(
                        f"WARNING: large epoch-to-epoch jump in mu_mean on {split} epoch={epoch}: "
                        f"indices={jump_dims} threshold={self._mu_mean_jump_threshold:.2f}"
                    )

            if logvar_mean is not None and "logvar_mean" in prev:
                prev_logvar_mean = float(prev["logvar_mean"])
                if abs(logvar_mean - prev_logvar_mean) > self._logvar_mean_jump_threshold:
                    warnings.append(
                        f"WARNING: large epoch-to-epoch jump in logvar_mean on {split} epoch={epoch}: "
                        f"prev={prev_logvar_mean:.4f}, current={logvar_mean:.4f}, "
                        f"threshold={self._logvar_mean_jump_threshold:.2f}"
                    )

        return warnings

    def log_metrics(
        self,
        split: str,
        epoch: int,
        metrics: Mapping[str, float | int],
        tag: str | None = None,
    ) -> None:
        prefix = f"[{_timestamp()}] {split} epoch={epoch}"
        if tag is not None:
            prefix += f" tag={tag}"

        summary_line = (
            f"{prefix} | "
            f"loss={self._format_float(metrics.get('loss'))} "
            f"recon={self._format_float(metrics.get('recon_img'))} "
            f"kl={self._format_float(metrics.get('kl'))} "
            f"beta={self._format_float(metrics.get('beta'))}"
        )
        self._detail_lines.append(summary_line)

        mu_std = self._extract_latent_vector(metrics, "mu_std")
        mu_mean = self._extract_latent_vector(metrics, "mu_mean")
        logvar_mean_dim = self._extract_latent_vector(metrics, "logvar_mean")

        detail_parts: list[str] = []
        if mu_std:
            detail_parts.append(f"mu_std={self._format_vector(mu_std)}")
        if mu_mean:
            detail_parts.append(f"mu_mean={self._format_vector(mu_mean)}")
        if logvar_mean_dim:
            detail_parts.append(f"logvar_mean_dim={self._format_vector(logvar_mean_dim)}")
        if "mu_abs_mean" in metrics:
            detail_parts.append(f"mu_abs_mean={self._format_float(metrics['mu_abs_mean'], digits=2)}")
        if "logvar_mean" in metrics:
            detail_parts.append(f"logvar_mean={self._format_float(metrics['logvar_mean'], digits=2)}")
        if "logvar_std" in metrics:
            detail_parts.append(f"logvar_std={self._format_float(metrics['logvar_std'], digits=2)}")
        if "lr" in metrics:
            detail_parts.append(f"lr={self._format_float(metrics['lr'], digits=6)}")

        standard_keys = {
            "loss",
            "recon_img",
            "kl",
            "beta",
            "lr",
            "mu_abs_mean",
            "logvar_mean",
            "logvar_std",
        }
        for key in sorted(metrics):
            if key in standard_keys:
                continue
            if key.startswith(("mu_std_", "mu_mean_", "logvar_mean_")):
                continue
            detail_parts.append(f"{key}={self._format_float(metrics[key])}")

        if detail_parts:
            self._detail_lines.append("    " + " | ".join(detail_parts))

        for warning in self._warning_lines_for_metrics(split=split, epoch=epoch, metrics=metrics):
            self._detail_lines.append("    " + warning)

        self._detail_lines.append("")

        self._prev_metrics_by_split[split] = {k: float(v) for k, v in metrics.items()}
