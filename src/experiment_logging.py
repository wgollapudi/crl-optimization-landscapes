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
    """
    Writes:
      - run.log     : general messages
      - metrics.log : scalar metrics, one line per event
      - config.log  : startup configuration dump
    """

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

        # Touch files so they exist immediately.
        self.run_log_path.touch(exist_ok=True)
        self.metrics_log_path.touch(exist_ok=True)
        self.config_log_path.touch(exist_ok=True)

    def log_message(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        self._append(self.run_log_path, line)

    def log_config(self, cfgs: Mapping[str, Any]) -> None:
        lines: list[str] = []
        lines.append(f"[{_timestamp()}] CONFIG")
        lines.append("=" * 80)

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

        lines.append("=" * 80)
        self._append_block(self.config_log_path, lines)

    def log_metrics(
        self,
        split: str,
        epoch: int,
        metrics: Mapping[str, float | int],
        prefix: str | None = None,
    ) -> None:
        fields: list[str] = [f"split={split}", f"epoch={epoch}"]

        if prefix is not None and prefix:
            fields.insert(0, f"tag={prefix}")

        for key, value in metrics.items():
            if isinstance(value, float):
                fields.append(f"{key}={value:.6f}")
            else:
                fields.append(f"{key}={value}")

        line = f"[{_timestamp()}] " + " ".join(fields)
        self._append(self.metrics_log_path, line)

    def log_epoch_header(self, epoch: int) -> None:
        self.log_message(f"===== EPOCH {epoch} =====")

    def log_checkpoint(self, path: str | Path, kind: str) -> None:
        self.log_message(f"saved_checkpoint kind={kind} path={Path(path)}")

    def log_best_metric(
        self,
        metric_name: str,
        value: float,
        epoch: int,
    ) -> None:
        self.log_message(
            f"new_best metric={metric_name} value={value:.6f} epoch={epoch}"
        )

    def close(self) -> None:
        # Present for symmetry / future extension.
        pass

    def _append(self, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            print(line)

    def _append_block(self, path: Path, lines: list[str]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")
        if self.echo:
            for line in lines:
                print(line)
