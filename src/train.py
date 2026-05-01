# train.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.optim import Adam

from datetime import datetime
from config import DataConfig, ModelConfig, OptimConfig, RunConfig
from data import DataModule
from engine import CheckpointManager, Trainer
from experiment_logging import ExperimentLogger
from losses import AnchorLossConfig, LinearWarmupSchedule, VAELoss
from models.causal_discrepancy_vae import CausalDiscrepancyVAE
from models.plain_vae import PlainVAE, PlainVAEConfig
from models.sparse_vae import SparseVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def infer_regime_name(data_path: Path) -> str:
    stem = data_path.stem.lower()
    REGIME_MAP = {
        "observational": "regimeA",
        "overlap_support": "regimeB",
        "single_node_interventions": "regimeC",
        "two_interventions_per_node": "regimeD",
    }
    if stem in REGIME_MAP.keys():
        return REGIME_MAP[stem]
    return "unknown"


def make_config_hash(args: argparse.Namespace) -> str:
    relevant = {
        "data_path": str(args.data_path),
        "batch_size": args.batch_size,
        "use_anchor_features": args.use_anchor_features,
        "model_name": args.model_name,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "anchor_dim": args.anchor_dim,
        "sparse_lambda": args.sparse_lambda,
        "mmd_weight": args.mmd_weight,
        "graph_l1_weight": args.graph_l1_weight,
        "num_intervention_variants": args.num_intervention_variants,
        "num_intervention_envs": args.num_intervention_envs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "epochs": args.epochs,
        "beta_warmup_epochs": args.beta_warmup_epochs,
        "seed": args.seed,
    }
    payload = repr(sorted(relevant.items())).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:8]


def resolve_outdir(args: argparse.Namespace) -> Path:
    if args.outdir not in {None, "", "auto"}:
        return Path(args.outdir)

    regime = infer_regime_name(args.data_path)
    cfg_hash = make_config_hash(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{regime}" / f"hash-{cfg_hash}_time-{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE on one CRL regime dataset.")

    # Data
    parser.add_argument("--data-path", type=Path, required=True, help="Path to one regime .npz file")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--use-anchor-features", action="store_true")

    # Model
    parser.add_argument(
        "--model-name",
        choices=["plain", "sparse", "causal_discrepancy"],
        default="plain",
    )
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--anchor-dim", type=int, default=0)
    parser.add_argument("--image-channels", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--sparse-lambda", type=float, default=1e-3)
    parser.add_argument("--mmd-weight", type=float, default=1.0)
    parser.add_argument("--graph-l1-weight", type=float, default=1e-3)
    parser.add_argument("--num-intervention-variants", type=int, default=2)
    parser.add_argument(
        "--num-intervention-envs",
        type=int,
        default=0,
        help="Number of env_id labels for CD-VAE intervention encoder; 0 means infer from dataset",
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--beta-warmup-epochs", type=int, default=10)

    # Run
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)

    return parser.parse_args()


def normalize_model_args(args: argparse.Namespace) -> None:
    if args.model_name == "sparse":
        if not args.use_anchor_features:
            raise ValueError("--model-name sparse requires --use-anchor-features")
        if args.anchor_dim == 0:
            args.anchor_dim = 2 * args.latent_dim

    if args.model_name == "causal_discrepancy":
        if args.use_anchor_features and args.anchor_dim == 0:
            args.anchor_dim = 2 * args.latent_dim
        if (not args.use_anchor_features) and args.anchor_dim != 0:
            raise ValueError(
                "--anchor-dim > 0 requires --use-anchor-features for "
                "causal_discrepancy"
            )


def build_configs(args: argparse.Namespace) -> tuple[DataConfig, ModelConfig, OptimConfig, RunConfig]:
    data_cfg = DataConfig(
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_anchor_features=args.use_anchor_features,
    )

    model_cfg = ModelConfig(
        image_shape=(args.image_channels, args.image_height, args.image_width),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        encoder_type="mlp",
        decoder_type="mlp",
        obs_distribution="bernoulli",
        anchor_dim=args.anchor_dim,
        model_name=args.model_name,
        sparse_lambda=args.sparse_lambda,
        mmd_weight=args.mmd_weight,
        graph_l1_weight=args.graph_l1_weight,
        num_intervention_variants=args.num_intervention_variants,
        num_intervention_envs=args.num_intervention_envs,
    )

    optim_cfg = OptimConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        epochs=args.epochs,
        beta_warmup_epochs=args.beta_warmup_epochs,
    )

    run_cfg = RunConfig(
        seed=args.seed,
        device=args.device,
        outdir=args.outdir,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )

    return data_cfg, model_cfg, optim_cfg, run_cfg


def validate_data_summary(model_cfg: ModelConfig, data_summary: dict) -> None:
    if model_cfg.model_name in {"sparse", "causal_discrepancy"} and model_cfg.anchor_dim > 0:
        dataset_anchor_dim = data_summary.get("anchor_dim")
        if dataset_anchor_dim is None:
            raise ValueError(
                f"{model_cfg.model_name} requires anchor features in the dataset; regenerate data "
                "with anchor_features and pass --use-anchor-features"
            )
        if int(dataset_anchor_dim) != model_cfg.anchor_dim:
            raise ValueError(
                f"--anchor-dim={model_cfg.anchor_dim} does not match dataset "
                f"anchor_dim={dataset_anchor_dim}"
            )


def build_plain_vae_config(model_cfg: ModelConfig) -> PlainVAEConfig:
    return PlainVAEConfig(
        image_shape=tuple(model_cfg.image_shape),
        latent_dim=model_cfg.latent_dim,
        hidden_dim=model_cfg.hidden_dim,
        anchor_dim=model_cfg.anchor_dim,
    )

def build_model(model_cfg: ModelConfig) -> torch.nn.Module:
    if model_cfg.model_name == "plain":
        return PlainVAE(build_plain_vae_config(model_cfg))

    if model_cfg.model_name == "sparse":
        return SparseVAE.from_model_config(model_cfg)

    if model_cfg.model_name == "causal_discrepancy":
        return CausalDiscrepancyVAE.from_model_config(model_cfg)

    raise ValueError(f"Unknown model_name: {model_cfg.model_name}")

def main() -> None:
    args = parse_args()
    normalize_model_args(args)
    args.outdir = resolve_outdir(args)
    data_cfg, model_cfg, optim_cfg, run_cfg = build_configs(args)

    run_cfg.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(run_cfg.seed)

    logger = ExperimentLogger(run_cfg.outdir, echo=True)
    logger.log_message("initializing run")

    dm = DataModule(data_cfg, include_metadata=False)
    dm.setup()
    data_summary = dm.summary()
    if model_cfg.model_name == "causal_discrepancy" and model_cfg.num_intervention_envs == 0:
        model_cfg.num_intervention_envs = int(data_summary["num_intervention_envs"])
    validate_data_summary(model_cfg, data_summary)

    model = build_model(model_cfg)

    optimizer = Adam(
        model.parameters(),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )

    loss_fn = VAELoss(
        beta_schedule=LinearWarmupSchedule(
            start_value=0.0,
            end_value=1.0,
            warmup_epochs=optim_cfg.beta_warmup_epochs,
        ),
        anchor_cfg=AnchorLossConfig(
            kind="mse",
            weight=1.0,
        ),
    )

    checkpoints = CheckpointManager(
        outdir=run_cfg.outdir,
        monitor="loss",
        mode="min",
        save_every=run_cfg.save_every,
    )

    logger.write_config(
        {
            "data": data_cfg,
            "data_summary": data_summary,
            "model": model_cfg,
            "optim": optim_cfg,
            "run": run_cfg,
            "model_class": model.__class__.__name__,
            "model_checkpoint_config": (
                model.config_dict() if hasattr(model, "config_dict") else None
            ),
        }
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=dm.train_loader(),
        val_loader=dm.val_loader(),
        logger=logger,
        checkpoint_manager=checkpoints,
        device=run_cfg.device,
        epochs=optim_cfg.epochs,
        grad_clip_norm=optim_cfg.grad_clip_norm,
        log_every=run_cfg.log_every,
        eval_every=run_cfg.eval_every,
    )

    trainer.fit()
    trainer.evaluate(dm.test_loader(), split="test", epoch=optim_cfg.epochs)
    logger.log_message("run complete")
    logger.close()


if __name__ == "__main__":
    main()
