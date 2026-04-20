# config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    path: Path
    batch_size: int
    num_workers: int
    pin_memory: bool
    use_anchor_features: bool = False

@dataclass
class ModelConfig:
    image_shape: tuple[int, int, int] # (C, H, W)
    latent_dim: int
    hidden_dim: int
    encoder_type: str      # "mlp" or "cnn"
    decoder_type: str      # "mlp" or "cnn"
    obs_distribution: str  # "bernoulli" for dSprites
    anchor_dim: int = 0

@dataclass
class OptimConfig:
    lr: float
    weight_decay: float
    grad_clip_norm: float | None
    epochs: int
    beta_warmup_epochs: int

@dataclass
class RunConfig:
    seed: int
    device: str
    outdir: Path
    log_every: int
    eval_every: int
    save_every: int
