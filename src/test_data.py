# .test_data.py

# test_data.py

from pathlib import Path
import torch

from config import DataConfig
from data import DataModule


def check_batch(batch):
    print("=== BATCH CHECK ===")

    # Required fields
    assert batch.x_img.ndim == 4, "x_img should be [B, C, H, W]"
    B, C, H, W = batch.x_img.shape
    print(f"x_img: {batch.x_img.shape}")

    assert C == 1, "dSprites should have 1 channel"
    assert H == 64 and W == 64, "Expected 64x64 images"

    assert batch.env_id.shape == (B,)
    assert batch.intervention_target.shape == (B,)
    assert batch.intervention_variant.shape == (B,)

    print("env_id:", batch.env_id[:5])
    print("intervention_target:", batch.intervention_target[:5])
    print("intervention_variant:", batch.intervention_variant[:5])

    # Optional fields
    if batch.x_anchor is not None:
        print("x_anchor:", batch.x_anchor.shape)

    if batch.z_true is not None:
        print("z_true:", batch.z_true.shape)

    # NaN / inf checks
    assert torch.isfinite(batch.x_img).all(), "NaNs in x_img"
    if batch.x_anchor is not None:
        assert torch.isfinite(batch.x_anchor).all(), "NaNs in x_anchor"
    if batch.z_true is not None:
        assert torch.isfinite(batch.z_true).all(), "NaNs in z_true"

    print("batch OK\n")


def main():
    # Change this path if needed
    data_path = Path("data/crl_dsprites/observational.npz")

    cfg = DataConfig(
        path=data_path,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        use_anchor_features=False,  # True later for Regime B
    )

    dm = DataModule(cfg, include_metadata=True)
    dm.setup()

    summary = dm.summary()
    print("=== DATA SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print()

    # Check each split
    for name, loader_fn in [
        ("train", dm.train_loader),
        ("val", dm.val_loader),
        ("test", dm.test_loader),
    ]:
        print(f"--- {name.upper()} ---")
        loader = loader_fn()
        batch = next(iter(loader))
        check_batch(batch)

    print("All checks passed.")


if __name__ == "__main__":
    main()
