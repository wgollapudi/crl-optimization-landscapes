#!/usr/bin/env python3
"""wrapper to download raw dSprites and run make_datasets.py.

Place this next to `make_datasets.py` or pass `--generator` explicitly.

Default behavior
- downloads the official dSprites NPZ if it is missing
- writes it to <root>/raw/
- writes generated CRL datasets to <root>/crl_dsprites/
- fixes shape to ellipse
- uses the defaults already encoded in make_datasets.py

Ex.
python build_dsprites.py
python build_dsprites.py --root ./data
python build_dsprites.py --shape heart
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

DSPRITES_URL = (
    "https://github.com/google-deepmind/dsprites-dataset/raw/refs/heads/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
DSPRITES_FILENAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Root directory for raw and generated datasets (default: ./data)",
    )
    parser.add_argument(
        "--shape",
        choices=["square", "ellipse", "heart"],
        default="ellipse",
        help="Fixed dSprites shape passed through to make_datasets.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed passed through to make_datasets.py",
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path(__file__).with_name("make_datasets.py"),
        help="Path to make_datasets.py (default: sibling of this script)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the raw dSprites NPZ even if it already exists",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def download_if_needed(dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"Using existing raw dSprites file at {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading dSprites to {dst} ...")
    with urlopen(DSPRITES_URL) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(dst)


def ensure_generator(path: Path) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find generator script: {path}")
    return path


def main() -> None:
    args = parse_args()

    root = args.root.resolve()
    raw_dir = root / "raw"
    outdir = root / "crl_dsprites"
    raw_npz = raw_dir / DSPRITES_FILENAME
    generator = ensure_generator(args.generator)

    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Refusing to write into non-empty output directory: {outdir}\n"
            "Pass --overwrite if that is intentional."
        )

    download_if_needed(raw_npz, force=args.force_download)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(generator),
        "--dsprites",
        str(raw_npz),
        "--outdir",
        str(outdir),
        "--shape",
        args.shape,
        "--seed",
        str(args.seed),
        "--log-file",
        str(outdir / "generation.log"),
    ]

    print("Running dataset generator...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done. Generated datasets are in {outdir}")


if __name__ == "__main__":
    main()

