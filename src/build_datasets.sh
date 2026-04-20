#!/usr/bin/env bash
# wrapper to download raw dSprites and run make_datasets.py.
#
# Default behavior
# - downloads the official dSprites NPZ if it is missing
# - writes it to <root>/raw/
# - writes generated CRL datasets to <root>/crl_dsprites/
# - fixes shape to ellipse
# - uses the defaults already encoded in make_datasets.py
#
# Ex.
# ./build_datasets.sh
# ./build_datasets.sh --root ./data
# ./build_datasets.sh --shape heart

set -euo pipefail

DSPRITES_URL="https://github.com/google-deepmind/dsprites-dataset/raw/refs/heads/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
DSPRITES_FILENAME="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

ROOT="data"
SHAPE="ellipse"
SEED="0"
GENERATOR=""
FORCE_DOWNLOAD=0
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage: build_datasets.sh [options]

Options:
  --root DIR            Root directory for raw and generated datasets (default: ./data)
  --shape SHAPE         Fixed dSprites shape: square | ellipse | heart (default: ellipse)
  --seed INT            Seed passed through to make_datasets.py (default: 0)
  --generator PATH      Path to make_datasets.py (default: sibling of this script)
  --force-download      Re-download the raw dSprites NPZ even if it already exists
  --overwrite           Allow writing into a non-empty output directory
  -h, --help            Show this help message
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_file() {
  local url="$1"
  local dst="$2"
  local tmp="${dst}.part"

  mkdir -p "$(dirname "$dst")"

  if [[ -f "$dst" && "$FORCE_DOWNLOAD" -eq 0 ]]; then
    echo "Using existing raw dSprites file at $dst"
    return
  fi

  rm -f "$tmp"
  echo "Downloading dSprites to $dst ..."

  if have_cmd curl; then
    curl -L --fail --output "$tmp" "$url"
  elif have_cmd wget; then
    wget -O "$tmp" "$url"
  else
    die "Neither curl nor wget is available."
  fi

  mv "$tmp" "$dst"
}

resolve_path() {
  python3 - "$1" <<'PY'
import pathlib, sys
print(pathlib.Path(sys.argv[1]).resolve())
PY
}

is_nonempty_dir() {
  local dir="$1"
  [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR_DEFAULT="$SCRIPT_DIR/make_datasets.py"
GENERATOR="$GENERATOR_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || die "--root requires an argument"
      ROOT="$2"
      shift 2
      ;;
    --shape)
      [[ $# -ge 2 ]] || die "--shape requires an argument"
      SHAPE="$2"
      case "$SHAPE" in
        square|ellipse|heart) ;;
        *) die "Invalid shape '$SHAPE' (expected square, ellipse, or heart)" ;;
      esac
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || die "--seed requires an argument"
      SEED="$2"
      shift 2
      ;;
    --generator)
      [[ $# -ge 2 ]] || die "--generator requires an argument"
      GENERATOR="$2"
      shift 2
      ;;
    --force-download)
      FORCE_DOWNLOAD=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

have_cmd python3 || die "python3 is required."

ROOT="$(resolve_path "$ROOT")"
GENERATOR="$(resolve_path "$GENERATOR")"

[[ -f "$GENERATOR" ]] || die "Could not find generator script: $GENERATOR"

RAW_DIR="$ROOT/raw"
OUTDIR="$ROOT/crl_dsprites"
RAW_NPZ="$RAW_DIR/$DSPRITES_FILENAME"

if is_nonempty_dir "$OUTDIR" && [[ "$OVERWRITE" -eq 0 ]]; then
  die "Output directory \"$OUTDIR\" is non-empty, pass --overwrite to overwrite."
fi

download_file "$DSPRITES_URL" "$RAW_NPZ"
mkdir -p "$OUTDIR"

CMD=(
  python3
  "$GENERATOR"
  --dsprites "$RAW_NPZ"
  --outdir "$OUTDIR"
  --shape "$SHAPE"
  --seed "$SEED"
  --log-file "$OUTDIR/generation.log"
)

echo "Running dataset generator..."
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done. Generated datasets are in $OUTDIR"
