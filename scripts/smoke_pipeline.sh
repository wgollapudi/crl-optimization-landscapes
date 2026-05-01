#!/usr/bin/env bash
# Run a tiny end-to-end pipeline as a proof of concept.
#
# This script intentionally writes to a disposable directory outside the normal
# data/runs/landscape_runs outputs. It exercises:
#   dataset generation -> training -> landscape probes -> landscape analysis
#
# The defaults are small enough for a quick CPU sanity check, not for science.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/smoke_pipeline.sh [options]

Options:
  --outdir DIR             Output root for all smoke artifacts
                           (default: smoke_runs/poc_<timestamp>)
  --dsprites PATH          Raw dSprites NPZ
                           (default: src/data/raw/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz)
  --seeds INT              Number of seeds per regime; use >=2 for pairwise probes (default: 2)
  --epochs INT             Training epochs per run (default: 2)
  --device DEVICE          Training/probe device (default: cpu)
  --figures                Ask analyze_landscapes.py to make figures if matplotlib is installed
  -h, --help               Show this help message

Environment:
  PYTHON                   Python executable to use (default: python)

Example:
  scripts/smoke_pipeline.sh --outdir /tmp/crl_landscape_poc --epochs 1
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

resolve_path() {
  "$PYTHON" - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"

OUTDIR="$REPO_ROOT/smoke_runs/poc_$(timestamp)"
DSPRITES="$SRC_DIR/data/raw/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
SEEDS=2
EPOCHS=2
DEVICE="cpu"
MAKE_FIGURES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      [[ $# -ge 2 ]] || die "--outdir requires a value"
      OUTDIR="$2"
      shift 2
      ;;
    --dsprites)
      [[ $# -ge 2 ]] || die "--dsprites requires a value"
      DSPRITES="$2"
      shift 2
      ;;
    --seeds)
      [[ $# -ge 2 ]] || die "--seeds requires a value"
      SEEDS="$2"
      shift 2
      ;;
    --epochs)
      [[ $# -ge 2 ]] || die "--epochs requires a value"
      EPOCHS="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device requires a value"
      DEVICE="$2"
      shift 2
      ;;
    --figures)
      MAKE_FIGURES=1
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

command -v "$PYTHON" >/dev/null 2>&1 || die "Python executable not found: $PYTHON"

OUTDIR="$(resolve_path "$OUTDIR")"
DSPRITES="$(resolve_path "$DSPRITES")"

[[ -f "$DSPRITES" ]] || die "Raw dSprites file not found: $DSPRITES"
[[ "$SEEDS" =~ ^[0-9]+$ ]] || die "--seeds must be an integer"
[[ "$EPOCHS" =~ ^[0-9]+$ ]] || die "--epochs must be an integer"
[[ "$SEEDS" -ge 1 ]] || die "--seeds must be >= 1"
[[ "$EPOCHS" -ge 1 ]] || die "--epochs must be >= 1"

DATA_DIR="$OUTDIR/data/crl_dsprites"
RUN_ROOT="$OUTDIR/runs"
LANDSCAPE_ROOT="$OUTDIR/landscape_runs"
ANALYSIS_ROOT="$OUTDIR/landscape_analysis"
LOG_DIR="$OUTDIR/logs"

mkdir -p "$DATA_DIR" "$RUN_ROOT" "$LANDSCAPE_ROOT" "$ANALYSIS_ROOT" "$LOG_DIR"

echo "[smoke] repo: $REPO_ROOT"
echo "[smoke] outdir: $OUTDIR"
echo "[smoke] python: $PYTHON"
echo "[smoke] device: $DEVICE"

cd "$SRC_DIR"

echo "[smoke] generating tiny datasets"
"$PYTHON" make_datasets.py \
  --dsprites "$DSPRITES" \
  --outdir "$DATA_DIR" \
  --shape ellipse \
  --seed 0 \
  --n-obs 256 \
  --n-per-intervention 64 \
  --overlap-grid-budget 256 \
  --log-file "$LOG_DIR/dataset_generation.log"

declare -A DATASETS=(
  [regimeA]="observational.npz"
  [regimeB]="overlap_support.npz"
  [regimeC]="single_node_interventions.npz"
  [regimeD]="two_interventions_per_node.npz"
)

for regime in regimeA regimeB regimeC regimeD; do
  echo "[smoke] training $regime"
  for ((seed = 0; seed < SEEDS; seed++)); do
    "$PYTHON" train.py \
      --data-path "$DATA_DIR/${DATASETS[$regime]}" \
      --outdir "$RUN_ROOT/$regime/seed_$seed" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --batch-size 64 \
      --hidden-dim 64 \
      --device "$DEVICE" \
      --eval-every 1 \
      --save-every 1 \
      > "$LOG_DIR/train_${regime}_seed_${seed}.log" 2>&1
  done
done

for regime in regimeA regimeB regimeC regimeD; do
  echo "[smoke] probing $regime"
  max_pairs=1
  if [[ "$SEEDS" -lt 2 ]]; then
    max_pairs=0
  fi

  "$PYTHON" landscape.py \
    --regime "$regime" \
    --run-root "$RUN_ROOT/$regime" \
    --outdir "$LANDSCAPE_ROOT" \
    --checkpoint-kind all \
    --probes endpoint perturbation slice1d slice2d gradnorm hessian pairwise \
    --device "$DEVICE" \
    --batch-size 64 \
    --train-subset-n 64 \
    --val-subset-n 32 \
    --max-batches 1 \
    --radii 1e-4 1e-3 \
    --directions-per-radius 2 \
    --slice-points 5 \
    --slice-alpha-max 0.1 \
    --slice2d-points 3 \
    --slice2d-alpha-max 0.1 \
    --slice2d-max-runs "$SEEDS" \
    --interp-points 5 \
    --max-pairs "$max_pairs" \
    --hessian-power-iters 2 \
    --hessian-power-restarts 1 \
    --hessian-trace-samples 2 \
    --hessian-max-batches 1 \
    > "$LOG_DIR/landscape_${regime}.log" 2>&1
done

echo "[smoke] analyzing landscape artifacts"
analysis_figure_flag="--no-figures"
if [[ "$MAKE_FIGURES" -eq 1 ]]; then
  analysis_figure_flag="--make-figures"
fi

"$PYTHON" analyze_landscapes.py \
  --landscape-root "$LANDSCAPE_ROOT" \
  --outdir "$ANALYSIS_ROOT" \
  --checkpoint-kind all \
  --splits val \
  --bootstrap-samples 100 \
  "$analysis_figure_flag" \
  > "$LOG_DIR/analyze_landscapes.log" 2>&1

echo "[smoke] done"
echo "[smoke] outputs:"
echo "  datasets:  $DATA_DIR"
echo "  runs:      $RUN_ROOT"
echo "  probes:    $LANDSCAPE_ROOT"
echo "  analysis:  $ANALYSIS_ROOT"
echo "  logs:      $LOG_DIR"
echo
echo "[smoke] key analysis tables:"
echo "  $ANALYSIS_ROOT/tables/seed_table.csv"
echo "  $ANALYSIS_ROOT/tables/pair_table.csv"
echo "  $ANALYSIS_ROOT/tables/regime_differences.csv"
