#!/usr/bin/env bash
# Full training + landscape extraction + analysis pipeline.
#
# This is intended for long cluster runs. It is resumable by default: training
# skips run directories with final.pt, and landscape.py skips existing probe
# artifacts unless --overwrite-landscape is passed.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_big_pipeline.sh [options]

Core options:
  --out-root DIR                  Root for runs/probes/analysis (default: .)
  --data-dir DIR                  Directory containing regime .npz datasets
                                  (default: src/data/crl_dsprites)
  --device DEVICE                 cpu | cuda | cuda:0 | ... (default: cuda)
  --seeds INT                     Seeds per regime (default: 20)
  --seed-start INT                First seed value (default: 0)
  --epochs INT                    Training epochs (default: 100)
  --batch-size INT                Training and probe batch size (default: 128)
  --hidden-dim INT                Model hidden dimension (default: 256)
  --lr FLOAT                      Learning rate (default: 1e-3)
  --save-every INT                Periodic checkpoint interval; use 1 for
                                  precise mid_best checkpoints (default: 1)

Landscape options:
  --checkpoint-kind KIND          start | mid_best | final | best_val | all
                                  (default: best_val)
  --train-subset-n INT            Landscape train subset size (default: 5000)
  --val-subset-n INT              Landscape val subset size (default: 5000)
  --max-pairs INT                 Pairwise interpolation pair cap (default: 100)
  --slice2d-max-runs INT          Runs per regime for 2D slices (default: 3)
  --hessian-max-batches INT       Max batches for Hessian estimates (default: 3)
  --hessian-power-iters INT       Hessian power iterations (default: 20)
  --hessian-trace-samples INT     Hutchinson trace samples (default: 20)
  --skip-expensive                Skip gradnorm/hessian/slice2d probe pass
  --overwrite-landscape           Recompute existing landscape artifacts

Pipeline control:
  --skip-train                    Do not train models
  --skip-landscape                Do not run landscape probes
  --skip-analysis                 Do not run analyze_landscapes.py
  --dry-run                       Print commands without running them
  --figures                       Let analyze_landscapes.py create figures
  --no-figures                    Skip analysis figures (default)
  -h, --help                      Show this help message

Environment:
  PYTHON                          Python executable (default: python)

Notes:
  Regime D is always trained anchor-augmented in this script.
  Regime B uses anchors automatically via train.py.
  Regime C is causal-only by default.

Example:
  scripts/run_big_pipeline.sh \
    --out-root /scratch/$USER/crl_landscapes \
    --device cuda \
    --seeds 20 \
    --epochs 100 \
    --checkpoint-kind best_val
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

seconds_now() {
  date +%s
}

format_duration() {
  local seconds="$1"
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  local secs=$((seconds % 60))
  printf "%02dh:%02dm:%02ds" "$hours" "$minutes" "$secs"
}

resolve_path() {
  "$PYTHON" - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

progress_bar() {
  local done_count="$1"
  local total_count="$2"
  local width=28
  local pct=0
  local filled=0
  if [[ "$total_count" -gt 0 ]]; then
    pct=$((100 * done_count / total_count))
    filled=$((width * done_count / total_count))
  fi
  local empty=$((width - filled))
  printf "["
  printf "%${filled}s" "" | tr " " "#"
  printf "%${empty}s" "" | tr " " "."
  printf "] %3d%% (%d/%d)" "$pct" "$done_count" "$total_count"
}

log() {
  echo "[$(timestamp)] $*"
}

run_cmd() {
  local label="$1"
  shift
  local start end elapsed
  log "START $label"
  printf '  '
  printf '%q ' "$@"
  echo

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN $label"
    return 0
  fi

  start="$(seconds_now)"
  "$@"
  end="$(seconds_now)"
  elapsed=$((end - start))
  log "DONE  $label elapsed=$(format_duration "$elapsed")"
}

run_cmd_logged() {
  local label="$1"
  local logfile="$2"
  shift 2
  local start end elapsed
  log "START $label"
  log "LOG   $logfile"
  printf '  '
  printf '%q ' "$@"
  echo

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN $label"
    return 0
  fi

  start="$(seconds_now)"
  if ! "$@" > "$logfile" 2>&1; then
    log "FAILED $label; last log lines:"
    tail -40 "$logfile" >&2 || true
    exit 1
  fi
  end="$(seconds_now)"
  elapsed=$((end - start))
  log "DONE  $label elapsed=$(format_duration "$elapsed")"
}

mark_step() {
  local label="$1"
  STEP_DONE=$((STEP_DONE + 1))
  printf "\n"
  progress_bar "$STEP_DONE" "$STEP_TOTAL"
  printf "  %s\n" "$label"
}

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"

OUT_ROOT="$REPO_ROOT"
DATA_DIR="$SRC_DIR/data/crl_dsprites"
DEVICE="cuda"
SEEDS=20
SEED_START=0
EPOCHS=100
BATCH_SIZE=128
HIDDEN_DIM=256
LR="1e-3"
SAVE_EVERY=1

CHECKPOINT_KIND="best_val"
TRAIN_SUBSET_N=5000
VAL_SUBSET_N=5000
MAX_PAIRS=100
SLICE2D_MAX_RUNS=3
HESSIAN_MAX_BATCHES=3
HESSIAN_POWER_ITERS=20
HESSIAN_TRACE_SAMPLES=20

SKIP_TRAIN=0
SKIP_LANDSCAPE=0
SKIP_ANALYSIS=0
SKIP_EXPENSIVE=0
OVERWRITE_LANDSCAPE=0
DRY_RUN=0
ANALYSIS_FIGURES="--no-figures"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      [[ $# -ge 2 ]] || die "--out-root requires a value"
      OUT_ROOT="$2"
      shift 2
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || die "--data-dir requires a value"
      DATA_DIR="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device requires a value"
      DEVICE="$2"
      shift 2
      ;;
    --seeds)
      [[ $# -ge 2 ]] || die "--seeds requires a value"
      SEEDS="$2"
      shift 2
      ;;
    --seed-start)
      [[ $# -ge 2 ]] || die "--seed-start requires a value"
      SEED_START="$2"
      shift 2
      ;;
    --epochs)
      [[ $# -ge 2 ]] || die "--epochs requires a value"
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      [[ $# -ge 2 ]] || die "--batch-size requires a value"
      BATCH_SIZE="$2"
      shift 2
      ;;
    --hidden-dim)
      [[ $# -ge 2 ]] || die "--hidden-dim requires a value"
      HIDDEN_DIM="$2"
      shift 2
      ;;
    --lr)
      [[ $# -ge 2 ]] || die "--lr requires a value"
      LR="$2"
      shift 2
      ;;
    --save-every)
      [[ $# -ge 2 ]] || die "--save-every requires a value"
      SAVE_EVERY="$2"
      shift 2
      ;;
    --checkpoint-kind)
      [[ $# -ge 2 ]] || die "--checkpoint-kind requires a value"
      CHECKPOINT_KIND="$2"
      shift 2
      ;;
    --train-subset-n)
      [[ $# -ge 2 ]] || die "--train-subset-n requires a value"
      TRAIN_SUBSET_N="$2"
      shift 2
      ;;
    --val-subset-n)
      [[ $# -ge 2 ]] || die "--val-subset-n requires a value"
      VAL_SUBSET_N="$2"
      shift 2
      ;;
    --max-pairs)
      [[ $# -ge 2 ]] || die "--max-pairs requires a value"
      MAX_PAIRS="$2"
      shift 2
      ;;
    --slice2d-max-runs)
      [[ $# -ge 2 ]] || die "--slice2d-max-runs requires a value"
      SLICE2D_MAX_RUNS="$2"
      shift 2
      ;;
    --hessian-max-batches)
      [[ $# -ge 2 ]] || die "--hessian-max-batches requires a value"
      HESSIAN_MAX_BATCHES="$2"
      shift 2
      ;;
    --hessian-power-iters)
      [[ $# -ge 2 ]] || die "--hessian-power-iters requires a value"
      HESSIAN_POWER_ITERS="$2"
      shift 2
      ;;
    --hessian-trace-samples)
      [[ $# -ge 2 ]] || die "--hessian-trace-samples requires a value"
      HESSIAN_TRACE_SAMPLES="$2"
      shift 2
      ;;
    --skip-expensive)
      SKIP_EXPENSIVE=1
      shift
      ;;
    --overwrite-landscape)
      OVERWRITE_LANDSCAPE=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-landscape)
      SKIP_LANDSCAPE=1
      shift
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --figures)
      ANALYSIS_FIGURES="--make-figures"
      shift
      ;;
    --no-figures)
      ANALYSIS_FIGURES="--no-figures"
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

[[ "$SEEDS" =~ ^[0-9]+$ ]] || die "--seeds must be an integer"
[[ "$SEED_START" =~ ^[0-9]+$ ]] || die "--seed-start must be an integer"
[[ "$EPOCHS" =~ ^[0-9]+$ ]] || die "--epochs must be an integer"
[[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || die "--batch-size must be an integer"
[[ "$SAVE_EVERY" =~ ^[0-9]+$ ]] || die "--save-every must be an integer"
[[ "$SEEDS" -ge 1 ]] || die "--seeds must be >= 1"
[[ "$EPOCHS" -ge 1 ]] || die "--epochs must be >= 1"
[[ "$SAVE_EVERY" -ge 1 ]] || die "--save-every must be >= 1"

case "$CHECKPOINT_KIND" in
  start|mid_best|final|best_val|all) ;;
  *) die "--checkpoint-kind must be start, mid_best, final, best_val, or all" ;;
esac

OUT_ROOT="$(resolve_path "$OUT_ROOT")"
DATA_DIR="$(resolve_path "$DATA_DIR")"

RUN_ROOT="$OUT_ROOT/runs"
LANDSCAPE_ROOT="$OUT_ROOT/landscape_runs"
ANALYSIS_ROOT="$OUT_ROOT/landscape_analysis"
LOG_ROOT="$OUT_ROOT/pipeline_logs"

mkdir -p "$RUN_ROOT" "$LANDSCAPE_ROOT" "$ANALYSIS_ROOT" "$LOG_ROOT"

declare -A DATASETS=(
  [regimeA]="observational.npz"
  [regimeB]="overlap_support.npz"
  [regimeC]="single_node_interventions.npz"
  [regimeD]="two_interventions_per_node.npz"
)

for regime in regimeA regimeB regimeC regimeD; do
  [[ -f "$DATA_DIR/${DATASETS[$regime]}" ]] || die "Missing dataset for $regime: $DATA_DIR/${DATASETS[$regime]}"
done

STEP_TOTAL=0
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  STEP_TOTAL=$((STEP_TOTAL + 4 * SEEDS))
fi
if [[ "$SKIP_LANDSCAPE" -eq 0 ]]; then
  STEP_TOTAL=$((STEP_TOTAL + 4))
  if [[ "$SKIP_EXPENSIVE" -eq 0 ]]; then
    STEP_TOTAL=$((STEP_TOTAL + 4))
  fi
fi
if [[ "$SKIP_ANALYSIS" -eq 0 ]]; then
  STEP_TOTAL=$((STEP_TOTAL + 1))
fi
STEP_DONE=0

PIPELINE_START="$(seconds_now)"

log "pipeline configuration"
cat <<EOF
  repo:                  $REPO_ROOT
  out_root:              $OUT_ROOT
  data_dir:              $DATA_DIR
  run_root:              $RUN_ROOT
  landscape_root:        $LANDSCAPE_ROOT
  analysis_root:         $ANALYSIS_ROOT
  logs:                  $LOG_ROOT
  python:                $PYTHON
  device:                $DEVICE
  seeds:                 $SEEDS
  seed_start:            $SEED_START
  epochs:                $EPOCHS
  batch_size:            $BATCH_SIZE
  hidden_dim:            $HIDDEN_DIM
  lr:                    $LR
  save_every:            $SAVE_EVERY
  checkpoint_kind:       $CHECKPOINT_KIND
  train_subset_n:        $TRAIN_SUBSET_N
  val_subset_n:          $VAL_SUBSET_N
  max_pairs:             $MAX_PAIRS
  slice2d_max_runs:      $SLICE2D_MAX_RUNS
  hessian_max_batches:   $HESSIAN_MAX_BATCHES
  dry_run:               $DRY_RUN
EOF

cd "$SRC_DIR"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  for regime in regimeA regimeB regimeC regimeD; do
    for ((offset = 0; offset < SEEDS; offset++)); do
      seed=$((SEED_START + offset))
      run_dir="$RUN_ROOT/$regime/seed_$seed"
      log_file="$LOG_ROOT/train_${regime}_seed_${seed}.log"

      if [[ -f "$run_dir/final.pt" && "$DRY_RUN" -eq 0 ]]; then
        log "SKIP train $regime seed=$seed; final.pt exists"
        mark_step "train $regime seed=$seed skipped"
        continue
      fi

      train_cmd=(
        "$PYTHON" train.py
        --data-path "$DATA_DIR/${DATASETS[$regime]}"
        --outdir "$run_dir"
        --seed "$seed"
        --epochs "$EPOCHS"
        --batch-size "$BATCH_SIZE"
        --hidden-dim "$HIDDEN_DIM"
        --lr "$LR"
        --device "$DEVICE"
        --eval-every 1
        --save-every "$SAVE_EVERY"
      )

      if [[ "$regime" == "regimeD" ]]; then
        train_cmd+=(--use-anchor-features)
      fi

      run_cmd_logged "train $regime seed=$seed" "$log_file" "${train_cmd[@]}"
      mark_step "train $regime seed=$seed"
    done
  done
fi

landscape_overwrite_args=()
if [[ "$OVERWRITE_LANDSCAPE" -eq 1 ]]; then
  landscape_overwrite_args+=(--overwrite)
fi

if [[ "$SKIP_LANDSCAPE" -eq 0 ]]; then
  for regime in regimeA regimeB regimeC regimeD; do
    log_file="$LOG_ROOT/landscape_broad_${regime}.log"
    broad_cmd=(
      "$PYTHON" landscape.py
      --regime "$regime"
      --run-root "$RUN_ROOT/$regime"
      --outdir "$LANDSCAPE_ROOT"
      --checkpoint-kind "$CHECKPOINT_KIND"
      --probes endpoint perturbation slice1d pairwise
      --device "$DEVICE"
      --batch-size "$BATCH_SIZE"
      --train-subset-n "$TRAIN_SUBSET_N"
      --val-subset-n "$VAL_SUBSET_N"
      --max-pairs "$MAX_PAIRS"
      "${landscape_overwrite_args[@]}"
    )
    run_cmd_logged "landscape broad $regime" "$log_file" "${broad_cmd[@]}"
    mark_step "landscape broad $regime"
  done

  if [[ "$SKIP_EXPENSIVE" -eq 0 ]]; then
    for regime in regimeA regimeB regimeC regimeD; do
      log_file="$LOG_ROOT/landscape_expensive_${regime}.log"
      expensive_cmd=(
        "$PYTHON" landscape.py
        --regime "$regime"
        --run-root "$RUN_ROOT/$regime"
        --outdir "$LANDSCAPE_ROOT"
        --checkpoint-kind "$CHECKPOINT_KIND"
        --probes gradnorm hessian slice2d
        --device "$DEVICE"
        --batch-size "$BATCH_SIZE"
        --train-subset-n "$TRAIN_SUBSET_N"
        --val-subset-n "$VAL_SUBSET_N"
        --hessian-max-batches "$HESSIAN_MAX_BATCHES"
        --hessian-power-iters "$HESSIAN_POWER_ITERS"
        --hessian-trace-samples "$HESSIAN_TRACE_SAMPLES"
        --slice2d-max-runs "$SLICE2D_MAX_RUNS"
        "${landscape_overwrite_args[@]}"
      )
      run_cmd_logged "landscape expensive $regime" "$log_file" "${expensive_cmd[@]}"
      mark_step "landscape expensive $regime"
    done
  fi
fi

if [[ "$SKIP_ANALYSIS" -eq 0 ]]; then
  log_file="$LOG_ROOT/analyze_landscapes.log"
  analysis_cmd=(
    "$PYTHON" analyze_landscapes.py
    --landscape-root "$LANDSCAPE_ROOT"
    --outdir "$ANALYSIS_ROOT"
    --checkpoint-kind "$CHECKPOINT_KIND"
    --splits val
    --bootstrap-samples 2000
    "$ANALYSIS_FIGURES"
  )
  run_cmd_logged "analyze landscapes" "$log_file" "${analysis_cmd[@]}"
  mark_step "analyze landscapes"
fi

PIPELINE_END="$(seconds_now)"
PIPELINE_ELAPSED=$((PIPELINE_END - PIPELINE_START))

echo
log "pipeline complete elapsed=$(format_duration "$PIPELINE_ELAPSED")"
echo "Outputs:"
echo "  runs:       $RUN_ROOT"
echo "  probes:     $LANDSCAPE_ROOT"
echo "  analysis:   $ANALYSIS_ROOT"
echo "  logs:       $LOG_ROOT"
