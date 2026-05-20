#!/usr/bin/env bash
# run.sh — entry point for the AdoDAS-irt pipeline.
#
# Usage: ./run.sh <command> [extra-args...]
#   setup       Install dependencies via uv into ./.venv
#   smoke       End-to-end smoke test on synthetic data (no real features)
#   test        Run pytest unit tests
#   features    Stage 01 — build pooled wide tables + PCA + diffs + views
#   folds       Stage 02 — subject-disjoint stratified 5-fold split
#   trees       Stage 03 — tree-expert OOF (LGBM/XGB/HGBT/ExtraTrees)
#   irt         Stage 04 — IRT joint-head OOF
#   meta        Stage 06 — meta-blend + threshold optimisation + A1 calibration
#   refit       Stage 07 — refit-all-labeled + test prediction + submission CSV
#   pipeline    Full chain: features → folds → trees → irt → meta → refit
#   help        Show this help
#
# Config overrides (env vars; defaults in parentheses):
#   PATHS_CFG     (configs/paths.yaml)
#   FEATURES_CFG  (configs/features.yaml)
#   TREES_CFG     (configs/trees.yaml)
#   IRT_CFG       (configs/irt.yaml)
#   STACK_CFG     (configs/stack.yaml)
#   PYTHON        (auto: uses .venv/bin/python if present, else python3)

set -euo pipefail
cd "$(dirname "$0")"

# ---- defaults -------------------------------------------------------------
PATHS_CFG="${PATHS_CFG:-configs/paths.yaml}"
FEATURES_CFG="${FEATURES_CFG:-configs/features.yaml}"
TREES_CFG="${TREES_CFG:-configs/trees.yaml}"
IRT_CFG="${IRT_CFG:-configs/irt.yaml}"
STACK_CFG="${STACK_CFG:-configs/stack.yaml}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

log() { printf '\033[1;36m[run.sh]\033[0m %s\n' "$*" >&2; }

# ---- commands -------------------------------------------------------------
cmd_setup() {
  if ! command -v uv >/dev/null 2>&1; then
    log "uv not found. Install:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
  uv venv
  uv pip install -e ".[dev]"
  log "ready. activate with:  source .venv/bin/activate"
  log "or just keep using run.sh — it auto-detects .venv/bin/python"
}

cmd_smoke() {
  "$PYTHON" scripts/99_smoke_test.py "$@"
}

cmd_test() {
  "$PYTHON" -m pytest tests/ -v "$@"
}

cmd_features() {
  "$PYTHON" scripts/01_build_features.py \
    --paths-config "$PATHS_CFG" \
    --features-config "$FEATURES_CFG" "$@"
}

cmd_folds() {
  "$PYTHON" scripts/02_make_folds.py --paths-config "$PATHS_CFG" "$@"
}

cmd_trees() {
  "$PYTHON" scripts/03_train_trees_oof.py \
    --paths-config "$PATHS_CFG" --trees-config "$TREES_CFG" "$@"
}

cmd_irt() {
  "$PYTHON" scripts/04_train_irt_oof.py \
    --paths-config "$PATHS_CFG" --irt-config "$IRT_CFG" "$@"
}

cmd_meta() {
  "$PYTHON" scripts/06_meta_blend.py \
    --paths-config "$PATHS_CFG" --stack-config "$STACK_CFG" "$@"
}

cmd_refit() {
  "$PYTHON" scripts/07_refit_all_and_predict.py \
    --paths-config "$PATHS_CFG" \
    --trees-config "$TREES_CFG" \
    --irt-config "$IRT_CFG" \
    --stack-config "$STACK_CFG" "$@"
}

cmd_pipeline() {
  log "stage 01  build_features";  cmd_features
  log "stage 02  make_folds";      cmd_folds
  log "stage 03  train_trees_oof"; cmd_trees
  log "stage 04  train_irt_oof";   cmd_irt
  log "stage 06  meta_blend";      cmd_meta
  log "stage 07  refit + predict"; cmd_refit
  log "pipeline done — see output/submission/"
}

usage() {
  sed -n '2,28p' "$0"
}

main() {
  if [ $# -eq 0 ]; then usage; exit 1; fi
  local cmd="$1"; shift
  case "$cmd" in
    setup|smoke|test|features|folds|trees|irt|meta|refit|pipeline)
      "cmd_$cmd" "$@" ;;
    help|-h|--help) usage ;;
    *) log "unknown command: $cmd"; usage; exit 1 ;;
  esac
}

main "$@"
