#!/bin/bash
# ============================================================
# TimeChat-7B inference
# GitHub: https://github.com/RenShuhuai-Andy/TimeChat
# NOTE: Must run from the TimeChat repo directory (needs eval_configs/)
# ============================================================

# --- Configure these paths before first run ---
CONDA_PATH="$(conda info --base)/etc/profile.d/conda.sh"
REPO_DIR=""       # e.g. /path/to/TimeChat
MODEL_DIR=""      # e.g. /path/to/TimeChat-7b
CSV_PATH=""       # e.g. /path/to/labeled_full.csv
VIDEO_ROOT=""     # e.g. /path/to/videos
# -----------------------------------------------

source "$CONDA_PATH"
conda activate timechat

export CUDA_VISIBLE_DEVICES=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

python -u "$SCRIPT_DIR/infer_timechat.py" \
    --repo-dir "$REPO_DIR" \
    --model-dir "$MODEL_DIR" \
    --csv-path "$CSV_PATH" \
    --video-root "$VIDEO_ROOT" \
    --output-dir "$SCRIPT_DIR/output/timechat"
