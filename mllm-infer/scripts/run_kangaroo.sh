#!/bin/bash
# ============================================================
# Kangaroo inference
# GitHub: https://github.com/KangarooGroup/Kangaroo
# ============================================================

# --- Configure these paths before first run ---
CONDA_PATH="$(conda info --base)/etc/profile.d/conda.sh"
MODEL_PATH=""     # e.g. /path/to/kangaroo
CSV_PATH=""       # e.g. /path/to/labeled_full.csv
VIDEO_ROOT=""     # e.g. /path/to/videos
# -----------------------------------------------

source "$CONDA_PATH"
conda activate kangaroo

export CUDA_VISIBLE_DEVICES=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python -u "$SCRIPT_DIR/infer_kangaroo.py" \
    --model-path "$MODEL_PATH" \
    --csv-path "$CSV_PATH" \
    --video-root "$VIDEO_ROOT" \
    --output-dir "$SCRIPT_DIR/output/kangaroo"
