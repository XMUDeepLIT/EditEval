#!/bin/bash
# ============================================================
# LLaVA-NeXT-Video-34B inference
# GitHub: https://github.com/LLaVA-VL/LLaVA-NeXT
# ============================================================

# --- Configure these paths before first run ---
CONDA_PATH="$(conda info --base)/etc/profile.d/conda.sh"
REPO_DIR=""       # e.g. /path/to/LLaVA-NeXT
MODEL_PATH=""     # e.g. /path/to/LLaVA-NeXT-Video-34B
CSV_PATH=""       # e.g. /path/to/labeled_full.csv
VIDEO_ROOT=""     # e.g. /path/to/videos
# -----------------------------------------------

source "$CONDA_PATH"
conda activate llava

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python -u "$SCRIPT_DIR/infer_llava_next.py" \
    --repo-dir "$REPO_DIR" \
    --model-path "$MODEL_PATH" \
    --conv-mode vicuna_v1 \
    --csv-path "$CSV_PATH" \
    --video-root "$VIDEO_ROOT" \
    --num-frames 32 \
    --mm-spatial-pool-stride 2 \
    --mm-spatial-pool-mode average \
    --mm-newline-position after \
    --output-dir "$SCRIPT_DIR/output/llava_next_34b"
