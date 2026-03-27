#!/bin/bash
# ============================================================
# LLaVA-OneVision-7B inference
# GitHub: https://github.com/LLaVA-VL/LLaVA-NeXT
# ============================================================

# --- Configure these paths before first run ---
CONDA_PATH="$(conda info --base)/etc/profile.d/conda.sh"
REPO_DIR=""       # e.g. /path/to/LLaVA-NeXT
MODEL_PATH=""     # e.g. /path/to/llava-onevision-qwen2-7b-ov
CSV_PATH=""       # e.g. /path/to/labeled_full.csv
VIDEO_ROOT=""     # e.g. /path/to/videos
# -----------------------------------------------

source "$CONDA_PATH"
conda activate llava

export CUDA_VISIBLE_DEVICES=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python -u "$SCRIPT_DIR/infer_llava_ov.py" \
    --repo-dir "$REPO_DIR" \
    --model-path "$MODEL_PATH" \
    --csv-path "$CSV_PATH" \
    --video-root "$VIDEO_ROOT" \
    --output-dir "$SCRIPT_DIR/output/llava_onevision"
