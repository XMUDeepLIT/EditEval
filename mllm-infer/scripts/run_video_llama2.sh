#!/bin/bash
# ============================================================
# VideoLLaMA2-7B-16F inference
# GitHub: https://github.com/DAMO-NLP-SG/VideoLLaMA2
# ============================================================

# --- Configure these paths before first run ---
CONDA_PATH="$(conda info --base)/etc/profile.d/conda.sh"
REPO_DIR=""       # e.g. /path/to/VideoLLaMA2
MODEL_PATH=""     # e.g. /path/to/VideoLLaMA2-7B-16F
CSV_PATH=""       # e.g. /path/to/labeled_full.csv
VIDEO_ROOT=""     # e.g. /path/to/videos
# -----------------------------------------------

source "$CONDA_PATH"
conda activate llama2

export CUDA_VISIBLE_DEVICES=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python -u "$SCRIPT_DIR/infer_video_llama2.py" \
    --repo-dir "$REPO_DIR" \
    --model-path "$MODEL_PATH" \
    --csv-path "$CSV_PATH" \
    --video-root "$VIDEO_ROOT" \
    --output-dir "$SCRIPT_DIR/output/video_llama2"
