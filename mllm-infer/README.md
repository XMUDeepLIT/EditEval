# Video Editing Quality Evaluation — MLLM Inference

Evaluate video editing quality using various open-source Multimodal Large Language Models (MLLMs). Each model scores edited videos on a 1–5 scale across three dimensions:

| Dimension | Description |
|-----------|-------------|
| **Textual Faithfulness** | How well the edited video aligns with the editing text description |
| **Frame Consistency** | Continuity between adjacent frames |
| **Video Fidelity** | Realism of the video, including color accuracy and overall visual quality |

## Directory Structure

```
opensource/
├── common.py                  # Shared utilities: criteria, prompt template, data loading, result saving
├── infer_vila.py              # VILA-1.5 inference
├── infer_qwen_vl.py           # Qwen-VL-Chat inference (image input, 4 sampled frames)
├── infer_video_llama2.py      # VideoLLaMA2 inference (supports multi-GPU parallel)
├── infer_timechat.py          # TimeChat inference
├── infer_llava_ov.py          # LLaVA-OneVision inference
├── infer_kangaroo.py          # Kangaroo inference
├── infer_llava_next.py        # LLaVA-NeXT-Video inference (unified entry for 7B/32B/34B)
├── README.md
└── scripts/
    ├── run_vila.sh
    ├── run_qwen_vl.sh
    ├── run_video_llama2.sh
    ├── run_video_llama2_multi.sh   # Multi-GPU parallel version
    ├── run_timechat.sh
    ├── run_llava_ov.sh
    ├── run_kangaroo.sh
    ├── run_llava_next_7b.sh
    ├── run_llava_next_32b.sh
    └── run_llava_next_34b.sh
```

## Supported Models

| Model | Conda Env | GitHub Repository | Launch Script |
|-------|-----------|-------------------|---------------|
| VILA-1.5-40B | `vila` | [NVlabs/VILA](https://github.com/NVlabs/VILA) | `run_vila.sh` |
| Qwen-VL-Chat | `qwen` | [QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL) | `run_qwen_vl.sh` |
| VideoLLaMA2-7B | `llama2` | [DAMO-NLP-SG/VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) | `run_video_llama2.sh` |
| TimeChat-7B | `timechat` | [RenShuhuai-Andy/TimeChat](https://github.com/RenShuhuai-Andy/TimeChat) | `run_timechat.sh` |
| LLaVA-OneVision-7B | `llava` | [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) | `run_llava_ov.sh` |
| Kangaroo | `kangaroo` | [KangarooGroup/Kangaroo](https://github.com/KangarooGroup/Kangaroo) | `run_kangaroo.sh` |
| LLaVA-NeXT-Video-7B | `llava` | [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) | `run_llava_next_7b.sh` |
| LLaVA-NeXT-Video-32B | `llava` | [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) | `run_llava_next_32b.sh` |
| LLaVA-NeXT-Video-34B | `llava` | [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) | `run_llava_next_34b.sh` |

## Environment Setup

Each model uses an isolated conda environment. Follow the README of each GitHub repository to install dependencies:

```bash
# Example: VILA
git clone https://github.com/NVlabs/VILA.git
conda create -n vila python=3.10
conda activate vila
cd VILA && pip install -e .   # Follow VILA's README for installation

# Example: VideoLLaMA2
git clone https://github.com/DAMO-NLP-SG/VideoLLaMA2.git
conda create -n llama2 python=3.10
conda activate llama2
cd VideoLLaMA2 && pip install -e .

# Other models follow the same pattern — refer to their respective READMEs.
```

Download model weights from HuggingFace to a local directory.

## Quick Start

### Option 1: Shell Scripts

1. Edit `scripts/run_<model>.sh` and fill in the path variables at the top:

```bash
REPO_DIR="/path/to/cloned/repo"      # Path to the cloned repository
MODEL_PATH="/path/to/model/weights"   # Path to model weights
CSV_PATH="/path/to/labeled_full.csv"  # Evaluation data CSV
VIDEO_ROOT="/path/to/videos"          # Directory containing video files
```

2. Run:

```bash
bash scripts/run_kangaroo.sh          # Default: GPU 0
bash scripts/run_kangaroo.sh 1        # Use GPU 1
```

### Option 2: Run Python Directly

```bash
conda activate kangaroo
CUDA_VISIBLE_DEVICES=0 python infer_kangaroo.py \
    --model-path /path/to/kangaroo \
    --csv-path /path/to/labeled_full.csv \
    --video-root /path/to/videos \
    --output-dir output/kangaroo
```

### VideoLLaMA2 Multi-GPU Parallel

```bash
# After configuring path variables in scripts/run_video_llama2_multi.sh:
bash scripts/run_video_llama2_multi.sh              # Default: 1280 samples, 8 workers
bash scripts/run_video_llama2_multi.sh 1280 "0 0 1 1 2 2 3 3"  # Custom GPU layout
```

## Data Format

### Input CSV

The CSV file must contain the following columns:

| Column | Description |
|--------|-------------|
| `caption` | Original video description |
| `editing_prompt` | Editing text condition |
| `editing_entity` | Entity involved in the edit |
| `key` | Sample identifier |
| `编辑后视频` | Edited video identifier (filename is extracted from this column) |
| `frames` | Number of video frames |

### Output Format

Each inference run produces the following structure:

```
output/<model>/
├── single/
│   ├── 000000.json
│   ├── 000001.json
│   └── ...
└── full.json
```

Each JSON file contains scores and reasoning for three dimensions:

```json
{
    "Textual_Faithfulness": "The score is 4. Reason: ...",
    "Frame_Consistency": "The score is 3. Reason: ...",
    "Video_Fidelity": "The score is 4. Reason: ..."
}
```
