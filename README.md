# EditEval

EditEval is a benchmark for evaluating text-guided video editing methods. We provide data and scripts including:

1. **200 source videos** and **1,280 edited videos** produced by 8 video editing models
2. **1,010 editing text prompts** covering 8 task categories
3. **Human annotations** from 4 annotators across 3 evaluation dimensions, along with inter-annotator agreement computation
4. **MLLM inference outputs** from 8 multimodal large language models, with correlation analysis against human annotations

## Directory Structure

```
EditEval/
├── source_video/
│   ├── download.sh                  # Download 200 source videos from Google Drive
│   └── *.mp4                        # Source video files (after download)
├── edited_video/
│   ├── download.sh                  # Download 1,280 edited videos from Google Drive
│   └── *.mp4                        # Edited video files (after download)
├── mllm_results/
│   ├── download.sh                  # Download MLLM inference results from Google Drive
│   ├── 4o/                          # GPT-4o
│   ├── 4o_0806/                     # GPT-4o-0806
│   ├── gemini/                      # Gemini-Pro
│   ├── one_vision_7b/               # LLaVA-OneVision-7B
│   ├── qwen_vl/                     # Qwen-VL-Chat
│   ├── timechat/                    # TimeChat
│   ├── videollama2/                 # VideoLLaMA2
│   └── vila/                        # VILA-1.5-40B
├── mllm-infer/
│   ├── common.py                    # Shared utilities: criteria, prompt template, data loading
│   ├── infer_vila.py                # VILA-1.5 inference
│   ├── infer_qwen_vl.py            # Qwen-VL-Chat inference (image input, 4 sampled frames)
│   ├── infer_video_llama2.py        # VideoLLaMA2 inference (supports multi-GPU parallel)
│   ├── infer_timechat.py            # TimeChat inference
│   ├── infer_llava_ov.py           # LLaVA-OneVision inference
│   ├── infer_kangaroo.py            # Kangaroo inference
│   ├── infer_llava_next.py          # LLaVA-NeXT-Video inference (7B/32B/34B)
│   ├── README.md
│   └── scripts/                     # Shell scripts for each model
├── annotations/
│   ├── data_worker_1.csv            # Annotation results from annotator 1
│   ├── data_worker_2.csv            # Annotation results from annotator 2
│   ├── data_worker_3.csv            # Annotation results from annotator 3
│   └── data_worker_4.csv            # Annotation results from annotator 4
├── labeled_full.csv                 # Aggregated human annotation scores (1,280 samples)
├── edit_eval_text_prompts.csv       # 1,010 editing text prompts with task metadata
├── inter-annotator-agreement.py     # Compute inter-annotator agreement
├── compute_correlation.py           # Compute MLLM-human score correlations
└── README.md
```

## Video Editing Models (Edited Videos)

The 1,280 edited videos are produced by the following **8 video editing models**, each contributing 160 samples:

| Model | Samples |
|---|---|
| FateZero | 160 |
| RAVE | 160 |
| Text2Video-Zero | 160 |
| TokenFlow | 160 |
| Tune-A-Video | 160 |
| Vidtome | 160 |
| pix2video | 160 |
| vid2vid-zero | 160 |

## Benchmark Statistics

### Video Statistics

| Statistic | Number |
|---|---|
| Total video clips | 200 |
| Video resolution | 480 × 480 |
| Video length | 25 frames |

### Text Prompts Statistics

| Statistic | Number |
|---|---|
| **Total text prompts** | **1,010** |
| **Single-Target Editing** | **706** |
| &nbsp;&nbsp;&nbsp;&nbsp;Animal Editing | 56 |
| &nbsp;&nbsp;&nbsp;&nbsp;Human Editing | 107 |
| &nbsp;&nbsp;&nbsp;&nbsp;Object Editing | 96 |
| &nbsp;&nbsp;&nbsp;&nbsp;Background Editing | 143 |
| &nbsp;&nbsp;&nbsp;&nbsp;Overall Style Editing | 152 |
| &nbsp;&nbsp;&nbsp;&nbsp;Color Transfer | 152 |
| **Multiple-Target Editing** | **304** |

### Text Prompts Statistics for Meta Evaluation

| Statistic | Number |
|---|---|
| **Total text prompts** | **160** |
| **Single-Target Editing** | **96** |
| &nbsp;&nbsp;&nbsp;&nbsp;Animal Editing | 12 |
| &nbsp;&nbsp;&nbsp;&nbsp;Human Editing | 12 |
| &nbsp;&nbsp;&nbsp;&nbsp;Object Editing | 12 |
| &nbsp;&nbsp;&nbsp;&nbsp;Background Editing | 12 |
| &nbsp;&nbsp;&nbsp;&nbsp;Overall Style Editing | 24 |
| &nbsp;&nbsp;&nbsp;&nbsp;Color Transfer | 24 |
| **Multiple-Target Editing** | **64** |
| Maximum editing target | 5 |
| Minimum editing target | 1 |
| Average editing target | 1.8 |
| Maximum caption length | 29 |
| Minimum caption length | 6 |
| Average caption length | 14.4 |
| Maximum text prompt length | 29 |
| Minimum text prompt length | 7 |
| Average text prompt length | 16.5 |

## Evaluation Dimensions

Each sample is evaluated on **3 dimensions** (scored 1–5 by human annotators):

- **Textual Faithfulness**: How well the edited video aligns with the editing text prompt
- **Frame Consistency**: Temporal coherence and smoothness across frames
- **Video Fidelity**: Overall visual quality and realism of the edited video

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install gdown pandas scipy prettytable krippendorff numpy
```

### 2. Download Source Videos

Download the 200 source videos from Google Drive:

```bash
cd source_video
bash download.sh
cd ..
```

### 3. Download Edited Videos

Download the 1,280 edited videos from Google Drive:

```bash
cd edited_video
bash download.sh
cd ..
```

### 4. Download MLLM Results

Download the MLLM inference outputs (8 models × 1,280 JSON files) from Google Drive:

```bash
cd mllm_results
bash download.sh
cd ..
```

## MLLM Evaluation Models

The `mllm_results/` directory contains inference outputs from 8 multimodal large language models. Each model directory contains 1,280 JSON files (one per sample), with scores for the 3 evaluation dimensions.

| Directory | Model |
|---|---|
| `4o/` | GPT-4o |
| `4o_0806/` | GPT-4o-0806 |
| `gemini/` | Gemini-Pro |
| `one_vision_7b/` | LLaVA-OneVision-7B |
| `qwen_vl/` | Qwen-VL-Chat |
| `timechat/` | TimeChat |
| `videollama2/` | VideoLLaMA2 |
| `vila/` | VILA-1.5-40B |

## MLLM Inference Code

The `mllm-infer/` directory provides inference scripts for reproducing MLLM evaluation results using open-source models. Each model scores edited videos on a 1–5 scale across three dimensions: **Textual Faithfulness**, **Frame Consistency**, and **Video Fidelity**.

### Supported Models

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

### Running Inference

Each model uses an isolated conda environment. Clone the corresponding GitHub repository and install dependencies following its README, then download model weights from HuggingFace.

**Option 1: Shell Scripts**

Edit `mllm-infer/scripts/run_<model>.sh` and fill in the path variables, then run:

```bash
cd mllm-infer
bash scripts/run_kangaroo.sh          # Default: GPU 0
bash scripts/run_kangaroo.sh 1        # Use GPU 1
```

**Option 2: Run Python Directly**

```bash
conda activate kangaroo
CUDA_VISIBLE_DEVICES=0 python mllm-infer/infer_kangaroo.py \
    --model-path /path/to/kangaroo \
    --csv-path labeled_full.csv \
    --video-root /path/to/videos \
    --output-dir output/kangaroo
```

For more details, see [`mllm-infer/README.md`](mllm-infer/README.md).

## Human Annotation & Inter-Annotator Agreement

The `annotations/` directory contains per-annotator scores (with personal information removed). The aggregated scores (averaged over 4 annotators) are in `labeled_full.csv`.

Compute inter-annotator agreement:

```bash
python inter-annotator-agreement.py
```

Expected output:

```
----------------------------------------
Metrics for Textual Faithfulness:
Averaged Kendall's τc: 0.6408 ± 0.0728
Averaged Spearman’s ρ: 0.7095 ± 0.0810
Krippendorff’s α: 0.6995
----------------------------------------
Metrics for Frame Consistency:
Averaged Kendall's τc: 0.6505 ± 0.0226
Averaged Spearman’s ρ: 0.7293 ± 0.0244
Krippendorff’s α: 0.6688
----------------------------------------
Metrics for Video Fidelity:
Averaged Kendall's τc: 0.6126 ± 0.0295
Averaged Spearman’s ρ: 0.6935 ± 0.0299
Krippendorff’s α: 0.6628
```

## MLLM Score Correlation with Human Annotations

Compute Pearson, Spearman, and Kendall-τ correlations between MLLM-generated scores and human annotations:

**Run on a single model:**

```bash
python compute_correlation.py --labeled_csv labeled_full.csv --mllm_dir mllm_results/vila
```

**Run on multiple models:**

```bash
python compute_correlation.py --labeled_csv labeled_full.csv --mllm_dir mllm_results/vila mllm_results/videollama2
```

**Run on all models:**

```bash
python compute_correlation.py --labeled_csv labeled_full.csv --mllm_dir all
```

## Citation

If you find EditEval useful in your research, please cite our paper:

```bibtex
@inproceedings{liu2025editeval,
  author    = {Bingshuai Liu and Ante Wang and Zijun Min and Chenyang Lyu and 
               Longyue Wang and Zhihao Wang and Xu Han and Peng Li and Jinsong Su},
  title     = {EditEval: Towards Comprehensive and Automatic Evaluation for Text-guided Video Editing},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)},
  year      = {2025},
  pages     = {3507--3516},
  publisher = {ACM},
  doi       = {10.1145/3746027.3755100},
  url       = {https://doi.org/10.1145/3746027.3755100}
}
```
