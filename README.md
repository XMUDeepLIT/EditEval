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
Averaged Kendall's τc: 0.6378 ± 0.0749
Averaged Spearman's ρ: 0.7058 ± 0.0833
Krippendorff's α: 0.6960
----------------------------------------
Metrics for Frame Consistency:
Averaged Kendall's τc: 0.6510 ± 0.0207
Averaged Spearman's ρ: 0.7298 ± 0.0225
Krippendorff's α: 0.6687
----------------------------------------
Metrics for Video Fidelity:
Averaged Kendall's τc: 0.6125 ± 0.0297
Averaged Spearman's ρ: 0.6938 ± 0.0299
Krippendorff's α: 0.6625
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
