"""
Shared utilities for video editing quality evaluation using MLLMs.

All model-specific inference scripts import from this module to avoid
code duplication for: evaluation criteria, prompt templates, data loading,
and result persistence.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Evaluation criteria (1–5 scale)
# ---------------------------------------------------------------------------

CRITERIA = {
    "Textual_Faithfulness": (
        "Textual Faithfulness\n"
        "This measures the degree to which the edited video aligns with the text description provided for editing.\n"
        "1: The edited video completely misaligns with the text description.\n"
        "2: The edited video mostly misaligns with the text description.\n"
        "3: The edited video generally aligns with the text description, but many details are missing.\n"
        "4: The edited video aligns with the text description in most aspects, with only a few details not reflected.\n"
        "5: The edited video fully aligns with the text description, capturing all details accurately."
    ),
    "Frame_Consistency": (
        "Frame Consistency\n"
        "This assesses the continuity between adjacent frames in the edited video.\n"
        "1: There is no continuity between frames, resulting in a poor viewing experience.\n"
        "2: The continuity between frames is poor, with noticeable jumps.\n"
        "3: The continuity between frames is average, with minor jumps in some scenes.\n"
        "4: The continuity between frames is good, with only minimal jumps in a very few scenes.\n"
        "5: The frames flow smoothly and continuously without any noticeable jumps."
    ),
    "Video_Fidelity": (
        "Video Fidelity\n"
        "This evaluates the realism of the edited video, including factors such as color accuracy, "
        "overall visual quality, and viewer experience.\n"
        "1: The video suffers from severe color distortion, poor visual quality, and weak overall presentation, "
        "leading to a very poor viewing experience.\n"
        "2: The video has significant color distortion and overall visual quality issues, with noticeable inconsistencies.\n"
        "3: The video has slight color distortion and is generally acceptable, but some unnatural elements are still noticeable.\n"
        "4: The video is close to realistic, with good overall quality and only minor imperfections in rare instances.\n"
        "5: The video is fully realistic, with excellent visual quality and no noticeable flaws, providing a perfect viewing experience."
    ),
}

RESULT_KEYS = list(CRITERIA.keys())

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "You are given a video that has been edited by a video editing model, "
    "alongside its corresponding text condition and the description of the original video. "
    "Your task is to watch the video and evaluate it on a scale from 1 to 5 according to "
    "the scoring criteria provided below. After generating the score, provide a brief "
    'explanation of your reasoning. Answer in the format:\n\n'
    '"The score is {{generated_score}}. Reason: {{explanation of why this score was given}}."\n\n'
    "Criteria:\n{criteria}\n\n"
    "Origin Video:\n{org_caption}\n\n"
    "Video editing text condition:\n{text_prompt}"
)


def build_prompt(criteria_key: str, org_caption: str, text_prompt: str) -> str:
    """Build an evaluation prompt for a given criterion."""
    return PROMPT_TEMPLATE.format(
        criteria=CRITERIA[criteria_key],
        org_caption=org_caption,
        text_prompt=text_prompt,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    index: int
    video_path: str
    caption: str
    editing_prompt: str
    editing_entity: str
    key: str
    num_frames: Optional[int] = None


def load_eval_dataset(csv_path: str, video_root: str) -> List[EvalSample]:
    """Load the benchmark CSV and return a list of EvalSample objects."""
    df = pd.read_csv(csv_path)
    samples = []
    for i, row in df.iterrows():
        video_name = str(row["编辑后视频"]).split("/")[-1]
        video_path = os.path.join(video_root, f"{video_name}.mp4")
        num_frames = int(row["frames"]) if "frames" in df.columns else None
        samples.append(EvalSample(
            index=i,
            video_path=video_path,
            caption=str(row["caption"]),
            editing_prompt=str(row["editing_prompt"]),
            editing_entity=str(row["editing_entity"]),
            key=str(row["key"]),
            num_frames=num_frames,
        ))
    return samples


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_single_result(result: Dict[str, str], output_dir: str, index: int):
    """Save a single sample's evaluation result as JSON."""
    single_dir = os.path.join(output_dir, "single")
    os.makedirs(single_dir, exist_ok=True)
    path = os.path.join(single_dir, f"{str(index).zfill(6)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


def save_full_results(results: List[Dict[str, str]], output_dir: str):
    """Save all evaluation results as a single JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "full.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def run_evaluation(
    samples: List[EvalSample],
    infer_fn,
    output_dir: str,
    criteria_keys: Optional[List[str]] = None,
):
    """
    Generic evaluation loop.

    Parameters
    ----------
    samples : list of EvalSample
    infer_fn : callable(prompt: str, sample: EvalSample) -> str
        Model-specific function that takes a prompt string and an EvalSample,
        returns the model's text output.
    output_dir : str
    criteria_keys : optional subset of RESULT_KEYS to evaluate
    """
    if criteria_keys is None:
        criteria_keys = RESULT_KEYS

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for sample in tqdm(samples, desc="Evaluating"):
        cur_res = {}
        for key in criteria_keys:
            prompt = build_prompt(key, sample.caption, sample.editing_prompt)
            try:
                output = infer_fn(prompt, sample)
                cur_res[key] = output
            except Exception as e:
                print(f"[WARN] sample {sample.index}, criterion {key}: {e}")
                cur_res[key] = f"ERROR: {e}"

        save_single_result(cur_res, output_dir, sample.index)
        results.append(cur_res)

    save_full_results(results, output_dir)
    print(f"Results saved to {output_dir}")
    return results
