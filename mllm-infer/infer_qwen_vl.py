"""
Qwen-VL-Chat inference for video editing quality evaluation.

Setup:
  1. Clone https://github.com/QwenLM/Qwen-VL and install per its README
  2. conda activate qwen
  3. Download Qwen-VL-Chat from HuggingFace

Note: Qwen-VL only supports image input. Videos are pre-extracted into frames,
      then 4 frames are evenly sampled per video.

Usage:
  python infer_qwen_vl.py --model-path /path/to/Qwen-VL-Chat \
      --csv-path data.csv --video-root /path/to/videos --output-dir output/qwen_vl
"""

import argparse
import glob
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import EvalSample, load_eval_dataset, run_evaluation

torch.manual_seed(1234)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL video evaluation inference")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Qwen-VL-Chat model")
    parser.add_argument("--output-dir", type=str, default="output/qwen_vl")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to evaluation CSV file")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--num-frames", type=int, default=4,
                        help="Number of frames to sample from each video")
    return parser.parse_args()


def evenly_sample_files(file_list, num_samples=4):
    n = len(file_list)
    if n < num_samples:
        raise ValueError(f"Need at least {num_samples} files, got {n}")
    step = n // num_samples
    return [file_list[i * step] for i in range(num_samples)]


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="cuda", trust_remote_code=True
    ).eval()

    def infer_fn(prompt: str, sample: EvalSample) -> str:
        frame_dir = sample.video_path.replace(".mp4", "")
        frame_files = sorted(glob.glob(f"{frame_dir}/*"))
        frame_files = evenly_sample_files(frame_files, args.num_frames)

        query_parts = [{"image": f} for f in frame_files]
        query_parts.append({"text": prompt})
        query = tokenizer.from_list_format(query_parts)

        response, _ = model.chat(tokenizer, query=query, history=None)
        return response

    samples = load_eval_dataset(args.csv_path, args.video_root)
    run_evaluation(samples, infer_fn, args.output_dir)


if __name__ == "__main__":
    main()
