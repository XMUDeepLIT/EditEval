"""
Kangaroo inference for video editing quality evaluation.

Setup:
  1. Clone https://github.com/KangarooGroup/Kangaroo and install per its README
  2. conda activate kangaroo
  3. Download Kangaroo model from HuggingFace

Usage:
  python infer_kangaroo.py --model-path /path/to/kangaroo \
      --csv-path data.csv --video-root /path/to/videos --output-dir output/kangaroo
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import EvalSample, load_eval_dataset, run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Kangaroo video evaluation inference")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Kangaroo model")
    parser.add_argument("--output-dir", type=str, default="output/kangaroo")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to evaluation CSV file")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    def infer_fn(prompt: str, sample: EvalSample) -> str:
        output, _ = model.chat(
            video_path=sample.video_path,
            query=prompt,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        return output

    samples = load_eval_dataset(args.csv_path, args.video_root)
    run_evaluation(samples, infer_fn, args.output_dir)


if __name__ == "__main__":
    main()
