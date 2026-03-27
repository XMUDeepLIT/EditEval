"""
LLaVA-OneVision inference for video editing quality evaluation.

Setup:
  1. Clone https://github.com/LLaVA-VL/LLaVA-NeXT and install per its README
  2. conda activate llava
  3. Download llava-onevision-qwen2-7b-ov from HuggingFace

Usage:
  python infer_llava_ov.py --repo-dir /path/to/LLaVA-NeXT \
      --model-path /path/to/llava-onevision-qwen2-7b-ov \
      --csv-path data.csv --video-root /path/to/videos --output-dir output/llava_onevision
"""

import argparse
import copy
import os
import sys
import warnings

import numpy as np
import torch
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-OneVision video evaluation")
    parser.add_argument("--repo-dir", type=str, required=True,
                        help="Path to cloned LLaVA-NeXT repository")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to LLaVA-OneVision model checkpoint")
    parser.add_argument("--model-name", type=str, default="llava_qwen")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--output-dir", type=str, default="output/llava_onevision")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to evaluation CSV file")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    return parser.parse_args()


def load_video_frames(video_path, max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
    return vr.get_batch(frame_idx).asnumpy()


def main():
    args = parse_args()

    sys.path.insert(0, args.repo_dir)

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from common import EvalSample, load_eval_dataset, run_evaluation

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, args.model_name,
        device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    device = "cuda"

    def infer_fn(prompt: str, sample: EvalSample) -> str:
        num_frames = sample.num_frames or 16
        video_frames = load_video_frames(sample.video_path, num_frames)

        frames_tensor = image_processor.preprocess(
            video_frames, return_tensors="pt"
        )["pixel_values"].half().cuda()

        question = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        conv = copy.deepcopy(conv_templates[args.conv_mode])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_str = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        image_sizes = [frame.size for frame in video_frames]

        cont = model.generate(
            input_ids,
            images=[frames_tensor],
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=args.max_new_tokens,
            modalities=["video"],
        )
        return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

    samples = load_eval_dataset(args.csv_path, args.video_root)
    run_evaluation(samples, infer_fn, args.output_dir)


if __name__ == "__main__":
    main()
