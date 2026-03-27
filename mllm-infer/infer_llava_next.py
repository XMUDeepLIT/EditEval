"""
LLaVA-NeXT-Video inference for video editing quality evaluation.

Setup:
  1. Clone https://github.com/LLaVA-VL/LLaVA-NeXT and install per its README
  2. conda activate llava
  3. Download one of: LLaVA-NeXT-Video-7B / 32B-Qwen / 34B from HuggingFace

Usage (7B example):
  python infer_llava_next.py --repo-dir /path/to/LLaVA-NeXT \
      --model-path /path/to/LLaVA-NeXT-Video-7B --conv-mode vicuna_v1 \
      --csv-path data.csv --video-root /path/to/videos --output-dir output/llava_next_7b

Usage (32B-Qwen):
  python infer_llava_next.py --repo-dir /path/to/LLaVA-NeXT \
      --model-path /path/to/LLaVA-NeXT-Video-32B-Qwen --conv-mode qwen_1_5 \
      --mm-newline-position grid --output-dir output/llava_next_32b ...
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-NeXT-Video evaluation")
    parser.add_argument("--repo-dir", type=str, required=True,
                        help="Path to cloned LLaVA-NeXT repository")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to LLaVA-NeXT-Video model checkpoint")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--output-dir", type=str, default="output/llava_next")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to evaluation CSV file")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--mm-spatial-pool-mode", type=str, default="average")
    parser.add_argument("--mm-newline-position", type=str, default="no_token")
    parser.add_argument("--load-8bit", action="store_true", default=False)
    parser.add_argument("--force-sample", action="store_true", default=False)
    return parser.parse_args()


def load_video(video_path, num_frames, force_sample=False):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    frame_idx = list(range(total))
    if len(frame_idx) > num_frames or force_sample:
        frame_idx = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    return vr.get_batch(frame_idx).asnumpy()


def main():
    args = parse_args()

    sys.path.insert(0, args.repo_dir)

    from llava.constants import (
        DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from common import EvalSample, load_eval_dataset, run_evaluation

    model_name = get_model_name_from_path(args.model_path)
    cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

    overwrite_config = {
        "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
        "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
        "mm_newline_position": args.mm_newline_position,
    }

    if "qwen" not in args.model_path.lower():
        if "224" in cfg_pretrained.mm_vision_tower:
            least_tokens = args.num_frames * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
        else:
            least_tokens = args.num_frames * (24 // args.mm_spatial_pool_stride) ** 2 + 1000

        scaling = math.ceil(least_tokens / 4096)
        if scaling >= 2:
            if "vicuna" in cfg_pretrained._name_or_path.lower():
                overwrite_config["rope_scaling"] = {
                    "factor": float(scaling), "type": "linear"
                }
            overwrite_config["max_sequence_length"] = 4096 * scaling
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name,
        load_8bit=args.load_8bit, overwrite_config=overwrite_config,
    )

    if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = 151643

    def infer_fn(prompt: str, sample: EvalSample) -> str:
        if not os.path.exists(sample.video_path):
            raise FileNotFoundError(sample.video_path)

        video = load_video(sample.video_path, args.num_frames, args.force_sample)
        video_tensor = image_processor.preprocess(
            video, return_tensors="pt"
        )["pixel_values"].half().cuda()

        qs = prompt
        if model.config.mm_use_im_start_end:
            qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN +
                  DEFAULT_IM_END_TOKEN + "\n" + qs)
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_str = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            gen_kwargs = dict(
                inputs=input_ids, images=[video_tensor],
                attention_mask=attention_mask, modalities=["video"],
                do_sample=False, temperature=0.0, max_new_tokens=args.max_new_tokens,
                top_p=0.1, num_beams=1, use_cache=True,
            )
            if "mistral" not in cfg_pretrained._name_or_path.lower():
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
            output_ids = model.generate(**gen_kwargs)

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if "mistral" not in cfg_pretrained._name_or_path.lower() and output.endswith(stop_str):
            output = output[:-len(stop_str)].strip()
        return output

    samples = load_eval_dataset(args.csv_path, args.video_root)
    run_evaluation(samples, infer_fn, args.output_dir)


if __name__ == "__main__":
    main()
