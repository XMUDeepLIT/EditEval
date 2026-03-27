"""
VILA-1.5 inference for video editing quality evaluation.

Setup:
  1. Clone https://github.com/NVlabs/VILA and install per its README
  2. conda activate vila
  3. Download VILA1.5-40b (or other checkpoints) from HuggingFace

Usage:
  python infer_vila.py --repo-dir /path/to/VILA --model-path /path/to/VILA1.5-40b \
      --csv-path data.csv --video-root /path/to/videos --output-dir output/vila
"""

import argparse
import os
import re
import sys

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="VILA video evaluation inference")
    parser.add_argument("--repo-dir", type=str, required=True,
                        help="Path to cloned VILA repository")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to VILA model checkpoint")
    parser.add_argument("--output-dir", type=str, default="output/vila")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to evaluation CSV file")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    sys.path.insert(0, args.repo_dir)

    from llava.constants import (
        DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER, IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria, get_model_name_from_path,
        opencv_extract_frames, process_images, tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from common import EvalSample, load_eval_dataset, run_evaluation

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_name, None
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    def infer_fn(prompt: str, sample: EvalSample) -> str:
        assert os.path.exists(sample.video_path), f"Video not found: {sample.video_path}"

        num_frames = max(1, (sample.num_frames or 16) - 1)
        images = opencv_extract_frames(sample.video_path, num_frames)

        qs = "<image>\n" + prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        elif DEFAULT_IMAGE_TOKEN not in qs:
            token = image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
            qs = (token + "\n") * len(images) + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_str = conv.get_prompt()

        images_tensor = process_images(images, image_processor, model.config).to(
            model.device, dtype=torch.float16
        )
        input_ids = tokenizer_image_token(
            prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if output.endswith(stop_str):
            output = output[:-len(stop_str)].strip()
        return output

    samples = load_eval_dataset(args.csv_path, args.video_root)
    run_evaluation(samples, infer_fn, args.output_dir)


if __name__ == "__main__":
    main()
