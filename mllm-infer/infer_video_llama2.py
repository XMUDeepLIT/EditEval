"""
VideoLLaMA2 inference for video editing quality evaluation.

Setup:
  1. Clone https://github.com/DAMO-NLP-SG/VideoLLaMA2 and install per its README
  2. conda activate llama2
  3. Download VideoLLaMA2-7B-16F from HuggingFace

Supports both single-GPU and multi-GPU parallel execution:
  Single:  python infer_video_llama2.py --repo-dir /path/to/VideoLLaMA2 ...
  Multi:   python infer_video_llama2.py --repo-dir ... --start-index 0 --end-index 319

Usage:
  python infer_video_llama2.py --repo-dir /path/to/VideoLLaMA2 \
      --model-path /path/to/VideoLLaMA2-7B-16F \
      --csv-path data.csv --video-root /path/to/videos --output-dir output/video_llama2
"""

import argparse
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="VideoLLaMA2 video evaluation inference")
    parser.add_argument("--repo-dir", type=str, required=True,
                        help="Path to cloned VideoLLaMA2 repository")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to VideoLLaMA2 model checkpoint")
    parser.add_argument("--output-dir", type=str, default="output/video_llama2")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to evaluation CSV file")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--conv-mode", type=str, default="llama2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--start-index", type=int, default=None,
                        help="Start index for parallel processing")
    parser.add_argument("--end-index", type=int, default=None,
                        help="End index (inclusive) for parallel processing")
    return parser.parse_args()


def main():
    args = parse_args()

    sys.path.insert(0, args.repo_dir)

    from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
    from videollama2.conversation import conv_templates
    from videollama2.mm_utils import (
        get_model_name_from_path, process_video, tokenizer_MMODAL_token,
    )
    from videollama2.model.builder import load_pretrained_model
    from common import EvalSample, load_eval_dataset, run_evaluation

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, None, model_name
    )
    model = model.to("cuda:0")

    def infer_fn(prompt: str, sample: EvalSample) -> str:
        tensor = process_video(
            sample.video_path, processor, model.config.image_aspect_ratio
        ).to(dtype=torch.float16, device="cuda:0", non_blocking=True)

        default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
        modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
        question = default_mm_token + "\n" + prompt

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_str = conv.get_prompt()

        input_ids = tokenizer_MMODAL_token(
            prompt_str, tokenizer, modal_token_index, return_tensors="pt"
        ).unsqueeze(0).to("cuda:0")

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images_or_videos=[tensor],
                modal_list=["video"],
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    samples = load_eval_dataset(args.csv_path, args.video_root)
    if args.start_index is not None and args.end_index is not None:
        samples = samples[args.start_index:args.end_index + 1]

    run_evaluation(samples, infer_fn, args.output_dir)


if __name__ == "__main__":
    main()
