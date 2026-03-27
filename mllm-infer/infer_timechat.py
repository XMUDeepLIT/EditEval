"""
TimeChat inference for video editing quality evaluation.

Setup:
  1. Clone https://github.com/RenShuhuai-Andy/TimeChat and install per its README
  2. conda activate timechat
  3. Download TimeChat-7b checkpoint

IMPORTANT: This script must be run from the TimeChat repo directory
           (because it reads eval_configs/timechat.yaml by default).

Usage:
  cd /path/to/TimeChat
  python /path/to/infer_timechat.py --repo-dir . --model-dir /path/to/TimeChat-7b \
      --csv-path data.csv --video-root /path/to/videos --output-dir output/timechat
"""

import argparse
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="TimeChat video evaluation inference")
    parser.add_argument("--repo-dir", type=str, required=True,
                        help="Path to cloned TimeChat repository")
    parser.add_argument("--cfg-path", type=str,
                        default="eval_configs/timechat.yaml")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to TimeChat-7b model directory")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="output/timechat")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to evaluation CSV file")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--options", nargs="+", default=None,
                        help="Override config settings (key=value)")
    return parser.parse_args()


def build_timechat_args(args):
    """Build a minimal args namespace for TimeChat's Config class."""

    class TCArgs:
        pass

    tc = TCArgs()
    tc.cfg_path = args.cfg_path
    tc.gpu_id = args.gpu_id
    tc.num_beams = args.num_beams
    tc.temperature = args.temperature
    tc.text_query = ""
    tc.video_path = ""
    tc.options = args.options or []
    return tc


def main():
    args = parse_args()

    sys.path.insert(0, args.repo_dir)

    from timechat.common.config import Config
    from timechat.common.registry import registry
    from timechat.conversation.conversation_video import Chat, conv_llava_llama_2
    from timechat.processors.video_processor import load_video

    # Registration imports required by TimeChat
    from timechat.datasets.builders import *  # noqa: F401,F403
    from timechat.models import *  # noqa: F401,F403
    from timechat.processors import *  # noqa: F401,F403
    from timechat.runners import *  # noqa: F401,F403
    from timechat.tasks import *  # noqa: F401,F403

    from common import EvalSample, load_eval_dataset, run_evaluation

    tc_args = build_timechat_args(args)
    cfg = Config(tc_args)
    device = f"cuda:{args.gpu_id}"

    model_ckpt = os.path.join(args.model_dir, "timechat_7b.pth")
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = model_ckpt
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device).eval()

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )

    chat = Chat(model, vis_processor, device=device)
    print("TimeChat initialization finished")

    _cached_sample_idx = {"idx": None, "chat_state": None, "img_list": None}

    def infer_fn(prompt: str, sample: EvalSample) -> str:
        if _cached_sample_idx["idx"] != sample.index:
            img_list = []
            chat_state = conv_llava_llama_2.copy()
            chat_state.system = (
                "You are able to understand the visual content that the user provides. "
                "Follow the instructions carefully and explain your answers in detail."
            )
            chat.upload_video_without_audio(
                video_path=sample.video_path,
                conv=chat_state,
                img_list=img_list,
                n_frms=args.num_frames,
            )
            _cached_sample_idx["idx"] = sample.index
            _cached_sample_idx["chat_state"] = chat_state
            _cached_sample_idx["img_list"] = img_list

        chat_state = _cached_sample_idx["chat_state"]
        img_list = _cached_sample_idx["img_list"]

        chat.ask(prompt, chat_state)
        response = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=args.num_beams,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_length=2000,
        )[0]
        return response

    samples = load_eval_dataset(args.csv_path, args.video_root)
    run_evaluation(samples, infer_fn, args.output_dir)


if __name__ == "__main__":
    main()
