"""
Compute correlation between MLLM results and human annotations.

Usage:
    python compute_correlation.py --labeled_csv labeled_full_update_v2.csv --mllm_dir mllm_results/vila_updated
    python compute_correlation.py --labeled_csv labeled_full_update_v2.csv --mllm_dir mllm_results/vila_updated mllm_results/timechat_updated
    python compute_correlation.py --labeled_csv labeled_full_update_v2.csv --mllm_dir all   # run all subdirs in mllm_results/
"""

import argparse
import glob
import json
import os
import re

import pandas as pd
from prettytable import PrettyTable
from scipy.stats import kendalltau, pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def extract_answer(text, unmatched=None, idx=None):
    """Extract numeric score from MLLM text output like 'the score is 3.5'."""
    pattern = r"the score is (\d+(\.\d+)?)(?=\s*[.,]|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        if unmatched is not None:
            unmatched.append(idx if idx is not None else text)
        return 2.5  # default fallback


def calculate_correlation(pred_scores, human_scores):
    """Return dict with pearson / spearman / kendalltau between two score lists."""
    assert len(pred_scores) == len(human_scores), \
        f"Length mismatch: pred={len(pred_scores)}, human={len(human_scores)}"
    return {
        "pearson": pearsonr(pred_scores, human_scores)[0],
        "spearman": spearmanr(pred_scores, human_scores)[0],
        "kendalltau": kendalltau(pred_scores, human_scores)[0],
    }


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def load_mllm_scores(mllm_dir):
    """
    Load MLLM JSON results from a directory.
    Each file is {000000..NNNNNN}.json containing keys:
        Textual_Faithfulness, Frame_Consistency, Video_Fidelity
    Returns a list of dicts (one per sample, sorted by filename).
    """
    score_files = sorted(glob.glob(os.path.join(mllm_dir, "*.json")))
    if not score_files:
        raise FileNotFoundError(f"No JSON files found in {mllm_dir}")
    scores = []
    for f in score_files:
        with open(f, "r") as fp:
            scores.append(json.load(fp))
    return scores


def compute_correlations(mllm_dir, human_tf, human_fc, human_vf, verbose=True):
    """
    Compute Pearson / Spearman / Kendall-tau correlations for three dimensions.

    Args:
        mllm_dir: path to folder of per-sample JSON files
        human_tf: list of human Textual Faithfulness scores
        human_fc: list of human Frame Consistency scores
        human_vf: list of human Video Fidelity scores
        verbose:  whether to print unmatched info

    Returns:
        dict of {dimension: {pearson, spearman, kendalltau}}
    """
    scores = load_mllm_scores(mllm_dir)

    dimensions = {
        "Textual_Faithfulness": ("编辑文本对齐度", human_tf),
        "Frame_Consistency":    ("帧连续性",      human_fc),
        "Video_Fidelity":       ("视频真实度",     human_vf),
    }

    results = {}
    for dim_key, (dim_cn, human_scores) in dimensions.items():
        unmatched = []
        pred = [
            float(extract_answer(
                item.get(dim_key, ""),
                unmatched=unmatched,
                idx=i,
            ))
            for i, item in enumerate(scores)
        ]

        # Truncate to match lengths (some models only ran partial)
        n = min(len(pred), len(human_scores))
        pred = pred[:n]
        ref = list(human_scores[:n])

        if verbose:
            print(f"  [{dim_key}] unmatched: {len(unmatched)}")

        corr = calculate_correlation(pred, ref)
        results[dim_key] = corr

    return results


def print_results(results, model_name=""):
    """Pretty-print correlation results as a table."""
    header = f" {model_name} " if model_name else ""
    print(f"\n{'=' * 20}{header}{'=' * 20}")

    table = PrettyTable()
    table.field_names = ["Dimension", "Pearson", "Spearman", "Kendall-τ"]
    table.align["Dimension"] = "l"

    dim_names = {
        "Textual_Faithfulness": "Textual Faithfulness",
        "Frame_Consistency":    "Frame Consistency",
        "Video_Fidelity":       "Video Fidelity",
    }

    for dim_key in ["Textual_Faithfulness", "Frame_Consistency", "Video_Fidelity"]:
        corr = results[dim_key]
        table.add_row([
            dim_names[dim_key],
            f"{corr['pearson']:.4f}",
            f"{corr['spearman']:.4f}",
            f"{corr['kendalltau']:.4f}",
        ])

    # Also print the latex-friendly format: pearson & spearman & kendalltau
    print(table)
    print("\nLaTeX-friendly format:")
    for dim_key in ["Textual_Faithfulness", "Frame_Consistency", "Video_Fidelity"]:
        c = results[dim_key]
        print(f"  {c['pearson']:.2f} & {c['spearman']:.2f} & {c['kendalltau']:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute correlation between MLLM results and human scores."
    )
    parser.add_argument(
        "--labeled_csv", type=str, required=True,
        help="Path to the labeled CSV (e.g. labeled_full_update_v2.csv)",
    )
    parser.add_argument(
        "--mllm_dir", type=str, nargs="+", required=True,
        help="Path(s) to MLLM result directory (each containing 000000.json ... NNNNNN.json). "
             "Use 'all' to run on all subdirectories under mllm_results/.",
    )
    parser.add_argument(
        "--mllm_root", type=str, default="mllm_results",
        help="Root directory for MLLM results when using --mllm_dir all (default: mllm_results)",
    )
    args = parser.parse_args()

    # Load human scores from labeled CSV
    print(f"Loading human scores from: {args.labeled_csv}")
    df = pd.read_csv(args.labeled_csv)
    human_tf = list(df["Textual Faithfulness"])
    human_fc = list(df["Frame Consistency"])
    human_vf = list(df["Video Fidelity"])
    print(f"  Loaded {len(human_tf)} samples\n")

    # Resolve MLLM directories
    if args.mllm_dir == ["all"]:
        root = args.mllm_root
        mllm_dirs = sorted([
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        print(f"Running on all {len(mllm_dirs)} model(s) under {root}/\n")
    else:
        mllm_dirs = args.mllm_dir

    # Compute & print correlations for each model
    for mllm_dir in mllm_dirs:
        model_name = os.path.basename(mllm_dir.rstrip("/"))
        print(f"Processing: {mllm_dir}")
        try:
            results = compute_correlations(mllm_dir, human_tf, human_fc, human_vf)
            print_results(results, model_name=model_name)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
