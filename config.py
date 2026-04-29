"""
Centralized configuration for the DPLIL project.
All model-specific and experiment-specific parameters should be set here
or overridden via command-line arguments in each script.
"""
import argparse
from pathlib import Path


def get_base_args(parser=None):
    """Add common arguments shared across all scripts."""
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=str, required=True,
                        help="Root directory of the project")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained VLM checkpoint")
    parser.add_argument("--hidden_size", type=int, required=True,
                        help="Hidden dimension of the language model backbone")
    parser.add_argument("--num_layers", type=int, required=True,
                        help="Number of decoder layers in the language model")
    parser.add_argument("--image_token_id", type=int, required=True,
                        help="Token ID used for image tokens in the VLM")
    parser.add_argument("--languages", type=str, nargs="+",
                        default=["en", "zh", "ko", "de"],
                        help="Languages to evaluate")
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser


def get_paths(args):
    """Derive standard directory paths from project root."""
    root = Path(args.project_root)
    return {
        "project_root": root,
        "data_dir": root / "data",
        "checkpoint_dir": root / "checkpoints",
        "result_dir": root / "results",
    }


SHORT_ANSWER_SUFFIX = {
    "en": " Answer the question using a single word or phrase.",
    "zh": " 请用一个词或短语回答。",
    "ko": " 한 단어 또는 짧은 구문으로 답하세요.",
    "de": " Beantworten Sie die Frage mit einem einzigen Wort oder einer kurzen Phrase.",
    "ja": " 一言または短いフレーズで答えてください。",
}

LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "de": "German",
}


def normalize_answer(ans: str) -> str:
    """Normalize answer for cross-lingual comparison."""
    ans = ans.lower().strip()
    ans = ans.split("\n")[0].strip()
    for prefix in ["answer:", "答案:", "답:", "antwort:"]:
        if ans.startswith(prefix):
            ans = ans[len(prefix):].strip()
    ans = ans.rstrip(".,。、")

    yes_words = {"yes", "是", "是的", "对", "对的", "はい", "네", "ja", "예"}
    no_words = {"no", "不是", "没有", "不", "否", "いいえ", "아니요", "아니오", "nein"}
    if ans in yes_words:
        return "yes"
    if ans in no_words:
        return "no"
    return ans
