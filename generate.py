"""
Evaluate the accuracy of a trained sequence completion transformer.

Usage:
    python generate.py
    python generate.py path/to/eval.txt
    python generate.py --out-dir out_1char --data-dir 1-Char/data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from completion_core.inference import evaluate_file
from completion_core.modeling import load_model_and_config
from completion_core.vocabulary import Vocabulary


@dataclass
class EvalConfig:
    out_dir: Path = Path("out_1char")
    data_dir: Path = Path("1-Char/data")
    eval_file: Path | None = None
    max_new_tokens: int = 10


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_file",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the evaluation file (default: <data-dir>/input.txt)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out_1char"))
    parser.add_argument("--data-dir", type=Path, default=Path("1-Char/data"))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Upper bound on generated tokens per prompt (default: 10)",
    )
    args = parser.parse_args()

    return EvalConfig(
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        eval_file=args.eval_file,
        max_new_tokens=args.max_new_tokens,
    )


def main() -> None:
    cfg = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_file = cfg.eval_file or (cfg.data_dir / "input.txt")

    vocab = Vocabulary.from_pickle(cfg.data_dir / "meta.pkl")
    model, model_cfg = load_model_and_config(
        weights_path=cfg.out_dir / "model.pth",
        vocab=vocab,
        device=device,
    )
    print(
        "Checkpoint loaded "
        f"(seq_len={model_cfg.seq_len}, device={device}, "
        f"format={'2-char' if vocab.is_2char else '1-char'})\n"
    )

    correct, total = evaluate_file(
        eval_file=eval_file,
        model=model,
        vocab=vocab,
        max_new_tokens=cfg.max_new_tokens,
        max_supported_len=model_cfg.seq_len,
        device=device,
    )

    print(f"\n{'=' * 40}")
    print(f"ACCURACY REPORT - {eval_file.name}")
    print(f"  Lines processed : {total}")
    if total > 0:
        print(f"  Correct         : {correct}")
        print(f"  Accuracy        : {correct / total * 100:.2f}%")
    else:
        print("  No valid sequences were found.")
    print("=" * 40)


if __name__ == "__main__":
    main()