"""
Evaluate a trained sequence completion transformer on even-length reverse tasks.
"""

from __future__ import annotations

import argparse
import itertools
import string
from pathlib import Path

import torch

from completion_core.inference import complete_sequence
from completion_core.modeling import load_model_and_config
from completion_core.vocabulary import Vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="Checks lengths up to 2*n")
    parser.add_argument("--charset-size", type=int, default=6, help="Alphabet dimension")
    parser.add_argument("--train-file", type=Path, default=Path("inputs/reverse.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocabulary.from_pickle(args.data_dir / "meta.pkl")
    model, model_cfg = load_model_and_config(
        args.out_dir / "model.pth",
        vocab,
        device,
    )

    print(f"Loaded format: {'2-char' if vocab.is_2char else '1-char'}")

    trained_sequences: set[str] = set()
    available_chars = string.ascii_lowercase[:min(max(1, args.charset_size), 26)]

    print(f"\nEvaluating combinations across charset: {list(available_chars)}")
    print("-" * 60)

    for length_step in range(1, args.n + 1):
        actual_len = 2 * length_step
        correct = 0
        total = 0

        for items in itertools.product(available_chars, repeat=actual_len):
            lhs_string = "".join(items)
            if lhs_string in trained_sequences:
                continue

            ground_truth = lhs_string[::-1]
            prediction = complete_sequence(
                lhs=lhs_string,
                model=model,
                vocab=vocab,
                max_new_tokens=len(vocab.tokenize_string(lhs_string)) + 1,
                max_supported_len=model_cfg.seq_len,
                device=device,
            )

            if prediction == ground_truth:
                correct += 1
            total += 1

        accuracy = 0.0 if total == 0 else (correct / total) * 100
        print(
            f"String Length {actual_len:02d} | "
            f"Unseen Evaluated: {total:<6} | "
            f"Correct: {correct:<6} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    print("-" * 60)


if __name__ == "__main__":
    main()