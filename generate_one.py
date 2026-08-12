from __future__ import annotations

import argparse
from pathlib import Path

import torch

from completion_core.inference import complete_sequence
from completion_core.modeling import load_model_and_config
from completion_core.vocabulary import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a single prompt using the sequence completion transformer."
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The input prompt string for the model (e.g., 'abc' or 'abc=')",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out",
        help="Directory containing the model checkpoint (default: out)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing metadata vocabulary (default: data)",
    )
    return parser.parse_args()


def normalize_prompt(prompt: str) -> str:
    if prompt.endswith("="):
        return prompt
    if "=" in prompt:
        return prompt.split("=", maxsplit=1)[0] + "="
    return prompt + "="


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    weights_path = out_dir / "model.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find model weights at {weights_path}")

    vocab = Vocabulary.from_pickle(data_dir / "meta.pkl")
    model, model_cfg = load_model_and_config(
        weights_path=weights_path,
        vocab=vocab,
        device=device,
    )

    prompt = normalize_prompt(args.prompt)
    lhs = prompt.split("=", maxsplit=1)[0]
    prediction = complete_sequence(
        lhs=lhs,
        model=model,
        vocab=vocab,
        max_new_tokens=20,
        max_supported_len=model_cfg.seq_len,
        device=device,
    )
    print(prediction)


if __name__ == "__main__":
    main()