"""
Prepare a character-level dataset for train_completions.py.

Reads a plain-text file where each line is one sequence, splits it 90/10 into
train and validation sets, builds a character vocabulary (plus a pad token '_'),
and writes train.bin, val.bin, and meta.pkl to an output directory.

Usage:
    python prepare_1char.py                          # reads ../inputs/capital.txt, writes data/
    python prepare_1char.py path/to/input.txt
    python prepare_1char.py ../inputs/capital.txt --out-dir 1-Char/data
    python prepare_1char.py ../inputs/capital.txt --train-split 0.95
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from completion_core.prep import (
    build_char_vocab,
    encode_chars,
    load_lines,
    save_outputs,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PrepareConfig:
    input_file: Path = Path("inputs/capital.txt")
    out_dir: Path = Path("data")
    train_split: float = 0.9
    pad_token: str = "_"        # Must match the pad token expected by the trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> PrepareConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("inputs/capital.txt"),
        help="Path to the input text file (default: inputs/capital.txt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write train.bin, val.bin, meta.pkl (default: data/)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of lines used for training (default: 0.9)",
    )
    args = parser.parse_args()

    cfg = PrepareConfig()
    cfg.input_file = args.input_file
    cfg.out_dir = args.out_dir
    cfg.train_split = args.train_split
    return cfg


def main() -> None:
    cfg = parse_args()

    lines = load_lines(cfg.input_file)
    print(f"Lines loaded    : {len(lines)}")

    split = int(len(lines) * cfg.train_split)
    train_str = "".join(lines[:split])
    val_str = "".join(lines[split:])
    full_str = train_str + val_str

    stoi, itos = build_char_vocab(full_str, cfg.pad_token)
    print(f"Vocabulary size : {len(stoi)}")

    train_ids = encode_chars(train_str, stoi)
    val_ids = encode_chars(val_str, stoi)
    print(f"Train tokens    : {len(train_ids)}")
    print(f"Val tokens      : {len(val_ids)}")

    save_outputs(cfg.out_dir, train_ids, val_ids, stoi, itos)
    print(f"\nFiles written to '{cfg.out_dir}/'")


if __name__ == "__main__":
    main()