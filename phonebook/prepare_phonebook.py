from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import string
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_INPUT = Path("inputs") / "phonebook.txt"
REQUIRED_ALNUM_CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits


from completion_core.prep import (
    build_char_vocab,
    encode_chars,
    load_lines,
    save_outputs,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_phonebook_lines(path: Path) -> list[str]:
    """Normalize the final newline and reject records the trainer cannot consume."""
    normalized = []
    for number, raw in enumerate(load_lines(path), start=1):
        line = raw.rstrip("\r\n")
        if line.count("=") != 1 or not all(part.strip() for part in line.split("=")):
            raise ValueError(f"Invalid completion record at {path}:{number}; expected non-empty name=answer.")
        normalized.append(line + "\n")
    return normalized


@dataclass
class PrepareConfig:
    input_file: Path = DEFAULT_INPUT
    train_input_file: Path | None = None
    eval_input_file: Path | None = None
    out_dir: Path = Path("data")
    train_split: float = 1.0
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
        default=None,
        help=(
            "Path to a single input text file used with --train-split "
            f"(default: {DEFAULT_INPUT})"
        ),
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=None,
        help="Path to an explicit training input text file",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=None,
        help="Path to an explicit evaluation input text file",
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
        default=1.0,
        help="Fraction of lines used for training (default: 1.0)",
    )
    args = parser.parse_args()

    has_train_input = args.train_input is not None
    has_eval_input = args.eval_input is not None
    if has_train_input != has_eval_input:
        parser.error("Both --train-input and --eval-input must be provided together.")

    if (has_train_input or has_eval_input) and args.input_file is not None:
        parser.error(
            "Cannot combine positional input_file with --train-input/--eval-input."
        )

    if not 0.0 <= args.train_split <= 1.0:
        parser.error("--train-split must be between 0.0 and 1.0.")

    cfg = PrepareConfig()
    cfg.input_file = args.input_file or DEFAULT_INPUT
    cfg.train_input_file = args.train_input
    cfg.eval_input_file = args.eval_input
    cfg.out_dir = args.out_dir
    cfg.train_split = args.train_split
    return cfg


def main() -> None:
    cfg = parse_args()

    if cfg.train_input_file is not None and cfg.eval_input_file is not None:
        train_lines = load_phonebook_lines(cfg.train_input_file)
        val_lines = load_phonebook_lines(cfg.eval_input_file)
        print("Mode            : explicit train/eval files")
        print(f"Train lines     : {len(train_lines)} ({cfg.train_input_file})")
        print(f"Eval lines      : {len(val_lines)} ({cfg.eval_input_file})")
    else:
        lines = load_phonebook_lines(cfg.input_file)
        split = int(len(lines) * cfg.train_split)
        train_lines = lines[:split]
        val_lines = lines[split:]
        print("Mode            : split single input file")
        print(f"Input lines     : {len(lines)} ({cfg.input_file})")
        print(f"Split index     : {split} ({cfg.train_split:.2f})")

    train_str = "".join(train_lines)
    val_str = "".join(val_lines)
    full_str = train_str + val_str

    vocab_source = full_str + REQUIRED_ALNUM_CHARS
    stoi, itos = build_char_vocab(vocab_source, cfg.pad_token)
    print(f"Forced alnum set: enabled ({len(REQUIRED_ALNUM_CHARS)} chars)")
    print(f"Vocabulary size : {len(stoi)}")

    train_ids = encode_chars(train_str, stoi)
    val_ids = encode_chars(val_str, stoi)
    print(f"Train tokens    : {len(train_ids)}")
    print(f"Val tokens      : {len(val_ids)}")

    save_outputs(cfg.out_dir, train_ids, val_ids, stoi, itos)
    print(f"\nFiles written to '{cfg.out_dir}/'")


if __name__ == "__main__":
    main()
