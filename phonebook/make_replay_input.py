"""
Build a replay/rehearsal training file for phonebook Phase-B adaptation:
k copies of phonebook A's lines, followed by phonebook B's lines once.

This implements the candidate from phonebook/docs/mitigation-options.md
("mix repeated copies of phonebook A into Phase-B training... so the model
keeps seeing A while learning B") -- repeating A, not B, since the goal here
is preventing forgetting of A, not speeding up learning of B (the original
idea in phonebook/README.md's "Next steps" repeats B instead, for a
different question).

Line order within each source is preserved (not interleaved or shuffled):
train.py's DataLoader already shuffles every epoch (shuffle=True), so file
order has no effect on training dynamics -- it only matters for a human
reading the file, and preserving source order keeps that readable.

Usage:
    python phonebook/make_replay_input.py --phonebook-a inputs/phonebookA.txt \
        --phonebook-b inputs/phonebookB.txt --k 3 --out inputs/replay_k3.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_replay_lines(a_lines: list[str], b_lines: list[str], k: int) -> list[str]:
    if k < 1:
        raise ValueError(f"k must be >= 1 (repeating phonebook A zero or fewer times isn't replay), got {k}")
    return a_lines * k + b_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phonebook-a", type=Path, required=True)
    parser.add_argument("--phonebook-b", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True, help="Number of times to repeat phonebook A's lines")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError(f"k must be >= 1 (repeating phonebook A zero or fewer times isn't replay), got {args.k}")

    a_lines = args.phonebook_a.read_text(encoding="utf-8").splitlines()
    b_lines = args.phonebook_b.read_text(encoding="utf-8").splitlines()
    lines = build_replay_lines(a_lines, b_lines, args.k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} lines ({len(a_lines)} x {args.k} from A, {len(b_lines)} from B) to {args.out}")


if __name__ == "__main__":
    main()
