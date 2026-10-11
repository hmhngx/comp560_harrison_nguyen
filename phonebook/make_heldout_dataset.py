"""Create held-out 20-entry phonebooks: book A is the first ten entries, book B the last ten.

The trainer sizes its context from the training book alone, so a seed is usable
only when A and B have the same longest line. Otherwise Phase A crashes on the
longer evaluation book, or the Phase-A checkpoint cannot be loaded for Phase B.

Usage:
    python phonebook/make_heldout_dataset.py --seed 2 --out-dir phonebook/inputs
    python phonebook/make_heldout_dataset.py --first-qualifying 3 --out-dir phonebook/inputs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
ENTRIES = 20


def split_books(lines: list[str]) -> tuple[list[str], list[str]]:
    if len(lines) != ENTRIES:
        raise ValueError(f"expected {ENTRIES} entries, got {len(lines)}")
    return lines[:ENTRIES // 2], lines[ENTRIES // 2:]


def longest_line(lines: list[str]) -> int:
    """Longest line in tokens, counting the newline the trainer includes."""
    return max(len(line.rstrip("\n")) + 1 for line in lines)


def qualifies(a_lines: list[str], b_lines: list[str]) -> bool:
    return longest_line(a_lines) == longest_line(b_lines)


def first_qualifying_seeds(count: int, make_lines: Callable[[int], list[str]], start: int = 1, limit: int = 1000) -> list[int]:
    found = []
    for seed in range(start, start + limit):
        if qualifies(*split_books(make_lines(seed))):
            found.append(seed)
            if len(found) == count:
                return found
    raise ValueError(f"found only {len(found)} qualifying seeds in {limit} tries")


def generate_phonebook_lines(seed: int, count: int = ENTRIES) -> list[str]:
    """The README's generator, unchanged: python phonebook/create_phonebook.py COUNT --seed SEED."""
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "phonebook.txt"
        subprocess.run([sys.executable, str(ROOT_DIR / "phonebook" / "create_phonebook.py"), str(count),
                        "--seed", str(seed), "-o", str(output)], check=True, capture_output=True, text=True)
        return output.read_text(encoding="ascii").splitlines()


def book_paths(out_dir: Path, seed: int) -> tuple[Path, Path]:
    return out_dir / f"heldout_seed{seed}_phonebookA.txt", out_dir / f"heldout_seed{seed}_phonebookB.txt"


def write_books(seed: int, out_dir: Path) -> tuple[Path, Path]:
    a_lines, b_lines = split_books(generate_phonebook_lines(seed))
    if not qualifies(a_lines, b_lines):
        raise ValueError(f"seed {seed}: the longest line is {longest_line(a_lines)} in A but {longest_line(b_lines)} in B")
    paths = book_paths(out_dir, seed)
    for path in paths:
        if path.exists():
            raise ValueError(f"{path} already exists; refusing to overwrite a tracked input")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, lines in zip(paths, (a_lines, b_lines)):
        path.write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--first-qualifying", type=int, metavar="N")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        seeds = [args.seed] if args.seed is not None else first_qualifying_seeds(args.first_qualifying, generate_phonebook_lines)
        for seed in seeds:
            a_path, b_path = write_books(seed, args.out_dir)
            print(f"seed {seed}: wrote {a_path} and {b_path}")
    except ValueError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
