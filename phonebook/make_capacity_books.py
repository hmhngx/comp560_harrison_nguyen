"""Create nested capacity books and a held-out control book for the capacity experiment.

For each book seed one large random book is made with the README's own generator. The book of N
entries is its first N lines, so a larger book only adds entries. The control book holds entries
the model never trains on (no shared name or number, and short enough for the smallest context):
exact recall on it should stay near zero, which is what separates memorization from leakage.

Usage:
    python phonebook/make_capacity_books.py --seed 1 --n-max 2560 --out-dir phonebook/inputs
    python phonebook/make_capacity_books.py --seed 1 --rungs 10,20,40,57,80 --out-dir phonebook/inputs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.make_heldout_dataset import generate_phonebook_lines, longest_line

FIRST_RUNG = 10
CONTROL_ENTRIES = 20
CONTROL_CANDIDATES = 2000
CONTROL_SEED_OFFSET = 1_000_000


def doubling_ladder(n_max: int, first: int = FIRST_RUNG) -> list[int]:
    if n_max < first:
        raise ValueError(f"n_max must be at least {first}, got {n_max}")
    rungs = [first]
    while rungs[-1] * 2 <= n_max:
        rungs.append(rungs[-1] * 2)
    return rungs


def generate_big_book(seed: int, n_max: int) -> list[str]:
    """The README's generator, unchanged: python phonebook/create_phonebook.py N_MAX --seed SEED."""
    return generate_phonebook_lines(seed, count=n_max)


def prefix(lines: list[str], n: int) -> list[str]:
    if not 1 <= n <= len(lines):
        raise ValueError(f"prefix size must be between 1 and {len(lines)}, got {n}")
    return lines[:n]


def control_seed(seed: int) -> int:
    return CONTROL_SEED_OFFSET + seed


def make_control_book(candidates: list[str], train_lines: list[str], *, max_line: int,
                      count: int = CONTROL_ENTRIES) -> list[str]:
    """The first `count` candidates that fit the context and share no name or number with training."""
    names = {line.split("=")[0] for line in train_lines}
    numbers = {line.split("=")[1] for line in train_lines}
    chosen = []
    for line in candidates:
        name, number = line.split("=")
        if longest_line([line]) > max_line or name in names or number in numbers:
            continue
        chosen.append(line)
        if len(chosen) == count:
            return chosen
    raise ValueError(f"only {len(chosen)} of {count} control entries fit the context and avoid training entries")


def capacity_path(out_dir: Path, seed: int, n: int) -> Path:
    return out_dir / f"capacity_seed{seed}_n{n}.txt"


def control_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"capacity_seed{seed}_control.txt"


def write_capacity_inputs(out_dir: Path, seed: int, rungs: list[int]) -> dict:
    """Write one nested book per rung plus the control; returns {rung: path, "control": path}."""
    if not rungs or rungs != sorted(set(rungs)) or rungs[0] < 1:
        raise ValueError("rungs must be a strictly increasing list of positive sizes")
    big = generate_big_book(seed, rungs[-1])
    control = make_control_book(generate_big_book(control_seed(seed), CONTROL_CANDIDATES), big,
                                max_line=longest_line(prefix(big, rungs[0])))
    targets: dict = {n: capacity_path(out_dir, seed, n) for n in rungs}
    targets["control"] = control_path(out_dir, seed)
    for path in targets.values():
        if path.exists():
            raise ValueError(f"{path} already exists; refusing to overwrite a tracked input")
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in rungs:
        targets[n].write_bytes(("\n".join(prefix(big, n)) + "\n").encode("ascii"))
    targets["control"].write_bytes(("\n".join(control) + "\n").encode("ascii"))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n-max", type=int, help="write the doubling ladder 10, 20, 40, ... up to this size")
    group.add_argument("--rungs", help="comma-separated strictly increasing sizes to write")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        rungs = doubling_ladder(args.n_max) if args.n_max is not None else [int(n) for n in args.rungs.split(",")]
        for key, path in write_capacity_inputs(args.out_dir, args.seed, rungs).items():
            print(f"{key}: wrote {path}")
    except ValueError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
