"""
Driver for phonebook Phase-B adaptation sweeps.

Runs train.py once per (learning rate, seed) combination and appends a JSON
record of the *exact* command used for every run to a manifest file. This
exists specifically to fix the gap documented in phonebook/sweep_lr/README.md:
the original 12-run sweep's commands were never recorded anywhere and could
not be recovered from shell history after the fact.

Usage:
    python phonebook/run_lr_sweep.py --lrs 0.0001 0.001 0.01 --seeds 42 123 999
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ADAPT_FROM = Path("phonebook/out_phaseA/model.pth")
DEFAULT_DATA_DIR = Path("phonebook/data_phaseB")
DEFAULT_CONFIG = Path("phonebook/config/phonebook.py")


def build_command(lr: float, epochs: int, seed: int, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "train.py",
        "--data-dir", str(DEFAULT_DATA_DIR),
        "--out-dir", str(out_dir),
        "--adapt-from", str(DEFAULT_ADAPT_FROM),
        str(DEFAULT_CONFIG),
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]


def run_name(lr: float, epochs: int, seed: int) -> str:
    return f"lr{lr}_ep{epochs}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lrs", type=float, nargs="+", required=True, help="Learning rates to sweep")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seeds to run at each lr")
    parser.add_argument("--out-root", type=Path, default=Path("phonebook/sweep_lr/seeded"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "manifest.jsonl"

    combos = list(itertools.product(args.lrs, args.seeds))
    print(f"Planned runs: {len(combos)} ({len(args.lrs)} lr x {len(args.seeds)} seeds, {args.epochs} epochs each)")

    for i, (lr, seed) in enumerate(combos, start=1):
        name = run_name(lr, args.epochs, seed)
        out_dir = args.out_root / name
        cmd = build_command(lr, args.epochs, seed, out_dir)

        print(f"\n[{i}/{len(combos)}] {' '.join(cmd)}")
        if args.dry_run:
            continue

        start = time.perf_counter()
        result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
        elapsed = time.perf_counter() - start

        record = {
            "name": name,
            "command": cmd,
            "cwd": str(ROOT_DIR),
            "lr": lr,
            "epochs": args.epochs,
            "seed": seed,
            "returncode": result.returncode,
            "elapsed_sec": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}) in {elapsed:.1f}s")
            print(result.stderr[-2000:])
        else:
            print(f"  OK in {elapsed:.1f}s -> {out_dir}")

    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
