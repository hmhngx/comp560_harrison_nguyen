"""Throughput benchmark (not evidence): N concurrent 500-epoch replay runs, default torch threads versus one thread each.

It decided how the E5 and E4 batches were run (see the README of this folder for the numbers). It trains on the E3 phonebook's
first Phase-A checkpoint (hash checked against E3's pin) with k copies of B.

    python phonebook/calibration/throughput_benchmark.py --n 7 --mode one [--k 5] [--work WORK]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.calibration.e5_pilots import D0_CHECKPOINTS, DEFAULT_WORK, d0_data, train_env, verify_d0_pins


def benchmark(work: Path, n: int, mode: str, k: int) -> tuple[list[float], float]:
    verify_d0_pins()
    data = d0_data(work, k)
    env = train_env("1" if mode == "one" else "default")

    def job(i: int) -> float:
        out = work / "bench" / f"{mode}_{n}_{i}"
        cmd = [sys.executable, "train.py", "phonebook/config/phonebook.py", "--device", "cpu", "--data-dir", str(data), "--out-dir", str(out),
               "--seed", "42", "--lr", "0.001", "--epochs", "500", "--batch-size", "128", "--accuracy-interval", "1", "--l2sp-lambda", "0.0",
               "--adapt-from", str(ROOT_DIR / D0_CHECKPOINTS["d0_seed42"])]
        started = time.perf_counter()
        done = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, env=env)
        if done.returncode != 0:
            raise RuntimeError(done.stderr[-300:])
        return time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(n) as pool:
        times = list(pool.map(job, range(n)))
    return times, time.perf_counter() - started


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--mode", choices=("default", "one"), required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args(argv)
    times, wall = benchmark(args.work, args.n, args.mode, args.k)
    print(f"{args.mode} x{args.n} (k={args.k}): per-run {[round(t) for t in times]} s, wall {wall:.0f} s")


if __name__ == "__main__":
    main()
