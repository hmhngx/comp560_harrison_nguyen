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
import csv
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


def is_run_complete(out_dir: Path, expected_epochs: int) -> bool:
    """A run is complete iff metrics.csv's last logged epoch equals the
    requested total -- existence alone isn't a safe signal, since train.py
    opens and flushes metrics.csv incrementally starting partway through a
    run, so a run killed by --timeout (or a crash) leaves a real but partial
    file. model.pth (written once, at true completion) would also work as a
    signal, but it's gitignored and never committed (see this file's own
    README's "Not committed" policy) -- checking it would make skip-existing
    silently stop working for every already-committed historical run in a
    fresh checkout, which is the far more common case than a killed run."""
    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.exists():
        return False
    try:
        with metrics_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return False
    if len(rows) < 2:  # header only, or empty -- killed before the first flush
        return False
    try:
        last_epoch = int(rows[-1][0])
    except (ValueError, IndexError):
        return False
    return last_epoch == expected_epochs


def build_command(lr: float, epochs: int, seed: int, l2sp_lambda: float, out_dir: Path) -> list[str]:
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
        "--l2sp-lambda", str(l2sp_lambda),
    ]


def run_name(lr: float, epochs: int, seed: int, l2sp_lambda: float) -> str:
    # Suffix only when l2sp_lambda deviates from the off-default, so existing
    # lr/seed-only sweep directories (e.g. lr0.0001_ep100_seed42) keep their
    # names and are correctly recognized as already-run by --overwrite's
    # skip-existing check.
    suffix = f"_l2sp{l2sp_lambda}" if l2sp_lambda != 0.0 else ""
    return f"lr{lr}_ep{epochs}_seed{seed}{suffix}"


def get_git_state(cwd: Path) -> dict[str, object]:
    """Best-effort git commit + dirty-tree snapshot, so a manifest entry can
    always be tied to the exact code that produced it, not just the command."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty_out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=10
        ).stdout
        return {"commit": commit or None, "dirty": bool(dirty_out.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lrs", type=float, nargs="+", required=True, help="Learning rates to sweep")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seeds to run at each lr")
    parser.add_argument(
        "--l2sp-lambdas", type=float, nargs="+", default=[0.0],
        help="L2-SP lambda values to sweep, crossed with --lrs and --seeds (default: [0.0], i.e. off)",
    )
    parser.add_argument("--out-root", type=Path, default=Path("phonebook/sweep_lr/seeded"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite a run whose output directory already has a metrics.csv "
        "(default: skip it, so re-running a sweep never silently clobbers prior results)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-run timeout in seconds (default: 300). A run at this scale normally "
        "finishes in well under a minute; a hang almost certainly means new code has a bug.",
    )
    args = parser.parse_args()

    if any(l2sp_lambda < 0 for l2sp_lambda in args.l2sp_lambdas):
        parser.error("--l2sp-lambdas values must all be >= 0")

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "manifest.jsonl"
    git_state = get_git_state(ROOT_DIR)

    combos = list(itertools.product(args.lrs, args.seeds, args.l2sp_lambdas))
    print(
        f"Planned runs: {len(combos)} ({len(args.lrs)} lr x {len(args.seeds)} seeds x "
        f"{len(args.l2sp_lambdas)} l2sp-lambdas, {args.epochs} epochs each)"
    )
    if git_state["dirty"]:
        print("WARNING: working tree has uncommitted changes -- results below won't be tied to a clean commit.")

    for i, (lr, seed, l2sp_lambda) in enumerate(combos, start=1):
        name = run_name(lr, args.epochs, seed, l2sp_lambda)
        out_dir = args.out_root / name
        cmd = build_command(lr, args.epochs, seed, l2sp_lambda, out_dir)

        print(f"\n[{i}/{len(combos)}] {' '.join(cmd)}")

        if not args.overwrite and is_run_complete(out_dir, args.epochs):
            print(f"  SKIPPED -- {out_dir} already has results (pass --overwrite to re-run it)")
            continue
        if args.dry_run:
            continue

        start = time.perf_counter()
        timed_out = False
        try:
            result = subprocess.run(
                cmd, cwd=ROOT_DIR, capture_output=True, text=True, timeout=args.timeout
            )
            returncode = result.returncode
            stderr_tail = result.stderr[-2000:]
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            stderr_tail = f"TIMEOUT after {args.timeout}s"
        elapsed = time.perf_counter() - start

        record = {
            "name": name,
            "command": cmd,
            "cwd": str(ROOT_DIR),
            "lr": lr,
            "epochs": args.epochs,
            "seed": seed,
            "l2sp_lambda": l2sp_lambda,
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_sec": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_state["commit"],
            "git_dirty": git_state["dirty"],
        }
        with manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        if timed_out:
            print(f"  TIMED OUT after {args.timeout}s -- treating as a failure, moving on")
        elif returncode != 0:
            print(f"  FAILED (exit {returncode}) in {elapsed:.1f}s")
            print(stderr_tail)
        else:
            print(f"  OK in {elapsed:.1f}s -> {out_dir}")

    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
