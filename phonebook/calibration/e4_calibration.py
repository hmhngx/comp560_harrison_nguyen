"""E4 calibration: how many random entries can a 2-layer, 2-head model of width d memorize in a given number of epochs?

This is scratch work that fixed the E4 ladders and informed the locked predictions. It is NOT evidence: it runs on a
throwaway book seed (100, not an experiment seed) with training seeds 100 and 101 and is not part of any audited spec.
Same recipe as the E4 specs: from scratch, lr 0.001, batch 64, accuracy logged every 10 epochs, isolated loader stream,
one torch thread, validation on the held-out control book.

    python phonebook/calibration/e4_calibration.py books --work WORK
    python phonebook/calibration/e4_calibration.py run --work WORK --cells 8:10:1000,64:453:1000:101 --workers 7
    python phonebook/calibration/e4_calibration.py summary --results phonebook/calibration/e4_calibration_results.jsonl

A cell is D:N:EPOCHS[:TRAINING_SEED] (width, entries, epochs). A cell that fails is reported and makes `run` exit 1;
failed cells are never written to the results file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.analyze_experiment import load_metrics
from phonebook.make_capacity_books import doubling_ladder, write_capacity_inputs

BOOK_SEED = 100
N_MAX = 5120
EXTRA_RUNGS = (14, 28, 57, 113, 226, 453, 905)       # rungs between the doubling ladder, sliced from the top book
DEFAULT_WORK = Path(tempfile.gettempdir()) / "phonebook_calibration"


def book_path(inputs: Path, n: int, seed: int = BOOK_SEED) -> Path:
    return inputs / f"capacity_seed{seed}_n{n}.txt"


def write_calibration_books(inputs: Path, seed: int = BOOK_SEED, n_max: int = N_MAX, extras: tuple = EXTRA_RUNGS) -> list[int]:
    """The doubling ladder 10 .. n_max plus the control book, then the extra rungs as prefixes of the top book."""
    ladder = doubling_ladder(n_max)
    write_capacity_inputs(inputs, seed, ladder)
    top = book_path(inputs, ladder[-1], seed).read_text(encoding="ascii").splitlines()
    for n in extras:
        if n in ladder or not 1 <= n < len(top):
            raise ValueError(f"extra rung {n} is already on the ladder or outside the top book")
        path = book_path(inputs, n, seed)
        if path.exists():
            raise ValueError(f"{path} already exists; refusing to overwrite")
        path.write_bytes(("\n".join(top[:n]) + "\n").encode("ascii"))
    return sorted(set(ladder) | set(extras))


def data_dir(work: Path, n: int, book_seed: int = BOOK_SEED) -> Path:
    out = work / "data" / f"s{book_seed}_n{n}"
    if not (out / "train.bin").exists():
        subprocess.run([sys.executable, "phonebook/prepare_phonebook.py", "--train-input", str(book_path(work / "inputs", n, book_seed)),
                        "--eval-input", str(work / "inputs" / f"capacity_seed{book_seed}_control.txt"), "--out-dir", str(out)],
                       cwd=ROOT_DIR, check=True, capture_output=True, text=True)
    return out


def run_cell(work: Path, d: int, n: int, epochs: int, seed: int = 100, interval: int = 10, book_seed: int = BOOK_SEED) -> dict:
    out = work / "runs" / f"d{d}_n{n}_e{epochs}_s{seed}"
    cmd = [sys.executable, "train.py", "phonebook/config/phonebook.py", "--device", "cpu", "--data-dir", str(data_dir(work, n, book_seed)),
           "--out-dir", str(out), "--seed", str(seed), "--lr", "0.001", "--epochs", str(epochs), "--batch-size", "64",
           "--accuracy-interval", str(interval), "--l2sp-lambda", "0.0", "--embedding-dim", str(d), "--n-heads", "2", "--n-layers", "2"]
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.perf_counter()
    done = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - started
    if done.returncode != 0:
        return {"d": d, "n": n, "epochs": epochs, "error": done.stderr[-300:]}
    rows = load_metrics(out / "metrics.csv")
    acc = [(r["epoch"], r["Train Seq Acc"], r["Val Seq Acc"]) for r in rows if r["Train Seq Acc"] is not None]
    first = next((e for e, train, _ in acc if abs(train - 1.0) < 1e-9), None)
    return {"d": d, "n": n, "epochs": epochs, "seed": seed, "params": int(rows[0]["Param Count"]), "first_perfect": first,
            "final_train_seq": acc[-1][1], "best_train_seq": max(a[1] for a in acc), "max_control": max(a[2] for a in acc),
            "seconds": round(elapsed, 1), "curve": [[e, round(train, 4)] for e, train, _ in acc]}


def parse_cells(text: str) -> list[tuple]:
    cells = []
    for item in text.split(","):
        parts = [int(x) for x in item.split(":")]
        if len(parts) not in (3, 4):
            raise ValueError(f"cell {item!r} must be D:N:EPOCHS or D:N:EPOCHS:SEED")
        cells.append(tuple(parts))
    return cells


def run_batch(work: Path, cells: list[tuple], workers: int, results: Path, runner=run_cell) -> int:
    """Run the cells, append each success to `results`, return how many failed (reported on stdout)."""
    failed = []

    def go(cell):
        d, n, epochs = cell[:3]
        try:
            row = runner(work, d, n, epochs, seed=cell[3] if len(cell) > 3 else 100)
        except Exception as error:                      # a crash is a failure, never a silent skip
            row = {"error": repr(error)}
        if "error" in row:
            failed.append(cell)
            print(f"FAILED cell {cell}: {row['error']}", flush=True)
            return
        with results.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(row) + "\n")
        tag = f"perfect@{row['first_perfect']}"
        print(f"d={d} n={n} e={epochs} -> {tag}, final train {row['final_train_seq']:.3f}, control max {row['max_control']:.2f}, {row['seconds']}s", flush=True)

    results.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(go, cells))
    return len(failed)


def collapse(rows: list[dict]) -> dict:
    """The longest run per (width, entries, training seed): a 300-epoch run is a prefix of the 1000-epoch run of the same cell."""
    best: dict = {}
    for row in rows:
        key = (row["d"], row["n"], row["seed"])
        if key not in best or row["epochs"] > best[key]["epochs"]:
            best[key] = row
    return best


def summary(rows: list[dict]) -> str:
    best = collapse(rows)
    lines = ["first epoch with 100% exact recall of all n entries; no(x) = never, x = accuracy at the last epoch; * = only run for 300 epochs"]
    for d in sorted({k[0] for k in best}):
        parts = []
        for n in sorted({k[1] for k in best if k[0] == d}):
            vals = []
            for key in sorted(k for k in best if k[0] == d and k[1] == n):
                row = best[key]
                text = str(row["first_perfect"]) if row["first_perfect"] else "no(%.2f)" % row["final_train_seq"]
                vals.append(text + ("" if row["epochs"] == 1000 else "*"))
            parts.append(f"n={n}: {'/'.join(vals)}")
        lines.append(f"d={d:<2}  " + " | ".join(parts))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    books = sub.add_parser("books", help="write the throwaway calibration books")
    books.add_argument("--work", type=Path, default=DEFAULT_WORK)
    run = sub.add_parser("run", help="train cells and append their results")
    run.add_argument("--work", type=Path, default=DEFAULT_WORK)
    run.add_argument("--cells", required=True)
    run.add_argument("--workers", type=int, default=7)
    run.add_argument("--results", type=Path, help="default: WORK/results.jsonl")
    summ = sub.add_parser("summary", help="print the first-perfect table of a results file")
    summ.add_argument("--results", type=Path, default=Path(__file__).with_name("e4_calibration_results.jsonl"))
    args = parser.parse_args(argv)
    try:
        if args.command == "books":
            rungs = write_calibration_books(args.work / "inputs")
            print(f"wrote {len(rungs)} rungs and the control book to {args.work / 'inputs'}")
        elif args.command == "run":
            failed = run_batch(args.work, parse_cells(args.cells), args.workers, args.results or args.work / "results.jsonl")
            print(f"{failed} cell(s) failed")
            return 1 if failed else 0
        else:
            print(summary([json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]))
    except ValueError as error:
        sys.exit(f"error: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
