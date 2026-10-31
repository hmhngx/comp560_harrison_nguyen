"""A second computation of the headline numbers of E4 and E5, straight from the evidence files.

It reads only protocol.json and each run's metrics.csv and applies the rules written in
make_capacity_spec.py and make_e5_specs.py. It shares no code with the analysis modules, so a bug in one is
unlikely to be repeated in the other. It is written by the same author as those modules, not by a third party.

    python phonebook/independent_check.py e4 --root R [--root R ...] --budgets 100 300 1000 --against phonebook/results/e4_capacity.json
    python phonebook/independent_check.py e5 --root R [--root R ...] --lrs 0.001 0.003 0.0003 --primary-lr 0.001 \\
        --ks 1 2 3 4 5 9 --phonebooks 2 4 7 8 9 12 --against phonebook/results/e5_analysis.json

Exit status 1 and a list of differences if the two computations disagree.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

INF = math.inf
N_BOOK = 10


def _rows(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _count(text: str, total: int) -> int:
    value = float(text)
    count = round(value * total)
    if abs(value * total - count) > 5.1e-5 * total:
        raise ValueError(f"{text} is not a whole number of {total} entries")
    return count


def _json_number(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


# ------------------------------------------------------------------ E4


def e4(roots, budgets) -> dict:
    """{embedding width: {budget: capacity}}; a rung passes with 2 of its seeds perfect by the budget."""
    firsts: dict = {}
    for root in roots:
        protocol = json.loads((Path(root) / "protocol.json").read_text(encoding="utf-8"))
        for run in protocol["runs"]:
            too_long = [budget for budget in budgets if budget > run["epochs"]]
            if too_long:
                raise ValueError(f"{run['id']}: budget {too_long[0]} exceeds the {run['epochs']} epochs trained")
            seed_text, n_text = run["scenario"][1:].split("_n")
            seed, n = int(seed_text), int(n_text)
            first = None
            for row in _rows(Path(root) / run["scenario"] / run["condition"] / "metrics.csv"):
                if row["Train Seq Acc"] != "N/A" and _count(row["Train Seq Acc"], n) == n:
                    first = int(row["Epoch"])
                    break
            firsts.setdefault(run["model"]["embedding_dim"], {}).setdefault(n, {})[seed] = first
    result = {}
    for d, rungs in sorted(firsts.items()):
        result[d] = {}
        for budget in budgets:
            capacity, stopped = None, False
            for n in sorted(rungs):
                passes = sum(1 for first in rungs[n].values() if first is not None and first <= budget) >= 2
                if stopped:
                    continue
                if passes:
                    capacity = n
                else:
                    stopped = True
            result[d][budget] = capacity
    return result


def compare_e4(mine: dict, analysis: dict) -> list[str]:
    differences = []
    for d, by_budget in mine.items():
        if str(d) not in analysis:
            differences.append(f"model d={d} is missing from the analysis")
            continue
        theirs = analysis[str(d)]["capacity_by_budget"]
        for budget, capacity in by_budget.items():
            if theirs.get(str(budget)) != capacity:
                differences.append(f"model d={d} budget {budget}: independent {capacity}, analysis {theirs.get(str(budget))}")
    return differences


# ------------------------------------------------------------------ E5


def _stable_and_dip(rows: list[dict], a_copies: int, b_copies: int):
    epochs = []
    for row in rows:
        if row["Train Seq Acc"] == "N/A":
            continue
        a = _count(row["Val Seq Acc"], N_BOOK)
        train = _count(row["Train Seq Acc"], (a_copies + b_copies) * N_BOOK)
        if (train - a_copies * a) % b_copies:
            raise ValueError("training count is not divisible by the number of copies of B")
        epochs.append((int(row["Epoch"]), a, (train - a_copies * a) // b_copies))
    stable = None
    for epoch, a, b in reversed(epochs):
        if a != N_BOOK or b != N_BOOK:
            break
        stable = epoch
    return stable, min(a for _, a, _ in epochs)


def _cells(roots) -> list[dict]:
    cells = []
    for root in roots:
        protocol = json.loads((Path(root) / "protocol.json").read_text(encoding="utf-8"))
        for run in protocol["runs"]:
            name, init = run["scenario"].split("_s")
            a_copies, b_copies = run["train"]["a_copies"], run["train"]["b_copies"]
            stable, dip = _stable_and_dip(_rows(Path(root) / run["scenario"] / run["condition"] / "metrics.csv"),
                                          a_copies, b_copies)
            cells.append({"pb": int(name[2:]), "init": int(init), "seed": protocol["spec"]["seed"], "lr": run["lr"],
                          "k": None if a_copies == 0 else b_copies, "stable": INF if stable is None else stable})
    return cells


def e5(roots, *, lrs, primary_lr, ks, phonebooks, baseline_k=1, main_seed=42, min_phonebooks=5) -> dict:
    cells = _cells(roots)
    if sorted({cell["pb"] for cell in cells}) != sorted(phonebooks):
        raise ValueError("the evidence does not cover exactly the requested phonebooks")

    spreads = []
    groups: dict = {}
    for cell in cells:
        groups.setdefault((cell["pb"], cell["init"], cell["lr"], cell["k"]), []).append(cell["stable"])
    for values in groups.values():
        if len(values) >= 2:
            if INF in values:
                raise ValueError("a censored run in the noise arm has an unbounded spread")
            spreads.append(max(values) - min(values))
    if not spreads:
        raise ValueError("no configuration was run under two shuffle seeds")
    band = max(spreads)

    result = {"noise_band": band, "lr": {}}
    for lr in lrs:
        per_unit: dict = {}
        for cell in cells:
            if cell["lr"] == lr and cell["seed"] == main_seed and cell["k"] is not None:
                per_unit.setdefault((cell["pb"], cell["k"]), []).append(cell["stable"])
        medians = {key: statistics.median(values) for key, values in per_unit.items()}
        across = {k: statistics.median([medians[(pb, k)] for pb in phonebooks]) for k in ks}
        paired = {}
        for k in ks:
            if k == baseline_k:
                continue
            gains = []
            for pb in phonebooks:
                base, value = medians[(pb, baseline_k)], medians[(pb, k)]
                gains.append(0 if base == value == INF else INF if base == INF else -INF if value == INF else base - value)
            faster = sum(g > 0 for g in gains)
            median_gain = statistics.median(gains)
            paired[str(k)] = {"faster": faster, "tied": sum(g == 0 for g in gains), "slower": sum(g < 0 for g in gains),
                              "median_gain": _json_number(median_gain),
                              "beats": (faster >= min_phonebooks and median_gain > band) if lr == primary_lr else None}
        best = min(across.values())
        work = {k: across[k] * (N_BOOK + k * N_BOOK) for k in ks}
        censored = {str(k): sum(1 for c in cells if c["lr"] == lr and c["seed"] == main_seed and c["k"] == k and c["stable"] == INF)
                    for k in ks}
        result["lr"][str(lr)] = {
            "median_stable_joint": {str(k): _json_number(across[k]) for k in ks},
            "paired": paired,
            "plateau": ([] if best == INF else sorted(k for k in ks if across[k] <= best + band)) if lr == primary_lr else None,
            "cheapest_k_in_work": min(ks, key=lambda k: (work[k], k)),
            "censored": censored,
        }
    return result


def compare_e5(mine: dict, theirs: dict) -> list[str]:
    differences = []

    def check(label, independent, analysis):
        if independent != analysis:
            differences.append(f"{label}: independent {independent}, analysis {analysis}")

    check("noise band", mine["noise_band"], theirs["noise"]["band"])
    for lr, entry in mine["lr"].items():
        other = theirs["lr"].get(lr)
        if other is None:
            differences.append(f"lr {lr} is missing from the analysis")
            continue
        for k, value in entry["median_stable_joint"].items():
            check(f"lr {lr} k {k} median stable_joint", value, other["median_stable_joint"][k])
        for k, fields in entry["paired"].items():
            for name, value in fields.items():
                check(f"lr {lr} k {k} paired {name}", value, other["paired"][k][name])
        check(f"lr {lr} plateau", entry["plateau"], other["plateau"])
        check(f"lr {lr} cheapest k in work", entry["cheapest_k_in_work"], other["cheapest_k_in_work"])
        for k, count in entry["censored"].items():
            check(f"lr {lr} k {k} censored runs", count, other["censored"][k])
    return differences


# ------------------------------------------------------------------ command line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("e4")
    first.add_argument("--budgets", type=int, nargs="+", required=True)
    second = sub.add_parser("e5")
    second.add_argument("--lrs", type=float, nargs="+", required=True)
    second.add_argument("--primary-lr", type=float, required=True)
    second.add_argument("--ks", type=int, nargs="+", required=True)
    second.add_argument("--phonebooks", type=int, nargs="+", required=True)
    for command in (first, second):
        command.add_argument("--root", type=Path, action="append", required=True)
        command.add_argument("--against", type=Path, required=True, help="analysis JSON written by the analysis module")
    args = parser.parse_args()
    analysis = json.loads(args.against.read_text(encoding="utf-8"))
    if args.command == "e4":
        differences = compare_e4(e4(args.root, tuple(args.budgets)), analysis)
    else:
        differences = compare_e5(e5(args.root, lrs=args.lrs, primary_lr=args.primary_lr, ks=args.ks,
                                    phonebooks=args.phonebooks), analysis)
    for line in differences:
        print(line)
    print("agree" if not differences else f"{len(differences)} difference(s)")
    sys.exit(1 if differences else 0)


if __name__ == "__main__":
    main()
