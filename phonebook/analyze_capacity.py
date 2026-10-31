"""Capacity analysis: from the trainer's own logs to cells, per-budget capacities and bits per parameter.

A capacity run trains from scratch on one book of N random entries and holds out a small control book.
In the logged curves A is the number of training entries recalled exactly and B the number of control
entries recalled exactly (expected to stay 0: nothing in a random number can be guessed from a name).

    python phonebook/analyze_capacity.py --root ROOT [--root ROOT ...] --budgets 100 300 1000 --out phonebook/results --prefix e4

Rules (fixed before any capacity run):
- A cell is *perfect by budget E* if every training entry is recalled exactly at some logged epoch <= E.
- A rung N *passes* at budget E if at least 2 of its 3 book seeds are perfect by E.
- The capacity at budget E is the largest rung such that it and every smaller rung pass. A pass above
  the first failing rung is reported but never counted.
- The boundary is *stable* if the capacity at the largest budget equals the capacity at the one before;
  otherwise it is still *moving* and is a budget limit, not a capacity. If no rung failed at the largest
  budget the ladder ended before the boundary: that is a *ceiling*, a lower bound on the capacity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.analyze_experiment import load_metrics
from phonebook.report_experiments import load_runs

# create_phonebook.py draws each number from 800 * 800 * 10_000 = 6.4e9 equally likely values.
BITS_PER_ENTRY = math.log2(800 * 800 * 10_000)
SCENARIO = re.compile(r"s(\d+)_n(\d+)")


# ------------------------------------------------------------------ one run


def first_perfect_epoch(points: list[dict], n: int) -> int | None:
    return next((point["epoch"] for point in points if point["A"] == n), None)


def cell_summary(points: list[dict], n: int, budgets: tuple[int, ...]) -> dict:
    first = first_perfect_epoch(points, n)
    by_epoch = {point["epoch"]: point["A"] for point in points}
    max_control = max((point["B"] for point in points), default=0)
    return {
        "first_perfect": first,
        "perfect_by": {budget: first is not None and first <= budget for budget in budgets},
        "recall_at": {budget: by_epoch.get(budget) for budget in budgets},
        "final": points[-1]["A"] if points else None,
        "perfect_at_end": bool(points) and points[-1]["A"] == n,
        "max_control": max_control,
        "control_ok": max_control == 0,
    }


# ------------------------------------------------------------------ capacity rule


def capacity(cells: list[dict], *, budget: int, min_seeds: int = 2) -> dict:
    """cells: dicts with n, seed and first_perfect (epoch or None) for one model."""
    by_rung: dict[int, list[dict]] = {}
    for cell in cells:
        by_rung.setdefault(cell["n"], []).append(cell)
    for n, group in by_rung.items():
        if len({cell["seed"] for cell in group}) != len(group):
            raise ValueError(f"duplicate runs for one seed at rung {n}")
    seed_sets = {frozenset(cell["seed"] for cell in group) for group in by_rung.values()}
    if len(seed_sets) != 1:
        raise ValueError("every rung must have the same seeds")
    seeds = len(next(iter(seed_sets)))
    if min_seeds > seeds:
        raise ValueError(f"min_seeds {min_seeds} exceeds the {seeds} seeds per rung")

    rungs = {}
    for n in sorted(by_rung):
        passed = sum(cell["first_perfect"] is not None and cell["first_perfect"] <= budget for cell in by_rung[n])
        rungs[n] = {"passed": passed, "seeds": seeds, "ok": passed >= min_seeds}
    best, first_fail = None, None
    for n, rung in rungs.items():
        if first_fail is not None:
            continue
        if rung["ok"]:
            best = n
        else:
            first_fail = n
    isolated = [n for n, rung in rungs.items() if first_fail is not None and n > first_fail and rung["ok"]]
    return {"budget": budget, "capacity": best, "first_failing_rung": first_fail,
            "isolated_passes_above": isolated, "rungs": rungs}


def boundary_status(capacity_by_budget: dict, *, top_rung_passes: bool = False) -> str:
    """top_rung_passes: every rung passed at the largest budget, so no boundary was observed."""
    budgets = sorted(capacity_by_budget)
    if len(budgets) < 2:
        raise ValueError("boundary_status needs at least two budgets")
    early, late = capacity_by_budget[budgets[-2]], capacity_by_budget[budgets[-1]]
    if late is None:
        return "none"
    if top_rung_passes:
        return "ceiling"
    return "stable" if early == late else "moving"


def bits_per_param(entries: int | None, params: float) -> float | None:
    return None if entries is None else entries * BITS_PER_ENTRY / params


# ------------------------------------------------------------------ evidence


def load_capacity_cells(root: Path, budgets: tuple[int, ...]) -> list[dict]:
    root = Path(root)
    cells = []
    for run in load_runs(root, budgets=()):
        match = SCENARIO.fullmatch(run["scenario"])
        if match is None:
            raise ValueError(f"scenario id {run['scenario']!r} does not follow s<seed>_n<size>")
        seed, n = int(match.group(1)), int(match.group(2))
        if run["n_a"] != n:
            raise ValueError(f"{run['id']}: scenario id says {n} entries but the book size is {run['n_a']}")
        if "model" not in run:
            raise ValueError(f"{run['id']}: a capacity run must declare its model")
        too_long = [budget for budget in budgets if budget > run["epochs"]]
        if too_long:
            raise ValueError(f"{run['id']}: budget {too_long[0]} exceeds the {run['epochs']} epochs trained")
        rows = load_metrics(root / run["scenario"] / run["condition"] / "metrics.csv")
        counts = {int(row["Param Count"]) for row in rows}
        if len(counts) != 1:
            raise ValueError(f"{run['id']}: the logged parameter count changes within the run")
        model = run["model"]
        cells.append({"model_d": model["embedding_dim"], "n_layers": model["n_layers"], "n_heads": model["n_heads"],
                      "seed": seed, "n": n, "params": counts.pop(), "epochs": run["epochs"],
                      **cell_summary(run["points"], n, budgets)})
    return cells


def cell_rows(cells: list[dict], budgets: tuple[int, ...]) -> list[dict]:
    rows = []
    for cell in sorted(cells, key=lambda c: (c["model_d"], c["seed"], c["n"])):
        row = {key: cell[key] for key in ("model_d", "n_layers", "n_heads", "seed", "n", "params", "epochs")}
        row["first_perfect"] = "" if cell["first_perfect"] is None else cell["first_perfect"]
        for budget in budgets:
            row[f"perfect_by_{budget}"] = cell["perfect_by"][budget]
            row[f"recall_at_{budget}"] = "" if cell["recall_at"][budget] is None else cell["recall_at"][budget]
        row.update({key: cell[key] for key in ("final", "perfect_at_end", "max_control", "control_ok")})
        rows.append(row)
    return rows


def summarize_models(cells: list[dict], budgets: tuple[int, ...], min_seeds: int = 2) -> dict:
    report = {}
    for d in sorted({cell["model_d"] for cell in cells}):
        mine = [cell for cell in cells if cell["model_d"] == d]
        per_budget = {budget: capacity(mine, budget=budget, min_seeds=min_seeds) for budget in budgets}
        caps = {budget: result["capacity"] for budget, result in per_budget.items()}
        top = caps[max(budgets)]
        ceiling = top is not None and per_budget[max(budgets)]["first_failing_rung"] is None
        params_at_top = (statistics.median(c["params"] for c in mine if c["n"] == top) if top is not None
                         else statistics.median(c["params"] for c in mine))
        report[d] = {"params": params_at_top,
                     "capacity_by_budget": caps,
                     "detail": per_budget,
                     "boundary": boundary_status(caps, top_rung_passes=ceiling),
                     "capacity_is_lower_bound": ceiling,
                     "bits_per_param_at_largest_budget": bits_per_param(top, params_at_top),
                     "control_violations": sum(not c["control_ok"] for c in mine)}
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", required=True, help="evidence folder from run_experiment.py")
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="e4")
    parser.add_argument("--min-seeds", type=int, default=2, help="book seeds that must be perfect for a rung to pass")
    args = parser.parse_args(argv)
    budgets = tuple(sorted(set(args.budgets)))
    cells = [cell for root in args.root for cell in load_capacity_cells(root, budgets)]
    if not cells:
        raise ValueError("no capacity runs found in the evidence folders")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = cell_rows(cells, budgets)
    with (args.out / f"{args.prefix}_cells.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = summarize_models(cells, budgets, args.min_seeds)
    (args.out / f"{args.prefix}_capacity.json").write_bytes(
        (json.dumps(report, indent=2, allow_nan=False, default=str) + "\n").encode("utf-8"))
    for d, entry in report.items():
        print(f"d={d}: params {entry['params']:.0f}, capacity {entry['capacity_by_budget']}, boundary {entry['boundary']}")


if __name__ == "__main__":
    main()
