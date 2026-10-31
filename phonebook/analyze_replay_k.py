"""Replay-ratio analysis (E5): with one copy of A and k copies of B, which k reaches exact recall of
both books soonest, and is the difference larger than run-to-run noise?

    python phonebook/analyze_replay_k.py --root ROOT [--root ROOT ...] --lrs 0.001 0.003 0.0003 --primary-lr 0.001 \\
        --ks 1 2 3 4 5 9 --phonebooks 2 4 7 8 9 12 --out phonebook/results --prefix e5

Rules (fixed before any E5 run):
- stable_joint (E3's definition) is the first update from which every logged update has both books recalled
  exactly; a run that never gets there is censored and counts as infinitely slow.
- The unit of replication is the phonebook: the three Phase-A inits of one phonebook are reduced to their
  median first, and every comparison is paired within a phonebook against k = 1.
- The noise band is the largest spread of stable_joint among runs that differ only in the shuffle seed.
  It comes from single runs, so it is conservative for medians of three inits.
- k beats k = 1 only at the primary learning rate, only if it is faster in at least `min_phonebooks`
  phonebooks and its median paired gain exceeds the noise band. Other learning rates are descriptive.
- The answer to "which k is fastest" is the plateau: every k whose median is within the noise band of the best.
- Work is updates times sequences per update, 10 of A plus 10 per copy of B.
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

from phonebook.report_experiments import load_runs

INF = math.inf
SCENARIO = re.compile(r"pb(\d+)_s(\d+)")
N_A = N_B = 10


def work_sequences(updates: int | None, k: int, n_a: int = N_A, n_b: int = N_B) -> int | None:
    return None if updates is None else updates * (n_a + k * n_b)


def _slow(value):
    return INF if value is None else value


# ------------------------------------------------------------------ tables


def phonebook_medians(cells: list[dict], *, lr: float, seed: int, field: str = "stable_joint") -> dict:
    """{phonebook: {k: median over its Phase-A inits}} for one learning rate and one shuffle seed."""
    seen, groups = set(), {}
    for cell in cells:
        if cell["lr"] != lr or cell["seed"] != seed or cell["plain"]:
            continue
        key = (cell["phonebook"], cell["init"], cell["k"])
        if key in seen:
            raise ValueError(f"duplicate run for phonebook {key[0]}, init {key[1]}, k {key[2]}")
        seen.add(key)
        groups.setdefault((cell["phonebook"], cell["k"]), []).append(_slow(cell[field]))
    table: dict = {}
    for (phonebook, k), values in groups.items():
        table.setdefault(phonebook, {})[k] = statistics.median(values)
    return {phonebook: dict(sorted(row.items())) for phonebook, row in sorted(table.items())}


def paired_gains(table: dict, k: int, baseline_k: int = 1) -> dict:
    """Updates saved by k against the baseline, within each phonebook (positive means k is faster)."""
    gains = {}
    for phonebook, row in table.items():
        if baseline_k not in row or k not in row:
            raise ValueError(f"phonebook {phonebook} lacks k={baseline_k} or k={k}")
        base, value = row[baseline_k], row[k]
        if base == INF and value == INF:
            gains[phonebook] = 0
        elif base == INF:
            gains[phonebook] = INF
        elif value == INF:
            gains[phonebook] = -INF
        else:
            gains[phonebook] = base - value
    return gains


def sign_counts(gains) -> dict:
    gains = list(gains)
    return {"faster": sum(g > 0 for g in gains), "tied": sum(g == 0 for g in gains), "slower": sum(g < 0 for g in gains)}


def beats_baseline(gains, *, band: float, min_phonebooks: int) -> dict:
    gains = list(gains)
    counts = sign_counts(gains)
    median = statistics.median(gains)
    return {**counts, "median_gain": median, "band": band,
            "beats": counts["faster"] >= min_phonebooks and median > band}


def noise_band(cells: list[dict]) -> dict:
    groups: dict = {}
    for cell in cells:
        key = (cell["phonebook"], cell["init"], cell["lr"], cell["k"], cell["plain"])
        groups.setdefault(key, []).append(cell)
    rows = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        seeds = [cell["seed"] for cell in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate shuffle seed in noise group {key}")
        if len(group) < 2:
            continue
        if any(cell["stable_joint"] is None for cell in group):
            raise ValueError(f"censored run in noise group {key}: its spread is unbounded")
        values = [cell["stable_joint"] for cell in group]
        rows.append({"phonebook": key[0], "init": key[1], "lr": key[2], "k": key[3], "plain": key[4],
                     "seeds": len(group), "min": min(values), "max": max(values), "spread": max(values) - min(values)})
    if not rows:
        raise ValueError("no configuration has two shuffle seeds, so there is no noise to measure")
    return {"band": max(row["spread"] for row in rows), "groups": len(rows), "spread_rows": rows}


# ------------------------------------------------------------------ conclusions


def plateau(medians: dict, *, band: float) -> list:
    best = min(medians.values())
    if best == INF:
        return []
    return sorted(k for k, value in medians.items() if value <= best + band)


def cheapest_k(work_by_k: dict):
    return min(work_by_k, key=lambda k: (work_by_k[k], k))


def dip_is_monotone(min_a_by_k: dict) -> bool:
    ordered = [min_a_by_k[k] for k in sorted(min_a_by_k)]
    return all(earlier >= later for earlier, later in zip(ordered, ordered[1:]))


def analyze(cells: list[dict], *, lrs: list[float], primary_lr: float, ks: list[int], phonebooks: list[int],
            baseline_k: int = 1, main_seed: int = 42, min_phonebooks: int = 5) -> dict:
    expected = sorted(phonebooks)
    present = sorted({cell["phonebook"] for cell in cells})
    if present != expected:
        raise ValueError(f"expected phonebooks {expected}, found {present}")
    if primary_lr not in lrs or baseline_k not in ks:
        raise ValueError("the primary learning rate and the baseline k must be among those analysed")
    noise = noise_band(cells)
    result: dict = {"noise": noise, "primary_lr": primary_lr, "baseline_k": baseline_k, "lr": {}}
    for lr in lrs:
        table = phonebook_medians(cells, lr=lr, seed=main_seed)
        dip = phonebook_medians(cells, lr=lr, seed=main_seed, field="min_a")
        if sorted(table) != expected:
            raise ValueError(f"lr {lr}: expected phonebooks {expected}, found {sorted(table)}")
        for phonebook, row in table.items():
            missing = [k for k in ks if k not in row]
            if missing:
                raise ValueError(f"lr {lr}, phonebook {phonebook}: no runs for k={missing}")
        stable = {k: statistics.median([table[p][k] for p in expected]) for k in ks}
        work = {k: stable[k] * (N_A + k * N_B) for k in ks}
        min_a = {k: statistics.median([dip[p][k] for p in expected]) for k in ks}
        censored = {k: sum(1 for c in cells if c["lr"] == lr and c["seed"] == main_seed and not c["plain"]
                           and c["k"] == k and c["stable_joint"] is None) for k in ks}
        paired = {}
        for k in ks:
            if k == baseline_k:
                continue
            gains = list(paired_gains(table, k, baseline_k).values())
            if lr == primary_lr:
                paired[k] = beats_baseline(gains, band=noise["band"], min_phonebooks=min_phonebooks)
            else:
                paired[k] = {**sign_counts(gains), "median_gain": statistics.median(gains), "band": None, "beats": None}
        result["lr"][lr] = {
            "phonebook_medians": table,
            "median_stable_joint": stable,
            "median_work_sequences": work,
            "median_min_a": min_a,
            "censored": censored,
            "paired": paired,
            "plateau": plateau(stable, band=noise["band"]) if lr == primary_lr else None,
            "cheapest_k_in_work": cheapest_k(work),
            "dip_monotone": dip_is_monotone(min_a),
        }
    return result


# ------------------------------------------------------------------ evidence and reports


def load_replay_cells(root: Path) -> list[dict]:
    root = Path(root)
    seed = json.loads((root / "protocol.json").read_text(encoding="utf-8"))["spec"]["seed"]
    cells = []
    for run in load_runs(root):
        match = SCENARIO.fullmatch(run["scenario"])
        if match is None:
            raise ValueError(f"scenario id {run['scenario']!r} does not follow pb<phonebook>_s<init>")
        a, b = run["train"]["a_copies"], run["train"]["b_copies"]
        if a not in (0, 1):
            raise ValueError(f"{run['id']}: {a} copies of A; the experiment uses one copy, or none for plain fine-tuning")
        if a == 0 and b != 1:
            raise ValueError(f"{run['id']}: plain fine-tuning means B once, got {b} copies")
        summary = run["summary"]
        cells.append({"phonebook": int(match.group(1)), "init": int(match.group(2)), "seed": seed, "lr": run["lr"],
                      "k": b if a == 1 else None, "plain": a == 0, "condition": run["condition"],
                      "stable_joint": summary["stable_joint"], "first_joint": summary["first_joint"],
                      "min_a": summary["min_a"], "epoch_of_min_a": summary["epoch_of_min_a"],
                      "final_A": summary["final"]["A"], "final_B": summary["final"]["B"]})
    return cells


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


CELL_COLUMNS = ["phonebook", "init", "seed", "lr", "k", "plain", "condition", "stable_joint", "first_joint",
                "min_a", "epoch_of_min_a", "final_A", "final_B"]


def write_report(cells: list[dict], out: Path, *, prefix: str, lrs: list[float], primary_lr: float,
                 ks: list[int], phonebooks: list[int]) -> dict:
    analysis = analyze(cells, lrs=lrs, primary_lr=primary_lr, ks=ks, phonebooks=phonebooks)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{prefix}_cells.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CELL_COLUMNS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for cell in sorted(cells, key=lambda c: (c["lr"], c["phonebook"], c["init"], c["seed"], -1 if c["plain"] else c["k"])):
            writer.writerow({key: ("" if cell.get(key) is None else cell[key]) for key in CELL_COLUMNS})
    (out / f"{prefix}_analysis.json").write_bytes(
        (json.dumps(_json_safe(analysis), indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return analysis


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", required=True, help="evidence folder from run_experiment.py")
    parser.add_argument("--lrs", type=float, nargs="+", required=True)
    parser.add_argument("--primary-lr", type=float, required=True)
    parser.add_argument("--ks", type=int, nargs="+", required=True)
    parser.add_argument("--phonebooks", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="e5")
    args = parser.parse_args(argv)
    cells = [cell for root in args.root for cell in load_replay_cells(root)]
    analysis = write_report(cells, args.out, prefix=args.prefix, lrs=args.lrs, primary_lr=args.primary_lr,
                            ks=args.ks, phonebooks=args.phonebooks)
    print(f"noise band {analysis['noise']['band']} updates over {analysis['noise']['groups']} configurations")
    for lr, entry in analysis["lr"].items():
        print(f"lr {lr}: median stable_joint {entry['median_stable_joint']}, plateau {entry['plateau']}")


if __name__ == "__main__":
    main()
