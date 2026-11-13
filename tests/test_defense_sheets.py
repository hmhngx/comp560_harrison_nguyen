"""Defense sheets (phonebook/docs/defense/): the numbers in them are the numbers in the committed results.

A sheet is a study document, so a wrong number in it is worse than a missing one. Four rules keep it honest:

1. Every row of a sheet's numbers table (`| id | statement | value | source |`) is recomputed here from the committed result
   files, by code in this file that does not call the analysis scripts (the two-clocks rows read the per-update logs through the
   project's metrics loader, then apply their own definitions). The value cell must equal the recomputed text exactly, and the
   source cell must link to the files the recomputation read.
2. Every sentence of the sheet that contains a number must cite the numbers-table rows it relies on, and every number in the
   sentence must occur in a cited row. This catches a mistyped or stale number; it cannot catch a correct number attached to
   the wrong quantity, which is what the reading of the sheets is for.
3. The claims table quotes phonebook/RESULTS.md word for word, so a sheet cannot drift from the claims table.
4. Links resolve, the structure is the agreed one (six design bullets, twelve questions with an answer and a rubric, a glossary),
   and the sheets do not use words that claim more than the experiments did.

The checker itself is tested on small texts first (see the tests named `checker_*`).
"""
from __future__ import annotations

import collections
import csv
import functools
import json
import math
import re
import statistics
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHONEBOOK = ROOT / "phonebook"
RESULTS = PHONEBOOK / "results"
WEEK = PHONEBOOK / "experiments" / "week_2026-10-31"
DEFENSE = PHONEBOOK / "docs" / "defense"
SHEETS = {"replay": DEFENSE / "replay-ratio-e3-e5.md", "capacity": DEFENSE / "capacity-e4-e6.md"}
PREFIX = {"replay": "R", "capacity": "C"}
CLAIMS = {"replay": {5, 6, 7, 10}, "capacity": {11, 12, 13, 14, 15}}
HEADINGS = ["The question", "The claims and their status", "The design", "Numbers", "The mechanism", "What was not shown",
            "Twelve hard questions", "Glossary"]
GLOSSARY = {
    "replay": ["exact recall", "update", "k", "joint recall", "stable_joint", "dip", "noise band", "sign rule",
               "unit of replication"],
    "capacity": ["exact recall", "entry", "epoch", "first_perfect", "budget", "rung", "capacity", "lower bound",
                 "boundary", "bits per parameter"],
}


# ------------------------------------------------------------------ reading the committed results

@functools.lru_cache(maxsize=None)
def table(name: str) -> tuple[dict, ...]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as file:
        return tuple(csv.DictReader(file))


@functools.lru_cache(maxsize=None)
def document(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def spec(name: str) -> dict:
    return json.loads((PHONEBOOK / "specs" / name).read_text(encoding="utf-8"))


def to_int(text: str):
    return None if text == "" else int(text)


def one(value: float) -> str:
    return f"{float(value):.1f}"


def plain(value) -> str:
    """An integer-valued number without a trailing .0; anything else is a bug in the caller."""
    assert float(value) == int(value), value
    return str(int(value))


def listed(items, sep: str = ", ") -> str:
    return sep.join(str(item) for item in items)


# ------------------------------------------------------------------ the registry: id -> (sources, recomputation)

Entry = tuple[tuple[str, ...], Callable[[], str]]
REGISTRY: dict[str, Entry] = {}


def number(ident: str, *sources: str):
    def register(function):
        assert ident not in REGISTRY, ident
        REGISTRY[ident] = (tuple(sources), function)
        return function
    return register


KS = (1, 2, 3, 4, 5, 9)
PHONEBOOKS = (2, 4, 7, 8, 9, 12)
E5 = "results/e5_cells.csv"
E3 = "results/e3_cells.csv"


def e5_runs(lr: float, *, plain_runs: bool = False, seed: int = 42) -> list[dict]:
    return [r for r in table("e5_cells.csv")
            if float(r["lr"]) == lr and (r["plain"] == "True") == plain_runs and int(r["seed"]) == seed]


def e5_ks(lr: float) -> list[int]:
    return sorted({int(r["k"]) for r in e5_runs(lr)})


def e5_phonebook_medians(lr: float, field: str) -> dict:
    """{phonebook: {k: median over the three inits}}."""
    groups: dict = collections.defaultdict(list)
    for run in e5_runs(lr):
        groups[(int(run["phonebook"]), int(run["k"]))].append(int(run[field]))
    assert all(len(values) == 3 for values in groups.values()), "every (phonebook, k) needs its three inits"
    return {p: {k: statistics.median(groups[(p, k)]) for k in e5_ks(lr)} for p in PHONEBOOKS}


def e5_medians(lr: float, field: str = "stable_joint") -> dict:
    table_ = e5_phonebook_medians(lr, field)
    return {k: statistics.median(table_[p][k] for p in PHONEBOOKS) for k in e5_ks(lr)}


def e5_gains(lr: float, k: int) -> list:
    """Updates saved by k against k = 1, phonebook by phonebook (positive means k is faster)."""
    table_ = e5_phonebook_medians(lr, "stable_joint")
    return [table_[p][1] - table_[p][k] for p in PHONEBOOKS]


def e5_single_gains(k: int) -> list[int]:
    """Updates saved by k against k = 1 in each (phonebook, init) pair at lr 0.001."""
    by = {(int(r["phonebook"]), int(r["init"]), int(r["k"])): int(r["stable_joint"]) for r in e5_runs(0.001)}
    return [by[(p, i, 1)] - by[(p, i, k)] for (p, i, kk) in sorted(by) if kk == k]


@number("R01", E3)
def _r01() -> str:
    rows = table("e3_cells.csv")
    checkpoints, ks, lrs = {r["scenario"] for r in rows}, {r["k"] for r in rows}, {r["lr"] for r in rows}
    assert len(rows) == len(checkpoints) * len(ks) * len(lrs)
    return f"{len(rows)} runs = {len(checkpoints)} checkpoints x {len(ks)} values of k x {len(lrs)} learning rates"


@number("R02", "specs/e3_replay_ratio.json", "specs/e5_main_pb2.json")
def _r02() -> str:
    seen = set()
    for name in ("e3_replay_ratio.json", "e5_main_pb2.json"):
        content = spec(name)
        assert content["require_one_update_per_epoch"] is True
        for condition in content["conditions"]:
            seen.add((condition["epochs"], condition["batch_size"], condition["accuracy_interval"]))
    assert len(seen) == 1, seen
    ((epochs, batch, interval),) = seen
    return f"{epochs} updates per run, batch size {batch}, logged every {interval} update, one update per epoch"


@number("R03", E5)
def _r03() -> str:
    main = [r for r in table("e5_cells.csv") if r["seed"] == "42"]
    phonebooks, inits = sorted({int(r["phonebook"]) for r in main}), sorted({int(r["init"]) for r in main})
    assert tuple(phonebooks) == PHONEBOOKS
    return f"{len(phonebooks)} phonebooks ({listed(phonebooks)}) x {len(inits)} Phase-A inits ({listed(inits)})"


@number("R04", E5)
def _r04() -> str:
    runs, ks = e5_runs(0.001), e5_ks(0.001)
    assert tuple(ks) == KS and len(runs) == len(PHONEBOOKS) * 3 * len(ks)
    return f"{len(runs)} runs = {len(PHONEBOOKS)} phonebooks x 3 inits x {len(ks)} values of k ({listed(ks)})"


@number("R05", E5)
def _r05() -> str:
    shapes = {(len(e5_runs(lr)), tuple(e5_ks(lr))) for lr in (0.003, 0.0003)}
    assert len(shapes) == 1, shapes
    ((count, ks),) = shapes
    return f"learning rates 0.003 and 0.0003 use k = {listed(ks)}, {count} runs each"


@number("R06", E5)
def _r06() -> str:
    runs = e5_runs(0.001, plain_runs=True)
    ended_a0 = sum(int(r["final_A"]) == 0 for r in runs)
    ended_b10 = sum(int(r["final_B"]) == 10 for r in runs)
    never_joint = sum(r["stable_joint"] == "" for r in runs)
    return (f"{len(runs)} runs without a copy of A: A ended at 0 of 10 in {ended_a0}, B at 10 of 10 in {ended_b10}, "
            f"joint recall reached in {len(runs) - never_joint}")


@number("R07", E5)
def _r07() -> str:
    groups: dict = collections.defaultdict(list)
    for run in table("e5_cells.csv"):
        groups[(run["phonebook"], run["init"], run["lr"], run["k"], run["plain"])].append(run)
    noisy = {key: group for key, group in groups.items() if len(group) > 1}
    sizes = {len(group) for group in noisy.values()}
    assert len(sizes) == 1
    spreads = [max(int(r["stable_joint"]) for r in g) - min(int(r["stable_joint"]) for r in g) for g in noisy.values()]
    seeds = sorted({int(r["seed"]) for g in noisy.values() for r in g})
    return (f"{len(noisy)} configurations x {sizes.pop()} runs (shuffle seeds {listed(seeds)}), "
            f"largest spread of stable_joint {max(spreads)}")


@number("R08", E5)
def _r08() -> str:
    return listed(one(v) for v in e5_medians(0.001).values())


@number("R09", E5)
def _r09() -> str:
    return listed(one(v) for v in e5_medians(0.001, "min_a").values())


@number("R10", E5)
def _r10() -> str:
    counts = [sum(g > 0 for g in e5_gains(0.001, k)) for k in KS if k != 1]
    return f"{listed(counts)} of {len(PHONEBOOKS)}"


@number("R11", E5)
def _r11() -> str:
    return listed(one(statistics.median(e5_gains(0.001, k))) for k in KS if k != 1)


@number("R12")
def _r12() -> str:
    n = len(PHONEBOOKS)
    tail = sum(math.comb(n, i) for i in range(5, n + 1))
    return f"{tail}/{2 ** n} = {tail / 2 ** n:.3f}"


def single_run_line(k: int) -> str:
    gains = e5_single_gains(k)
    return (f"{sum(g > 0 for g in gains)} faster, {sum(g == 0 for g in gains)} tied, {sum(g < 0 for g in gains)} slower "
            f"of {len(gains)}; mean gain {statistics.mean(gains):.1f} updates")


@number("R13", E5)
def _r13() -> str:
    return single_run_line(2)


@number("R14", E5)
def _r14() -> str:
    return single_run_line(3)


@number("R15", E5)
def _r15() -> str:
    return single_run_line(9)


@number("R16", E5)
def _r16() -> str:
    return listed(e5_gains(0.001, 2))


@number("R17", E5)
def _r17() -> str:
    return listed(plain(median * (10 + 10 * k)) for k, median in e5_medians(0.001).items())


@number("R18", E5)
def _r18() -> str:
    return f"{listed(one(v) for v in e5_medians(0.003).values())} (k = {listed(e5_ks(0.003))})"


@number("R19", E5)
def _r19() -> str:
    return f"{listed(one(v) for v in e5_medians(0.0003).values())} (k = {listed(e5_ks(0.0003))})"


@number("R20", E5)
def _r20() -> str:
    parts = []
    for lr in (0.003, 0.0003):
        ks = [k for k in e5_ks(lr) if k != 1]
        faster = [sum(g > 0 for g in e5_gains(lr, k)) for k in ks]
        gain = [one(statistics.median(e5_gains(lr, k))) for k in ks]
        parts.append(f"lr {lr}: k = {listed(ks)} faster in {listed(faster)} of {len(PHONEBOOKS)}, median gains {listed(gain)}")
    return "; ".join(parts)


@number("R21", E5)
def _r21() -> str:
    return listed(one(e5_medians(lr)[1]) for lr in (0.003, 0.001, 0.0003))


@number("R22", "results/e5_prediction_scores.json")
def _r22() -> str:
    results = collections.Counter(line["result"] for line in document("e5_prediction_scores.json"))
    hits, exact, misses = results["HIT"] + results["HIT (exact)"], results["HIT (exact)"], results["MISS"]
    missed = [line["q"] for line in document("e5_prediction_scores.json") if line["result"] == "MISS"]
    return f"{hits} of {hits + misses} scored lines hit ({exact} exact); misses {' and '.join(missed)}"


def raw_e5_clock_rows() -> list[dict]:
    """One row per lr-0.001 run: when B is first complete and when A is back for good (A's stable return)."""
    from phonebook.report_experiments import load_runs

    rows = []
    for phonebook in PHONEBOOKS:
        for run in load_runs(WEEK / f"e5_main_pb{phonebook}"):
            if run["lr"] != 0.001 or run["train"]["a_copies"] != 1 or run["train"]["b_copies"] == 0:
                continue
            points, n_a = run["points"], run["n_a"]
            stable_a = None
            for point in reversed(points):
                assert point["A"] is not None
                if point["A"] != n_a:
                    break
                stable_a = point["epoch"]
            rows.append({"phonebook": phonebook, "k": run["train"]["b_copies"],
                         "first_b": run["summary"]["first_b_complete"], "stable_a": stable_a})
    assert len(rows) == 6 * 3 * 6
    return rows


@functools.lru_cache(maxsize=None)
def e5_clock_rows() -> tuple:
    return tuple(raw_e5_clock_rows())


@number("R23", "experiments/week_2026-10-31/")
def _r23() -> str:
    rows = e5_clock_rows()
    first_b = {(r["phonebook"], r["k"]): [] for r in rows}
    for r in rows:
        first_b[(r["phonebook"], r["k"])].append(r["first_b"])
    medians = {key: statistics.median(values) for key, values in first_b.items()}
    savings = [statistics.median(medians[(p, 1)] - medians[(p, k)] for p in PHONEBOOKS) for k in KS if k != 1]
    return listed(one(s) for s in savings)


@number("R24", "experiments/week_2026-10-31/")
def _r24() -> str:
    rows = e5_clock_rows()
    return listed(one(statistics.median(r["stable_a"] for r in rows if r["k"] == k)) for k in KS)


@number("R25", "experiments/week_2026-10-31/")
def _r25() -> str:
    rows = e5_clock_rows()
    return listed(sum(r["stable_a"] > r["first_b"] for r in rows if r["k"] == k) for k in KS) + " of 18"


@number("R26", E3)
def _r26() -> str:
    rows = table("e3_cells.csv")
    return listed(plain(statistics.median(int(r["stable_joint"]) for r in rows if r["lr"] == "0.001" and int(r["k"]) == k))
                  for k in (1, 2, 3, 5, 9))


@number("R27", E3)
def _r27() -> str:
    rows = table("e3_cells.csv")
    best, saved = [], []
    for lr in ("0.003", "0.001", "0.0003", "0.0001"):
        medians = {k: statistics.median(int(r["stable_joint"]) for r in rows if r["lr"] == lr and int(r["k"]) == k)
                   for k in (1, 2, 3, 5, 9)}
        k_star = min(medians, key=lambda k: (medians[k], k))
        best.append(k_star)
        saved.append(f"{plain(medians[1] - medians[k_star])} ({round(100 * (medians[1] - medians[k_star]) / medians[1])}%)")
    return f"k = {listed(best)}; updates saved against k = 1: {listed(saved)}"


def e3_sign_line(k: int) -> str:
    rows = table("e3_cells.csv")
    by = {(r["scenario"], r["lr"], int(r["k"])): int(r["stable_joint"]) for r in rows}
    gains = [by[(s, lr, 1)] - by[(s, lr, k)] for (s, lr, kk) in sorted(by) if kk == k]
    faster, tied, slower = sum(g > 0 for g in gains), sum(g == 0 for g in gains), sum(g < 0 for g in gains)
    m, low = faster + slower, min(faster, slower)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(low + 1)) / 2 ** m)
    return f"{faster} faster, {tied} tied, {slower} slower of {len(gains)} pairs; two-sided sign test p = {p:.4f}"


@number("R28", E3)
def _r28() -> str:
    return e3_sign_line(2)


@number("R29", E3)
def _r29() -> str:
    return e3_sign_line(9)


@number("R30", E3)
def _r30() -> str:
    rows = table("e3_cells.csv")
    return listed(plain(statistics.median(int(r["min_a"]) for r in rows if r["lr"] == "0.001" and int(r["k"]) == k))
                  for k in (1, 2, 3, 5, 9))


@number("R31", E3)
def _r31() -> str:
    rows = table("e3_cells.csv")
    censored = sum(r["stable_joint"] == "" for r in rows)
    ended = sum(int(r["final_A"]) == 10 and int(r["final_B"]) == 10 for r in rows)
    same = sum(r["first_joint"] == r["stable_joint"] for r in rows)
    return (f"{censored} censored of {len(rows)}; {ended} of {len(rows)} end with A and B at 10 of 10; "
            f"first_joint equals stable_joint in {same} of {len(rows)}")


@number("R32", "specs/", "inputs/")
def _r32() -> str:
    sizes = set()
    for phonebook in PHONEBOOKS:
        for scenario in spec(f"e5_main_pb{phonebook}.json")["scenarios"]:
            for key in ("book_a", "book_b"):
                sizes.add(len((ROOT / scenario[key]).read_text(encoding="utf-8").splitlines()))
    assert len(sizes) == 1, sizes
    return f"every book holds {sizes.pop()} entries (A and B of each of the {len(PHONEBOOKS)} phonebooks)"


@number("R33", "specs/e5_main_pb2.json")
def _r33() -> str:
    rule = spec("e5_main_pb2.json")["decision_rules"]["verdict"]
    match = re.fullmatch(r"At lr ([\d.]+) only: k beats k = 1 if it is faster in at least (\d+) of (\d+) phonebooks "
                         r"and its median paired gain exceeds the noise band\. Other learning rates are descriptive\.", rule)
    assert match, rule
    lr, need, total = match.groups()
    return (f"fixed before the runs: at lr {lr} only, k beats k = 1 if it is faster in at least {need} of {total} "
            f"phonebooks and its median paired gain exceeds the noise band")


@number("R34", "experiments/week_2026-10-31/e5_anchor_e3/compare_with_e3.json")
def _r34() -> str:
    comparison = json.loads((WEEK / "e5_anchor_e3" / "compare_with_e3.json").read_text(encoding="utf-8"))
    epochs = {entry["compared_epochs"] for entry in comparison.values()}
    assert len(epochs) == 1, epochs
    mismatches = sum(len(entry["mismatches"]) for entry in comparison.values())
    return f"{len(comparison)} anchor runs compared with E3 over {epochs.pop()} updates each, {mismatches} mismatches"


@number("R35", "results/e5_prediction_scores.json")
def _r35() -> str:
    line = next(item for item in document("e5_prediction_scores.json") if item["q"] == "Q10")
    assert line["result"] == "STORY SURVIVES", line["result"]
    names = {"k4_or_k5_faster_in_3_plus": "k = 4 or 5 faster than k = 1 in 3 or more phonebooks",
             "noise_band_3_plus": "noise band of 3 or more updates"}
    assert set(line["actual"]) == set(names), line["actual"]
    return "; ".join(f"{names[key]}: {'yes' if value else 'no'}" for key, value in line["actual"].items())


# ---- capacity

E4 = "results/e4_cells.csv"
E6 = "results/e6_cells.csv"
BUDGETS = (100, 300, 500, 1000)
WIDTHS = (8, 16, 32, 64)
SLASH = " / "          # between widths when each width has one value
SEMI = "; "            # between widths when each width has a value per seed (those use "/" between seeds)


@functools.lru_cache(maxsize=None)
def e4_cells() -> tuple:
    return tuple({"d": int(r["model_d"]), "seed": int(r["seed"]), "n": int(r["n"]), "params": int(r["params"]),
                  "first": to_int(r["first_perfect"]), "control": int(r["max_control"]),
                  "recall": {b: int(r[f"recall_at_{b}"]) for b in BUDGETS}}
                 for r in table("e4_cells.csv"))


@functools.lru_cache(maxsize=None)
def e6_cells() -> tuple:
    return tuple({"d": int(r["model_d"]), "seed": int(r["seed"]), "n": int(r["n"]), "params": int(r["params"]),
                  "first": to_int(r["first_perfect"]), "control": int(r["max_control"])}
                 for r in table("e6_cells.csv"))


def ladder(d: int) -> list[int]:
    return sorted({c["n"] for c in e4_cells() if c["d"] == d})


def perfect_by(cell: dict, budget: int) -> bool:
    return cell["first"] is not None and cell["first"] <= budget


def e4_capacity(d: int, budget: int):
    """Largest rung such that it and every smaller rung have at least 2 of 3 seeds perfect by the budget."""
    best = None
    for n in ladder(d):
        if sum(perfect_by(c, budget) for c in e4_cells() if c["d"] == d and c["n"] == n) >= 2:
            best = n
        else:
            break
    return best


def median_params(cells) -> float:
    return statistics.median(c["params"] for c in cells)


def bits_per_entry() -> float:
    text = (PHONEBOOK / "create_phonebook.py").read_text(encoding="utf-8")
    match = re.search(r"range\((\d+) \* (\d+) \* (\d[\d_]*)\)", text)
    assert match, "the phone-number space is no longer drawn from range(a * b * c)"
    a, b, c = (int(group.replace("_", "")) for group in match.groups())
    return math.log2(a * b * c)


@number("C01", "create_phonebook.py")
def _c01() -> str:
    text = (PHONEBOOK / "create_phonebook.py").read_text(encoding="utf-8")
    a, b, c = (int(g.replace("_", "")) for g in re.search(r"range\((\d+) \* (\d+) \* (\d[\d_]*)\)", text).groups())
    return f"{bits_per_entry():.3f} bits = log2({a} x {b} x {c:,})"


@number("C02", "specs/")
def _c02() -> str:
    seen: dict = collections.defaultdict(set)
    widths: dict = collections.defaultdict(set)
    for path in sorted((PHONEBOOK / "specs").glob("e[46]_*.json")):
        family = path.name[:2]
        for condition in json.loads(path.read_text(encoding="utf-8"))["conditions"]:
            model = condition["model"]
            widths[family].add(model["embedding_dim"])
            seen[family].add((condition["lr"], condition["batch_size"], condition["accuracy_interval"],
                              model["n_layers"], model["n_heads"], condition["epochs"]))
    assert {family: len(values) for family, values in seen.items()} == {"e4": 1, "e6": 1}, seen
    assert widths["e4"] == set(WIDTHS) == widths["e6"]
    (e4,), (e6,) = seen["e4"], seen["e6"]
    assert e4[:5] == e6[:5]
    lr, batch, interval, layers, heads, _ = e4
    return (f"widths {listed(WIDTHS)}, {layers} layers, {heads} heads, learning rate {lr}, batch size {batch}, "
            f"logged every {interval} epochs; {e4[5]} epochs in E4 and {e6[5]} in E6")


@number("C03", E4)
def _c03() -> str:
    cells = e4_cells()
    assert {c["seed"] for c in cells} == {1, 2, 3}
    rungs = [len(ladder(d)) for d in WIDTHS]
    largest = [max(ladder(d)) for d in WIDTHS]
    assert len(cells) == 3 * sum(rungs)
    return (f"rungs per width {listed(rungs)}; largest rung {listed(largest)}; 3 book seeds (1, 2, 3); "
            f"{len(cells)} runs")


def capacity_line(d: int) -> str:
    return listed(e4_capacity(d, b) or "none" for b in BUDGETS)


@number("C04", E4)
def _c04() -> str:
    return capacity_line(8)


@number("C05", E4)
def _c05() -> str:
    return capacity_line(16)


@number("C06", E4)
def _c06() -> str:
    return capacity_line(32)


@number("C07", E4)
def _c07() -> str:
    return capacity_line(64)


def capacity_cells(d: int) -> list[dict]:
    return [c for c in e4_cells() if c["d"] == d and c["n"] == e4_capacity(d, 1000)]


@number("C08", E4)
def _c08() -> str:
    return SLASH.join(f"{int(median_params(capacity_cells(d))):,}" for d in WIDTHS)


@number("C09", E4)
def _c09() -> str:
    return SLASH.join(f"{e4_capacity(d, 1000) * bits_per_entry() / median_params(capacity_cells(d)):.3f}" for d in WIDTHS)


@number("C10", E4)
def _c10() -> str:
    return SLASH.join("stable" if e4_capacity(d, 500) == e4_capacity(d, 1000) else "moving" for d in WIDTHS)


@number("C11", E4)
def _c11() -> str:
    parts = []
    for d in WIDTHS:
        firsts = [c["first"] for c in e4_cells() if c["d"] == d and c["n"] == 10]
        assert None not in firsts
        parts.append(f"{min(firsts)} to {max(firsts)}")
    return SLASH.join(parts)


@number("C12", E4)
def _c12() -> str:
    capacity = [e4_capacity(d, 1000) for d in WIDTHS]
    params = [median_params(capacity_cells(d)) for d in WIDTHS]
    cap_ratio = [capacity[i + 1] / capacity[i] for i in (1, 2)]
    par_ratio = [params[i + 1] / params[i] for i in (1, 2)]
    return (f"capacity ratios {cap_ratio[0]:.2f} and {cap_ratio[1]:.2f} (width 16 to 32, 32 to 64); "
            f"parameter ratios {par_ratio[0]:.2f} and {par_ratio[1]:.2f}")


@number("C13", E4, E6, "inputs/")
def _c13() -> str:
    sizes = {len((PHONEBOOK / "inputs" / f"capacity_seed{s}_control.txt").read_text(encoding="utf-8").splitlines())
             for s in (1, 2, 3)}
    assert len(sizes) == 1
    worst = max(c["control"] for c in e4_cells() + e6_cells())
    return f"{worst} of {sizes.pop()} control entries recalled in all {len(e4_cells())} E4 runs and all {len(e6_cells())} E6 runs"


@number("C14", E4)
def _c14() -> str:
    parts = []
    for d in WIDTHS:
        n = e4_capacity(d, 1000)
        firsts = [("none" if c["first"] is None else c["first"]) for c in sorted(capacity_cells(d), key=lambda c: c["seed"])]
        parts.append(f"{n} entries: {listed(firsts, '/')}")
    return SEMI.join(parts)


def first_failing_rung(d: int) -> int:
    return next(n for n in ladder(d) if n > e4_capacity(d, 1000))


@number("C15", E4)
def _c15() -> str:
    parts = []
    for d in WIDTHS:
        n = first_failing_rung(d)
        recall = [c["recall"][1000] for c in sorted(e4_cells(), key=lambda c: c["seed"]) if c["d"] == d and c["n"] == n]
        parts.append(f"{listed(recall, '/')} of {n}")
    return SEMI.join(parts)


@functools.lru_cache(maxsize=None)
def e6_curves() -> dict:
    curves: dict = collections.defaultdict(list)
    size = {}
    for row in table("fig_e6_curves.csv"):
        d, seed = int(row["model_d"]), int(row["seed"])
        curves[(d, seed)].append((int(row["epoch"]), int(row["recalled"])))
        size[d] = int(row["entries"])
    return {"curves": dict(curves), "size": size}


def e6_first(d: int, seed: int):
    n = e6_curves()["size"][d]
    return next((e for e, x in e6_curves()["curves"][(d, seed)] if x == n), None)


@number("C16", E6, "results/fig_e6_curves.csv")
def _c16() -> str:
    return SEMI.join("/".join(("never" if e6_first(d, s) is None else str(e6_first(d, s))) for s in (1, 2, 3))
                     for d in WIDTHS)


@number("C17", E6)
def _c17() -> str:
    parts = []
    for d in WIDTHS:
        cells = [c for c in e6_cells() if c["d"] == d]
        passing = [b for b in (1000, 2000, 3000, 4000) if sum(perfect_by(c, b) for c in cells) >= 2]
        parts.append(f"yes ({passing[0]})" if passing else "no")
    return SLASH.join(parts)


@number("C18", "results/fig_e6_curves.csv")
def _c18() -> str:
    parts = []
    for d in WIDTHS:
        ends = [x for s in (1, 2, 3) for e, x in e6_curves()["curves"][(d, s)] if e == 4000]
        parts.append(f"{listed(ends, '/')} of {e6_curves()['size'][d]}")
    return SEMI.join(parts)


@number("C19", "results/fig_e6_curves.csv")
def _c19() -> str:
    parts = []
    for d in WIDTHS:
        n = e6_curves()["size"][d]
        parts.append(listed((sum(x == n for e, x in e6_curves()["curves"][(d, s)]) for s in (1, 2, 3)), "/"))
    return SEMI.join(parts) + " of 400 logged epochs"


@number("C20", "results/fig_e6_curves.csv")
def _c20() -> str:
    best, where = [], []
    for s in (1, 2, 3):
        points = e6_curves()["curves"][(64, s)]
        top = max(x for e, x in points)
        best.append(top)
        where.append(next(e for e, x in points if x == top))
    return f"{listed(best, '/')} of {e6_curves()['size'][64]} at logged epochs {listed(where, '/')}"


def window_mean(d: int, low: int, high: int) -> Fraction:
    n = e6_curves()["size"][d]
    values = [Fraction(x, n) for s in (1, 2, 3) for e, x in e6_curves()["curves"][(d, s)] if low <= e <= high]
    assert len(values) == 300
    return sum(values) / len(values)


@number("C21", "results/fig_e6_curves.csv")
def _c21() -> str:
    parts = []
    for d in WIDTHS:
        early, late = window_mean(d, 1001, 2000), window_mean(d, 3001, 4000)
        parts.append(f"{float(early):.4f} to {float(late):.4f} ({float(late - early):+.4f})")
    return SLASH.join(parts)


@number("C22", "results/fig_e6_curves.csv")
def _c22() -> str:
    labels = []
    for d in WIDTHS:
        gain = window_mean(d, 3001, 4000) - window_mean(d, 1001, 2000)
        labels.append("rising" if gain >= Fraction(5, 1000) else "falling" if gain <= -Fraction(5, 1000) else "flat")
    return SLASH.join(labels)


@number("C23", E4, E6)
def _c23() -> str:
    bits = bits_per_entry()
    values = []
    for d in WIDTHS:
        cells = [c for c in e6_cells() if c["d"] == d]
        n = cells[0]["n"]
        passed = sum(perfect_by(c, 4000) for c in cells) >= 2
        entries = n if passed else e4_capacity(d, 1000)
        params = median_params(cells) if passed else median_params(capacity_cells(d))
        values.append(f"{entries * bits / params:.3f}")
    return SLASH.join(values)


@number("C24", "results/e6_extension.json")
def _c24() -> str:
    extension = document("e6_extension.json")
    flags = [flag for d in WIDTHS for flag in extension[str(d)]["first_1000_match_by_seed"].values()]
    return f"{sum(flags)} of {len(flags)} runs"


@number("C25", "results/e4_prediction_scores.json", "results/e6_prediction_scores.json")
def _c25() -> str:
    def line(name: str) -> str:
        results = collections.Counter(item["result"] for item in document(name))
        hits, exact, misses = results["HIT"] + results["HIT (exact)"], results["HIT (exact)"], results["MISS"]
        return f"{hits} of {hits + misses} scored lines hit ({exact} exact)"
    return f"E4: {line('e4_prediction_scores.json')}; E6: {line('e6_prediction_scores.json')}"


@number("C26", "results/fig_e6_curves.csv")
def _c26() -> str:
    points = e6_curves()["curves"][(32, 3)]
    top = max(x for e, x in points)
    return f"{top} of {e6_curves()['size'][32]} at logged epoch {next(e for e, x in points if x == top)}"


@number("C27", E4)
def _c27() -> str:
    ratios = [b / a for d in WIDTHS for a, b in zip(ladder(d), ladder(d)[1:])]
    return f"consecutive rungs differ by a factor of {min(ratios):.2f} to {max(ratios):.2f}"


@number("C28", "experiments/week_2026-10-31/e4_capacity_d8_seed1_a/protocol.json",
        "experiments/week_2026-10-31/e6_extend_d64_seed1/protocol.json", "../train.py")
def _c28() -> str:
    optimizers = set()
    for folder in ("e4_capacity_d8_seed1_a", "e6_extend_d64_seed1"):
        protocol = json.loads((WEEK / folder / "protocol.json").read_text(encoding="utf-8"))
        recorded = protocol["optimizer"]
        optimizers.add((recorded["name"], recorded["weight_decay"], recorded["grad_clip"]))
        # E7 added optional per-run overrides of the weight decay and the learning-rate schedule. The recorded optimizer is
        # what these runs used only if no run overrides it, and "constant" holds only if no run set a schedule.
        overridden = "per_run_keys" in recorded or any(key in run for run in protocol["runs"] for key in ("weight_decay", "lr_schedule"))
        assert not overridden, (
            f"{folder}: a run overrides the weight decay or sets a learning-rate schedule: the recorded optimizer no longer describes it")
    assert len(optimizers) == 1, optimizers
    ((name, decay, clip),) = optimizers
    from train import TrainConfig
    assert TrainConfig().lr_schedule is None, "the trainer now applies a learning-rate schedule by default: 'constant' is no longer true"
    trainer = (ROOT / "train.py").read_text(encoding="utf-8").lower()
    assert not any(word in trainer for word in ("scheduler", "warmup", "cosine")), (
        "train.py has a learning-rate schedule other than the opt-in step schedule: 'constant' needs a new check")
    return (f"{name}, weight decay {decay}, gradient clip {clip}, constant learning rate "
            f"(neither protocol records a schedule or an override; the trainer applies a schedule only when --lr-schedule is given)")


@number("C29", "specs/e4_capacity_d8_seed1_a.json")
def _c29() -> str:
    rule = spec("e4_capacity_d8_seed1_a.json")["decision_rules"]["capacity"]
    match = re.search(r"A rung passes at budget E if at least (\d+) of its (\d+) book seeds are perfect by E\. "
                      r"The capacity at E is the largest rung such that it and every smaller rung pass;", rule)
    assert match, rule
    return (f"fixed before the runs: a rung passes if at least {match.group(1)} of its {match.group(2)} book seeds are perfect "
            f"by the budget; the capacity is the largest rung such that it and every smaller rung pass")


# ------------------------------------------------------------------ reading a sheet

CITE = re.compile(r"\[((?:[RC]\d{2})(?:,\s*[RC]\d{2})*)\]")
LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
FILE_CODE = re.compile(r"`[^`\s]*(?:/|\.(?:csv|json|py|md|png|txt|jsonl))[^`]*`")
LABEL = re.compile(r"\b(?:E\d+[a-z]?|Q\d+|L2-SP)\b")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
NUMBER = re.compile(r"(?:(?<![\w.])-)?\d+(?:\.\d+)?")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(*_`\"'])")
LIST_MARK = re.compile(r"^(?:(?:[-*+]|\d+\.)\s+)+")
EXEMPT_PREFIXES = ("*Status as of",)
OVERCLAIM = re.compile(r"\b(?:proves?|proved|proof|significant(?:ly)?|guarantee[sd]?)\b", re.IGNORECASE)
IMPOSSIBLE_OK = ("not shown", "never shown", "has been shown", "not established", "cannot show", "cannot be shown",
                 "not a claim", "no claim", "refuse to claim")


def numbers_in(text: str) -> set[Fraction]:
    """The numbers a reader would take from the text, as exact fractions (ids, labels, links and file names are not data)."""
    text = LINK.sub(" ", text)
    text = FILE_CODE.sub(" ", text)
    text = CITE.sub(" ", text)
    text = LABEL.sub(" ", text)
    text = THOUSANDS.sub("", text)
    return {Fraction(match.group()) for match in NUMBER.finditer(text)}


def cited_ids(text: str) -> list[str]:
    return [item.strip() for group in CITE.findall(text) for item in group.split(",")]


def prose_lines(text: str):
    """(line number, text) of every line that is not a heading, table row, code block or blank line."""
    in_fence = False
    for number_, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith(("#", "|", "---")) or stripped.startswith(EXEMPT_PREFIXES):
            continue
        yield number_, stripped


def prose_problems(text: str, rows: dict[str, str]) -> list[str]:
    """rows maps a numbers-table id to the text (statement and value) whose numbers it vouches for."""
    problems = []
    for number_, line in prose_lines(text):
        for sentence in SENTENCE_END.split(LIST_MARK.sub("", line)):
            ids = cited_ids(sentence)
            unknown = [i for i in ids if i not in rows]
            if unknown:
                problems.append(f"line {number_}: cites {unknown}, which are not in the numbers table")
            tokens = numbers_in(sentence)
            if not tokens:
                continue
            if not ids:
                problems.append(f"line {number_}: has the numbers {sorted(map(str, tokens))} but cites no row: {sentence[:70]}")
                continue
            allowed = set().union(*(numbers_in(rows[i]) for i in ids if i in rows))
            extra = tokens - allowed
            if extra:
                problems.append(f"line {number_}: {sorted(map(str, extra))} is not in the cited rows {ids}: {sentence[:70]}")
    return problems


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def headings(text: str) -> list[str]:
    return re.findall(r"^## (\d+)\. (.+)$", text, flags=re.MULTILINE)


def section(text: str, name: str) -> str:
    match = re.search(rf"^## \d+\. {re.escape(name)}\s*$(.*?)(?=^## \d+\. |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    assert match, f"no section named {name!r}"
    return match.group(1)


def table_rows(block: str) -> list[list[str]]:
    lines = [line for line in block.splitlines() if line.strip().startswith("|")]
    assert len(lines) >= 3, "a table needs a header, a separator and a row"
    assert re.fullmatch(r"[|\s:\-]+", lines[1]), "the second table line must be the separator"
    return [split_row(line) for line in lines[2:]]


def numbers_table(text: str) -> list[dict]:
    block = section(text, "Numbers")
    header = split_row([line for line in block.splitlines() if line.strip().startswith("|")][0])
    assert header == ["id", "statement", "value", "source"], header
    rows = []
    for cells in table_rows(block):
        assert len(cells) == 4, cells
        rows.append(dict(zip(header, cells)))
    return rows


def results_claims() -> dict[int, list[str]]:
    claims = {}
    for line in (PHONEBOOK / "RESULTS.md").read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*(\d+)\s*\|", line)
        if match:
            claims[int(match.group(1))] = split_row(line)[1:3]
    return claims


def relative_links(text: str) -> list[str]:
    return [target for _, target in LINK.findall(text) if not target.startswith(("http://", "https://", "mailto:", "#"))]


def source_targets(cell: str, sheet: Path) -> set[Path]:
    return {(sheet.parent / target.split("#")[0]).resolve() for target in relative_links(cell)}


@functools.lru_cache(maxsize=None)
def sheet_text(key: str) -> str:
    return SHEETS[key].read_text(encoding="utf-8")


# ================================================================== the checker itself (small texts, no sheets)

def test_checker_numbers_are_exact_fractions_and_ignore_ids_labels_links_and_file_names() -> None:
    text = "In E5 (see [the table](../../results/e5_cells.csv) and `e5_analysis.json`) k = 2 gains 1.5 and 3,064 [R10, R11] -1.5 Q3."
    assert numbers_in(text) == {Fraction(2), Fraction(3, 2), Fraction(3064), Fraction(-3, 2)}
    assert numbers_in("40") == numbers_in("40.0") and numbers_in("0.5") == {Fraction(1, 2)}
    assert numbers_in("L2-SP and E1b and Q12") == set()


def test_checker_a_minus_sign_counts_only_when_it_is_a_sign() -> None:
    assert numbers_in("gain -6.0") == {Fraction(-6)}
    assert numbers_in("(-1.5)") == {Fraction(-3, 2)}
    assert numbers_in("epochs 1001-2000 and 1e-5") == {Fraction(1001), Fraction(2000), Fraction(1), Fraction(5)}


ROWS = {"R01": "6 phonebooks x 3 inits 40.0, 38.5", "R02": "gains 1.5, -6.0 of 7"}


def test_checker_accepts_a_sentence_whose_numbers_are_all_in_the_cited_rows() -> None:
    assert prose_problems("The median is 38.5 for k = 3 of 6 phonebooks [R01].", ROWS) == []
    assert prose_problems("Gains are -6.0 and 1.5 [R01, R02].", ROWS) == []


def test_checker_rejects_a_number_that_no_cited_row_contains() -> None:
    assert prose_problems("The median is 39.5 [R01].", ROWS)
    assert prose_problems("The gain is 6.0 [R02].", ROWS), "a missing minus sign is a different number"
    assert prose_problems("The gain is -6.0 [R01].", ROWS), "the number is in another row, not the cited one"


def test_checker_rejects_a_sentence_with_numbers_and_no_citation_or_an_unknown_id() -> None:
    (problem,) = prose_problems("The median is 38.5.", ROWS)
    assert "cites no row" in problem
    assert prose_problems("The median is 38.5 [R09].", ROWS)
    (problem,) = prose_problems("No numbers at all here [R09].", ROWS)
    assert "not in the numbers table" in problem
    assert prose_problems("No numbers at all here.", ROWS) == []


def test_checker_judges_each_sentence_on_its_own() -> None:
    # 40.0 is in the row cited by the first sentence, so only a checker that judges sentences separately objects
    (problem,) = prose_problems("The median is 38.5 [R01]. It was 40.0 before.", ROWS)
    assert "cites no row" in problem and "40.0" in problem


def test_checker_skips_headings_tables_code_blocks_and_the_status_line_and_strips_list_marks() -> None:
    text = "# Title 2026\n| 99 | 98 |\n```\npython x 97\n```\n*Status as of November 4, 2026.*\n- 1. Item 38.5 [R01].\n"
    assert prose_problems(text, ROWS) == []


def test_checker_numbers_table_needs_the_agreed_header_and_four_cells() -> None:
    good = "## 4. Numbers\n\n| id | statement | value | source |\n|---|---|---|---|\n| R01 | s | v | x |\n\n## 5. Next\n"
    assert numbers_table(good) == [{"id": "R01", "statement": "s", "value": "v", "source": "x"}]
    with pytest.raises(AssertionError):
        numbers_table(good.replace("| value |", "| val |"))
    with pytest.raises(AssertionError):
        numbers_table(good.replace("| R01 | s | v | x |", "| R01 | s | v |"))


def test_checker_citations_are_found_in_lists_and_may_name_several_rows() -> None:
    assert cited_ids("a [R01] b [R02, R03] c [C10]") == ["R01", "R02", "R03", "C10"]
    assert cited_ids("an ordinary [link](x) and [text]") == []


# ================================================================== the registry itself

def test_registry_ids_are_unique_well_formed_and_split_between_the_two_sheets() -> None:
    assert all(re.fullmatch(r"[RC]\d{2}", ident) for ident in REGISTRY)
    for key, prefix in PREFIX.items():
        ids = sorted(i for i in REGISTRY if i.startswith(prefix))
        assert ids == [f"{prefix}{n:02d}" for n in range(1, len(ids) + 1)], f"{key}: ids must be consecutive"


def fake_cell(d: int, seed: int, n: int, first) -> dict:
    return {"d": d, "seed": seed, "n": n, "params": 100, "first": first, "control": 0, "recall": {b: 0 for b in BUDGETS}}


def test_registry_capacity_needs_two_of_three_seeds_by_the_budget_and_stops_at_the_first_failing_rung(monkeypatch) -> None:
    # rung 10: seeds perfect at 50, 60, never; rung 20: 50, never, never; rung 40: all three perfect, but it follows a failing rung
    rungs = {10: (50, 60, None), 20: (50, None, None), 40: (50, 60, 70)}
    cells = tuple(fake_cell(8, seed, n, first) for n, firsts in rungs.items() for seed, first in zip((1, 2, 3), firsts))
    monkeypatch.setattr(sys.modules[__name__], "e4_cells", lambda: cells)
    assert e4_capacity(8, 49) is None, "no seed is perfect yet"
    assert e4_capacity(8, 50) is None, "one seed of three is not enough"
    assert e4_capacity(8, 60) == 10, "a seed perfect exactly at the budget counts"
    assert e4_capacity(8, 1000) == 10, "the pass at rung 40 is isolated, so it is not counted"


def test_registry_every_recomputation_runs_and_returns_plain_text_without_pipes() -> None:
    for ident, (sources, function) in REGISTRY.items():
        value = function()
        assert isinstance(value, str) and value and "|" not in value, ident
        for source in sources:
            assert (PHONEBOOK / source).exists(), f"{ident}: source {source} does not exist"


C28_FOLDERS = ("e4_capacity_d8_seed1_a", "e6_extend_d64_seed1")


def copy_c28_protocols(week: Path, change=lambda protocol: None) -> None:
    for folder in C28_FOLDERS:
        protocol = json.loads((WEEK / folder / "protocol.json").read_text(encoding="utf-8"))
        change(protocol)
        (week / folder).mkdir(parents=True)
        (week / folder / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")


def test_registry_c28_holds_on_a_faithful_copy_of_the_cited_protocols(monkeypatch, tmp_path: Path) -> None:
    copy_c28_protocols(tmp_path)
    monkeypatch.setitem(globals(), "WEEK", tmp_path)
    expected = ("AdamW, weight decay 0.1, gradient clip 1.0, constant learning rate (neither protocol records "
                "a schedule or an override; the trainer applies a schedule only when --lr-schedule is given)")
    assert REGISTRY["C28"][1]() == expected


@pytest.mark.parametrize("change", [
    lambda protocol: protocol["runs"][0].update(lr_schedule="step:2000:0.1"),
    lambda protocol: protocol["runs"][0].update(weight_decay=0.0),
    lambda protocol: protocol["optimizer"].update(per_run_keys=["weight_decay"]),
], ids=["a run sets a schedule", "a run overrides the weight decay", "the optimizer block lists per-run keys"])
def test_registry_c28_notices_an_override_or_a_schedule_in_a_cited_protocol(monkeypatch, tmp_path: Path, change) -> None:
    copy_c28_protocols(tmp_path, change)
    monkeypatch.setitem(globals(), "WEEK", tmp_path)
    with pytest.raises(AssertionError, match="no longer describes it"):
        REGISTRY["C28"][1]()


def test_registry_c28_notices_a_trainer_that_applies_a_schedule_by_default(monkeypatch) -> None:
    import train

    monkeypatch.setattr(train, "TrainConfig", lambda: type("Config", (), {"lr_schedule": "step:2000:0.1"})())
    with pytest.raises(AssertionError, match="by default"):
        REGISTRY["C28"][1]()


@pytest.mark.parametrize("word", ["get_cosine_schedule_with_warmup", "lr_scheduler.StepLR", "linear warmup"])
def test_registry_c28_notices_any_other_kind_of_schedule_in_the_trainer(monkeypatch, tmp_path: Path, word: str) -> None:
    (tmp_path / "train.py").write_text(f"# {word}", encoding="utf-8")
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="other than the opt-in step schedule"):
        REGISTRY["C28"][1]()


def test_registry_c28_accepts_the_opt_in_step_schedule_hook_that_the_trainer_now_has() -> None:
    trainer = (ROOT / "train.py").read_text(encoding="utf-8")
    assert "param_groups" in trainer and "lr_factor" in trainer          # the hook is there ...
    assert "constant learning rate" in REGISTRY["C28"][1]()              # ... and the claim about E4 and E6 still holds


def test_registry_agrees_with_the_analysis_outputs_it_does_not_call() -> None:
    capacity = document("e4_capacity.json")
    for d in WIDTHS:
        for budget in BUDGETS:
            assert e4_capacity(d, budget) == capacity[str(d)]["capacity_by_budget"][str(budget)], (d, budget)
        assert median_params(capacity_cells(d)) == capacity[str(d)]["params"]
        assert capacity[str(d)]["boundary"] == ("stable" if e4_capacity(d, 500) == e4_capacity(d, 1000) else "moving")
        assert e4_capacity(d, 1000) * bits_per_entry() / capacity[str(d)]["params"] == pytest.approx(
            capacity[str(d)]["bits_per_param_at_largest_budget"])
    analysis = {**document("e5_analysis.json")["lr"], **document("e5_robust_analysis.json")["lr"]}
    for lr in (0.001, 0.003, 0.0003):
        recorded = analysis[str(lr)]
        assert {str(k): v for k, v in e5_medians(lr).items()} == {str(k): v for k, v in recorded["median_stable_joint"].items()}
        assert {str(k): v for k, v in e5_medians(lr, "min_a").items()} == {str(k): v for k, v in recorded["median_min_a"].items()}
    extension = document("e6_extension.json")
    for d in WIDTHS:
        early, late = window_mean(d, 1001, 2000), window_mean(d, 3001, 4000)
        assert float(early) == pytest.approx(extension[str(d)]["mean_accuracy"]["1001_2000"])
        assert float(late) == pytest.approx(extension[str(d)]["mean_accuracy"]["3001_4000"])
        assert float(late - early) == pytest.approx(extension[str(d)]["gain"])
        assert extension[str(d)]["trend"] == _c22().split(SLASH)[WIDTHS.index(d)]
        firsts = extension[str(d)]["first_perfect"]
        assert [e6_first(d, s) for s in (1, 2, 3)] == [firsts[str(s)] for s in (1, 2, 3)]
        passes = extension[str(d)]["passes"]
        assert (sum(perfect_by(c, 4000) for c in e6_cells() if c["d"] == d) >= 2) == passes["4000"]


# ================================================================== the sheets

@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_exists_has_the_agreed_sections_in_order_and_a_status_line(key: str) -> None:
    text = sheet_text(key)
    assert [name for _, name in headings(text)] == HEADINGS
    assert [int(n) for n, _ in headings(text)] == list(range(1, len(HEADINGS) + 1))
    assert re.search(r"^\*Status as of [A-Z][a-z]+ \d{1,2}, 2026", text, flags=re.MULTILINE)
    assert not re.search(r"\b(?:TODO|TBD|XXX|FIXME)\b", text)


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_numbers_table_is_exactly_the_registry_and_every_value_is_recomputed(key: str) -> None:
    rows = numbers_table(sheet_text(key))
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert ids == sorted(i for i in REGISTRY if i.startswith(PREFIX[key])), "table and registry must list the same ids in order"
    wrong = []
    for row in rows:
        sources, function = REGISTRY[row["id"]]
        recomputed = function()
        if row["value"] != recomputed:
            wrong.append(f"{row['id']}: sheet says {row['value']!r}, the data say {recomputed!r}")
        wanted = {(PHONEBOOK / source).resolve() for source in sources}
        if source_targets(row["source"], SHEETS[key]) != wanted:
            wrong.append(f"{row['id']}: source cell links do not match {sorted(sources)}")
        if not row["statement"].strip() or not row["statement"].strip()[0].isupper():
            wrong.append(f"{row['id']}: the statement must be a sentence fragment that starts with a capital")
    assert not wrong, "\n".join(wrong)


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_every_number_in_the_text_is_cited_and_found_in_the_cited_rows(key: str) -> None:
    text = sheet_text(key)
    rows = {row["id"]: f"{row['statement']} {row['value']}" for row in numbers_table(text)}
    problems = prose_problems(text, rows)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_every_numbers_row_is_used_in_the_text(key: str) -> None:
    text = sheet_text(key)
    ids = {row["id"] for row in numbers_table(text)}
    used = {i for _, line in prose_lines(text) for i in cited_ids(line)}
    assert ids - used == set(), f"rows never cited: {sorted(ids - used)}"


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_claims_table_quotes_results_md_word_for_word(key: str) -> None:
    block = section(sheet_text(key), "The claims and their status")
    lines = [line for line in block.splitlines() if line.strip().startswith("|")]
    assert split_row(lines[0]) == ["#", "claim", "status"]
    official = results_claims()
    seen = set()
    for cells in table_rows(block):
        number_ = int(cells[0])
        seen.add(number_)
        assert number_ in official, f"claim {number_} is not in RESULTS.md"
        assert cells[1:] == official[number_], f"claim {number_} differs from RESULTS.md"
    assert seen == CLAIMS[key]


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_design_has_six_bullets(key: str) -> None:
    bullets = [line for line in section(sheet_text(key), "The design").splitlines() if line.startswith("- ")]
    assert len(bullets) == 6


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_has_twelve_questions_each_with_an_answer_and_a_rubric(key: str) -> None:
    block = section(sheet_text(key), "Twelve hard questions")
    lines = [line for line in block.splitlines() if line.strip()]
    questions = [i for i, line in enumerate(lines) if line.startswith("**Q")]
    assert [re.match(r"\*\*Q(\d+)\. ", lines[i]).group(1) for i in questions] == [str(n) for n in range(1, 13)]
    for index in questions:
        question = lines[index]
        assert question.endswith("?**"), question
        assert lines[index + 1].startswith("Answer. "), f"{question[:40]}: the answer line must follow the question"
        assert lines[index + 2].startswith("Must mention. "), f"{question[:40]}: the rubric line must follow the answer"
        assert lines[index + 2].count(";") >= 2, "the rubric lists at least three items separated by semicolons"


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_glossary_defines_the_terms_a_reader_needs(key: str) -> None:
    block = section(sheet_text(key), "Glossary")
    defined = re.findall(r"^- \*\*(.+?)\*\*", block, flags=re.MULTILINE)
    assert defined == GLOSSARY[key]


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_says_what_was_not_shown_and_marks_its_mechanism(key: str) -> None:
    text = sheet_text(key)
    not_shown = [line for line in section(text, "What was not shown").splitlines() if line.startswith("- ")]
    assert len(not_shown) >= 4
    mechanism = section(text, "The mechanism").lower()
    if key == "replay":
        assert "exploratory" in mechanism and "two clocks" in mechanism
    else:
        assert "time, not room" in mechanism and "width 64 is not shown" in mechanism


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_does_not_claim_more_than_the_experiments_did(key: str) -> None:
    text = sheet_text(key)
    assert not OVERCLAIM.search(text), OVERCLAIM.search(text).group()
    for number_, line in prose_lines(text):
        for sentence in SENTENCE_END.split(line):
            if re.search(r"\bimpossible\b", sentence, flags=re.IGNORECASE):
                assert any(phrase in sentence.lower() for phrase in IMPOSSIBLE_OK), \
                    f"line {number_}: 'impossible' must be framed as not shown: {sentence[:90]}"
    if key == "capacity":
        assert len(re.findall(r"lower bound", text)) >= 3


@functools.lru_cache(maxsize=None)
def tracked() -> tuple | None:
    """(files, folders) that git tracks, or None when this is not a git checkout."""
    try:
        done = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    files = {ROOT / name for name in done.stdout.decode("utf-8").split("\0") if name}
    return files, {parent for path in files for parent in path.parents}


@pytest.mark.parametrize("key", sorted(SHEETS))
def test_sheet_links_resolve_and_point_at_tracked_files(key: str) -> None:
    targets = [(SHEETS[key].parent / target.split("#")[0]).resolve() for target in relative_links(sheet_text(key))]
    assert len(targets) >= 10
    missing = [target for target in targets if not target.exists()]
    assert not missing, missing
    if tracked() is None:
        pytest.skip("not a git checkout, so tracked files cannot be listed")
    files, folders = tracked()
    untracked = [target for target in targets if target.parent != DEFENSE.resolve() and target not in files and target not in folders]
    assert not untracked, f"links to files git does not track (they would be broken on GitHub): {untracked}"


def test_sheets_link_to_each_other_and_to_the_results_document() -> None:
    assert "capacity-e4-e6.md" in sheet_text("replay") and "replay-ratio-e3-e5.md" in sheet_text("capacity")
    for key in SHEETS:
        assert "../../RESULTS.md" in sheet_text(key)
