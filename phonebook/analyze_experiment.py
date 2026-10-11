"""Derive exact-recall curves and decision tables from experiment evidence.

The trainer logs teacher-forced sequence accuracy for the training mixture and for
the validation book. When the training mixture repeats identical sequences, those
two numbers determine how many entries of book A and of book B are recalled
exactly. Teacher-forced exact recall equals strict generation recall (identical
on every saved book evaluation); the audit re-checks that on each endpoint.
"""
from __future__ import annotations

import csv
from pathlib import Path

NO_ACCURACY = "N/A"


def load_metrics(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row = {"epoch": int(raw["Epoch"])}
            for key, value in raw.items():
                if key != "Epoch":
                    row[key] = None if value == NO_ACCURACY else float(value)
            rows.append(row)
    return rows


def _count(fraction: float, total: int, what: str) -> int:
    value = fraction * total
    nearest = round(value)
    # The log prints four decimals, so the product is off by at most 5e-5 per sequence.
    if abs(value - nearest) > 5.001e-5 * total:
        raise ValueError(f"{what} accuracy {fraction} of {total} sequences is not an integer count")
    return nearest


def counts_from_row(row: dict, *, a_copies: int, b_copies: int, eval_book: str, n_a: int, n_b: int):
    """Exact (A, B) recall counts at one logged epoch; a book that is neither trained on nor evaluated is None."""
    sizes = {"A": n_a, "B": n_b}
    copies = {"A": a_copies, "B": b_copies}
    other = "B" if eval_book == "A" else "A"
    n_train = a_copies * n_a + b_copies * n_b
    seen = _count(row["Val Seq Acc"], sizes[eval_book], "validation")
    train_correct = _count(row["Train Seq Acc"], n_train, "training")
    counts = {eval_book: seen}
    if copies[other] > 0:
        remainder = train_correct - copies[eval_book] * seen
        if remainder % copies[other] != 0:
            raise ValueError("training counts are not divisible by the number of identical copies")
        counts[other] = remainder // copies[other]
    else:
        if train_correct != copies[eval_book] * seen:
            raise ValueError(f"inconsistent counts for book {eval_book}: the training and validation logs disagree")
        counts[other] = None
    for name, value in counts.items():
        if value is not None and not 0 <= value <= sizes[name]:
            raise ValueError(f"count {value} for book {name} is out of range 0..{sizes[name]}")
    return counts["A"], counts["B"]


def curve(rows: list[dict], *, a_copies: int, b_copies: int, eval_book: str, n_a: int, n_b: int) -> list[dict]:
    points = []
    for row in rows:
        if row["Train Seq Acc"] is None or row["Val Seq Acc"] is None:
            continue
        a, b = counts_from_row(row, a_copies=a_copies, b_copies=b_copies, eval_book=eval_book, n_a=n_a, n_b=n_b)
        points.append({"epoch": row["epoch"], "A": a, "B": b})
    return points


def summarize_curve(points: list[dict], *, n_a: int, n_b: int, budgets: tuple[int, ...]) -> dict:
    by_epoch = {point["epoch"]: point for point in points}
    known = [point for point in points if point["A"] is not None and point["B"] is not None]
    joint = [point["A"] == n_a and point["B"] == n_b for point in known]

    def first(predicate):
        return next((point for point in known if predicate(point)), None)

    complete_b = first(lambda point: point["B"] == n_b)
    first_joint = first(lambda point: point["A"] == n_a and point["B"] == n_b)
    stable_joint = None
    for index in range(len(known) - 1, -1, -1):
        if not joint[index]:
            break
        stable_joint = known[index]["epoch"]
    lowest = min(known, key=lambda point: (point["A"], point["epoch"]), default=None)
    best = max(known, key=lambda point: (point["A"] + point["B"], -point["epoch"]), default=None)
    last = points[-1] if points else None
    return {
        "at": {budget: ({"A": by_epoch[budget]["A"], "B": by_epoch[budget]["B"]} if budget in by_epoch else None)
               for budget in budgets},
        "final": {"epoch": last["epoch"], "A": last["A"], "B": last["B"]} if last else None,
        "first_b_complete": complete_b["epoch"] if complete_b else None,
        "a_at_first_b_complete": complete_b["A"] if complete_b else None,
        "first_joint": first_joint["epoch"] if first_joint else None,
        "stable_joint": stable_joint,
        "min_a": lowest["A"] if lowest else None,
        "epoch_of_min_a": lowest["epoch"] if lowest else None,
        "best_joint_sum": best["A"] + best["B"] if best else None,
        "epoch_of_best_joint_sum": best["epoch"] if best else None,
    }


def select_lambda(records: list[dict]) -> float:
    """Largest summed (A + B) over scenarios; ties go to the larger summed A, then the smaller lambda."""
    by_lambda: dict[float, dict] = {}
    for record in records:
        if record["lambda"] == 0:
            continue
        entry = by_lambda.setdefault(record["lambda"], {"scenarios": set(), "a": 0, "b": 0})
        entry["scenarios"].add(record["scenario"])
        entry["a"] += record["A"]
        entry["b"] += record["B"]
    if len({frozenset(entry["scenarios"]) for entry in by_lambda.values()}) > 1:
        raise ValueError("every lambda must cover the same scenarios")
    return min(by_lambda, key=lambda lam: (-(by_lambda[lam]["a"] + by_lambda[lam]["b"]), -by_lambda[lam]["a"], lam))


def decision_table(records: list[dict], *, n_a: int, n_b: int, min_scenarios: int) -> list[dict]:
    rows = []
    for lam in sorted({record["lambda"] for record in records if record["lambda"] != 0}):
        own = [record for record in records if record["lambda"] == lam]
        full = sum(record["A"] == n_a and record["B"] == n_b for record in own)
        near = sum(record["A"] >= n_a - 1 and record["B"] == n_b for record in own)
        verdict = "matches replay" if full >= min_scenarios else "nearly matches" if near >= min_scenarios else "no"
        rows.append({"lambda": lam, "full_scenarios": full, "near_scenarios": near, "verdict": verdict})
    return rows


def overall_verdict(rows: list[dict]) -> str:
    order = ["no", "nearly matches", "matches replay"]
    return max((row["verdict"] for row in rows), key=order.index)

