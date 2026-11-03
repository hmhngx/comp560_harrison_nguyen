"""Score the locked E4 (capacity) predictions against the capacity analysis, mechanically.

The rules were written down before any E4 result existed: a line is a HIT when the actual value lies inside the locked
[low, high] (or equals the locked answer), and nothing else counts - no rounding, no "close". A capacity of null
(no rung passes) counts as 0 entries.

    python phonebook/score_e4_predictions.py --capacity phonebook/results/e4_capacity.json \\
        --predictions phonebook/preregistration/e4_predictions.json [--json scores.json]

--capacity is the JSON written by analyze_capacity.py (budgets 100 300 500 1000), keyed by embedding width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WIDTHS = ("8", "16", "32", "64")
LARGEST = "1000"


def entries(value) -> int:
    """A capacity of null means no rung passed: zero entries."""
    return 0 if value is None else value


def in_range(value, low, high) -> bool:
    return value is not None and low <= value <= high


def verdict(ok: bool, exact: bool = False) -> str:
    if not ok:
        return "MISS"
    return "HIT (exact)" if exact else "HIT"


def score(report: dict, predictions: dict) -> list[dict]:
    rows = []

    def add(q, what, predicted, actual, result, note=""):
        rows.append({"q": q, "what": what, "predicted": predicted, "actual": actual, "result": result, "note": note})

    for key, budget in (("q1_capacity_at_1000", "1000"), ("q2_capacity_at_300", "300"), ("q3_capacity_at_100", "100")):
        q = predictions[key]
        for d in WIDTHS:
            spec = q[f"d{d}"]
            actual = entries(report[d]["capacity_by_budget"][budget])
            add(key[:2].upper(), f"capacity at {budget} epochs, d={d}", f"{spec['point']} [{spec['low']}, {spec['high']}]",
                actual, verdict(in_range(actual, spec["low"], spec["high"]), actual == spec["point"]),
                f"boundary {report[d]['boundary']}")

    q = predictions["q4_boundary_status_at_1000"]
    for d in WIDTHS:
        actual = report[d]["boundary"]
        add("Q4", f"boundary status, d={d}", q[f"d{d}"], actual, verdict(actual == q[f"d{d}"]))

    q = predictions["q5_capacity_grows_with_width"]
    caps = [entries(report[d]["capacity_by_budget"][LARGEST]) for d in WIDTHS]
    strict = all(a < b for a, b in zip(caps, caps[1:]))
    add("Q5", "capacity strictly increases with width at 1000", q["answer"], caps, verdict(strict and q["answer"] == "strict"))

    q = predictions["q6_bits_per_param_at_1000"]
    for d in WIDTHS:
        spec, actual = q[f"d{d}"], report[d]["bits_per_param_at_largest_budget"]
        add("Q6", f"bits per parameter at 1000 epochs, d={d}", f"{spec['point']} [{spec['low']}, {spec['high']}]",
            None if actual is None else round(actual, 4), verdict(in_range(actual, spec["low"], spec["high"])))

    q = predictions["q7_no_ladder_ceiling"]
    flags = {d: report[d]["capacity_is_lower_bound"] for d in WIDTHS}
    add("Q7", "no model reached the top of its ladder", q["answer"], flags, verdict(q["answer"] is True and not any(flags.values())))

    q = predictions["q8_control_book"]
    violations = sum(report[d]["control_violations"] for d in WIDTHS)
    add("Q8", "control-book entries recalled (runs with any)", q["max_control_violations"], violations,
        verdict(violations <= q["max_control_violations"], violations == q["max_control_violations"]))

    q = predictions["q10_result_that_would_change_my_mind"]
    top = entries(report["64"]["capacity_by_budget"][LARGEST])
    ratios = {f"d{b}/d{a}": (entries(report[b]["capacity_by_budget"][LARGEST]) / entries(report[a]["capacity_by_budget"][LARGEST])
                             if entries(report[a]["capacity_by_budget"][LARGEST]) else None)
              for a, b in (("16", "32"), ("32", "64"))}
    wide = top >= q["d64_capacity_at_least"]
    steep = any(r is not None and r >= q["width_doubling_ratio_at_least"] for r in ratios.values())
    add("Q10", "tripwires fired", "none", {"d64_capacity": top, "ratios": ratios, "wide": wide, "steep": steep},
        "STORY CHALLENGED" if wide or steep else "STORY SURVIVES", q["tripwire"])
    return rows


def render(rows: list[dict]) -> str:
    lines = [f"{r['q']:4} {r['what']:50} predicted {str(r['predicted']):20} actual {str(r['actual']):28} {r['result']}"
             + (f"   [{r['note']}]" if r["note"] else "") for r in rows]
    hits = sum(r["result"].startswith("HIT") for r in rows)
    scored = sum(r["result"].startswith(("HIT", "MISS")) for r in rows)
    lines.append(f"\n{hits} hits of {scored} scored lines (Q10 is a tripwire report, not scored)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="also write the scored lines as JSON")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    rows = score(load(args.capacity), load(args.predictions))
    print(render(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
