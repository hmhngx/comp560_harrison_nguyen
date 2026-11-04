"""Score the locked E6 (longer-run capacity) predictions against the extension analysis, mechanically.

The rules were written down before any E6 run existed: a scored line is a HIT when the actual value lies inside the
locked [low, high] (or equals the locked answer), and nothing else counts - no rounding, no "close". The check that
the first 1000 epochs equal E4, the trend of each width and the two tripwires are reported next to the scored lines
but never counted.

    python phonebook/score_e6_predictions.py --extension phonebook/results/e6_extension.json \\
        --predictions phonebook/preregistration/e6_predictions.json [--json scores.json]

--extension is the JSON written by analyze_extension.py, keyed by embedding width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WIDTHS = ("8", "16", "32", "64")
LARGEST = "4000"


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

    def shown(spec):
        return f"{spec['point']} [{spec['low']}, {spec['high']}]"

    q = predictions["q1_seeds_perfect_by_4000"]
    for d in WIDTHS:
        spec, actual = q[f"d{d}"], report[d]["perfect_by"][LARGEST]
        add("Q1", f"seeds perfect by 4000 epochs, d={d}", shown(spec), actual,
            verdict(in_range(actual, spec["low"], spec["high"]), actual == spec["point"]))

    q = predictions["q2_rung_passes_at_4000"]
    for d in WIDTHS:
        actual = report[d]["passes"][LARGEST]
        add("Q2", f"rung passes at 4000 epochs, d={d}", q[f"d{d}"], actual, verdict(actual == q[f"d{d}"]))

    q = predictions["q3_first_passing_budget"]
    for d in ("8", "16", "32"):
        spec, actual = q[f"d{d}"], report[d]["first_pass_budget"]
        add("Q3", f"first passing budget, d={d}", shown(spec), actual,
            verdict(in_range(actual, spec["low"], spec["high"]), actual == spec["point"]))

    spec = predictions["q4_widths_passing_at_4000"]
    passing = sum(report[d]["passes"][LARGEST] for d in WIDTHS)
    add("Q4", "widths whose rung passes at 4000 epochs", shown(spec), passing,
        verdict(in_range(passing, spec["low"], spec["high"]), passing == spec["point"]))

    q = predictions["q5_d64_last_windows"]
    for key, what, actual in (("mean_accuracy_3001_4000", "d=64 mean accuracy, epochs 3001-4000", report["64"]["mean_accuracy"]["3001_4000"]),
                              ("gain_vs_1001_2000", "d=64 gain over epochs 1001-2000", report["64"]["gain"])):
        spec = q[key]
        add("Q5", what, shown(spec), round(actual, 4), verdict(in_range(actual, spec["low"], spec["high"]), actual == spec["point"]))

    q = predictions["q6_still_perfect_at_epoch_4000"]
    wide = sum(report[d]["perfect_at_end"] for d in ("16", "32"))
    spec = q["d16_plus_d32_runs"]
    add("Q6", "d=16 and d=32 runs perfect at epoch 4000 (of 6)", shown(spec), wide,
        verdict(in_range(wide, spec["low"], spec["high"]), wide == spec["point"]),
        f"{report['16']['reached_perfect'] + report['32']['reached_perfect']} of the 6 reached 100% at some epoch")
    narrow = report["8"]["perfect_at_end"]
    add("Q6", "d=8 runs perfect at epoch 4000 (of 3)", f"at least {q['d8_at_least']}", narrow, verdict(narrow >= q["d8_at_least"]))

    q = predictions["q7_control_book"]
    violations = sum(report[d]["control_violations"] for d in WIDTHS)
    add("Q7", "runs that recalled any control entry", q["max_control_violations"], violations,
        verdict(violations <= q["max_control_violations"], violations == q["max_control_violations"]))

    matches = [report[d]["first_1000_matches_earlier"] for d in WIDTHS]
    result = "CHECK FAILED" if any(m is False for m in matches) else "CHECK NOT RUN" if any(m is None for m in matches) else "CHECK PASSED"
    add("CHECK", "first 1000 epochs of every run equal the E4 run", "all equal", dict(zip(WIDTHS, matches)), result)

    for d in WIDTHS:
        entry = report[d]
        add("TREND", f"trend, d={d}", "", f"{entry['trend']} (gain {entry['gain']:+.4f})", "REPORTED")

    q = predictions["q8_results_that_would_change_my_mind"]
    d64_passes = report["64"]["passes"][LARGEST]
    add("TRIPWIRE", "d=64 passes at 4000 epochs", "no", d64_passes, "TRIPWIRE FIRED" if d64_passes else "TRIPWIRE NOT FIRED",
        q["tripwire_d64_passes"])
    both_fail = not report["16"]["passes"][LARGEST] and not report["32"]["passes"][LARGEST]
    add("TRIPWIRE", "d=16 and d=32 both fail at 4000 epochs", "no", both_fail, "TRIPWIRE FIRED" if both_fail else "TRIPWIRE NOT FIRED",
        q["tripwire_d16_and_d32_both_fail"])
    return rows


def render(rows: list[dict]) -> str:
    lines = [f"{r['q']:9} {r['what']:52} predicted {str(r['predicted']):22} actual {str(r['actual']):30} {r['result']}"
             + (f"   [{r['note']}]" if r["note"] else "") for r in rows]
    hits = sum(r["result"].startswith("HIT") for r in rows)
    scored = sum(r["result"].startswith(("HIT", "MISS")) for r in rows)
    lines.append(f"\n{hits} hits of {scored} scored lines (the check, the trends and the tripwires are reported, not scored)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="also write the scored lines as JSON")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    rows = score(load(args.extension), load(args.predictions))
    print(render(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
