"""Score the locked E7 (optimizer settings at the wide widths) predictions against the extension analyses, mechanically.

The rules were written down before any E7 run existed: a scored line is a HIT when the actual value lies inside the
locked [low, high] (or equals the locked answer), and nothing else counts - no rounding, no "close". The check that the
first 2000 epochs of every arm S run equal E6, the trend of each width in both arms and the tripwires are reported next to
the scored lines but never counted.

    python phonebook/analyze_extension.py --root <arm W evidence> --budgets 1000 2000 3000 4000 --epochs-at-100 \\
        --prefix e7w --out phonebook/results
    python phonebook/analyze_extension.py --root <arm S evidence> --earlier-root <E6 evidence> --earlier-label E6 \\
        --prefix-epochs 2000 --budgets 1000 2000 3000 4000 --epochs-at-100 --prefix e7s --out phonebook/results
    python phonebook/score_e7_predictions.py --w phonebook/results/e7w_extension.json --s phonebook/results/e7s_extension.json \\
        --predictions phonebook/preregistration/e7_predictions.json [--json scores.json]

--w and --s are the JSON files written by analyze_extension.py for arm W and arm S, keyed by embedding width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("W", "S")
WIDTHS = ("16", "32", "64")
WIDE = ("16", "32")                      # the tripwires about holding look at these two widths; width 64 has tripwires of its own
LARGEST = "4000"
EARLY, LATE = "1001_2000", "3001_4000"
SEEDS = 3
PASS_SEEDS = 2                           # a rung passes at a budget when at least this many seeds are perfect by then
HOLD_MIN = 50                            # a seed holds when at least this many of its 100 logged late epochs have every entry recalled
LATE_LOGGED = 100
PREFIX_KEY = "first_2000_matches_earlier"
WITHIN = 0.003                           # "within 0.003 of E6's mean" (tripwires)
LIFT = 0.980                             # a width-64 mean below this is "not lifted" (tripwire)
MIN_HOLDERS = 2                          # "holds in fewer than 2 seeds" (tripwires)


def in_range(value, low, high) -> bool:
    return value is not None and low <= value <= high


def verdict(ok: bool, exact: bool = False) -> str:
    if not ok:
        return "MISS"
    return "HIT (exact)" if exact else "HIT"


def shown(spec) -> str:
    return f"{spec['point']} [{spec['low']}, {spec['high']}]"


def gap(a: float, b: float) -> float:
    """a - b without the float error of subtracting two decimal numbers: 0.9128 - 0.9078 is 0.00499999999999989, not 0.005."""
    return round(a - b, 10)


def spec_for(predictions: dict, key: str, arm: str, d: str):
    """The locked line of one arm and width in one group of the predictions."""
    try:
        return predictions[key][arm][f"d{d}"]
    except KeyError:
        raise ValueError(f"the predictions have no {key} line for arm {arm}, d={d}") from None


def holders(entry: dict) -> int:
    """Seeds that hold: at least HOLD_MIN of the LATE_LOGGED logged epochs of 3001-4000 have every entry recalled."""
    return sum(count[LATE] >= HOLD_MIN for count in entry["epochs_at_100"]["by_seed"].values())


def late_counts(entry: dict) -> str:
    """How close each seed is to the holding line: its logged late epochs at 100 percent."""
    counts = [str(count[LATE]) for _, count in sorted(entry["epochs_at_100"]["by_seed"].items(), key=lambda item: int(item[0]))]
    return f"epochs at 100% of the {LATE_LOGGED} logged in 3001-4000, per seed: " + ", ".join(counts)


REPORT_KEYS = ("seeds", "n", "perfect_by", "passes", "first_pass_budget", "perfect_at_end", "mean_accuracy", "gain", "trend", "control_violations")
E6_KEYS = ("mean_accuracy_3001_4000", "seeds_that_hold", "margin_for_beating_e6")


def check_inputs(reports: dict, predictions: dict) -> None:
    if predictions.get("schema") != "e7-predictions-v1":
        raise ValueError(f"the predictions have schema {predictions.get('schema')!r}; this scorer reads 'e7-predictions-v1'")
    for key in ("rungs", "e6_control", "tripwires"):
        if key not in predictions:
            raise ValueError(f"the predictions have no {key}")
    missing = [key for key in E6_KEYS if key not in predictions["e6_control"]]
    if missing:
        raise ValueError(f"the predictions' e6_control lacks {', '.join(missing)}")
    for arm in ARMS:
        if arm not in reports:
            raise ValueError(f"no report for arm {arm}")
        for d in WIDTHS:
            where = f"arm {arm}, d={d}"
            if d not in reports[arm]:
                raise ValueError(f"{where}: not in the report")
            entry = reports[arm][d]
            missing = [key for key in REPORT_KEYS if key not in entry]
            if missing:
                raise ValueError(f"{where}: the report lacks {', '.join(missing)}")
            if entry["seeds"] != SEEDS:
                raise ValueError(f"{where}: the report has {entry['seeds']} seeds, the predictions are for {SEEDS} seeds")
            rung = predictions["rungs"][f"d{d}"]
            if entry["n"] != rung:
                raise ValueError(f"{where}: the report is for {entry['n']} entries, the rung of the predictions is {rung}")
            if "epochs_at_100" not in entry:
                raise ValueError(f"{where}: no epochs_at_100 in the report; run analyze_extension.py with --epochs-at-100")
            logged = entry["epochs_at_100"]["logged"][LATE]
            if logged != LATE_LOGGED:
                raise ValueError(f"{where}: holding is defined on {LATE_LOGGED} logged epochs in 3001-4000, the report has {logged}")
            if len(entry["epochs_at_100"]["by_seed"]) != SEEDS:
                raise ValueError(f"{where}: epochs_at_100 lists {len(entry['epochs_at_100']['by_seed'])} seeds, the predictions are for {SEEDS} seeds")
            for budget, perfect in entry["perfect_by"].items():
                if entry["passes"][budget] != (perfect >= PASS_SEEDS):
                    raise ValueError(f"{where}: the report says the rung passes at {budget} epochs is {entry['passes'][budget]} with {perfect} seeds perfect "
                                     f"by then; a rung passes with at least {PASS_SEEDS} (run analyze_extension.py without --min-seeds)")
            if arm == "S" and PREFIX_KEY not in entry:
                raise ValueError(f"{where}: no {PREFIX_KEY} in the report; run analyze_extension.py with --prefix-epochs 2000")


def tripwire_rules(reports: dict, e6: dict, check: str) -> dict:
    """The rule of every tripwire, as the first sentence of its text in the locked file says it: a function that returns
    True when the tripwire fires, False when it does not, and None when it cannot be told (the check against E6 was not run)."""
    late = lambda arm, d: reports[arm][d]["mean_accuracy"][LATE]
    e6_mean = lambda d: e6["mean_accuracy_3001_4000"][f"d{d}"]
    near_e6 = lambda arm, d: abs(gap(late(arm, d), e6_mean(d))) <= WITHIN
    more_than_e6 = lambda arm, d: holders(reports[arm][d]) > e6["seeds_that_hold"][f"d{d}"]
    passes = lambda arm, d: reports[arm][d]["passes"][LARGEST]
    prefix_fails = {"CHECK FAILED": True, "CHECK PASSED": False, "CHECK NOT RUN": None}[check]
    return {
        "w_d64_passes": lambda: passes("W", "64"),
        "w_d64_fails_to_pass": lambda: not passes("W", "64"),
        "neither_arm_lifts_d64": lambda: late("W", "64") < LIFT and late("S", "64") < LIFT,
        "w_null_s_large": lambda: all(near_e6("W", d) for d in WIDTHS) and all(more_than_e6("S", d) for d in WIDE),
        "w_large_s_null": lambda: all(more_than_e6("W", d) for d in WIDE) and all(near_e6("S", d) for d in WIDE),
        "both_large": lambda: all(more_than_e6(arm, d) for arm in ARMS for d in WIDE),
        "w_hurts": lambda: any(late("W", d) < e6_mean(d) for d in WIDTHS),
        "s_prefix_fails": lambda: prefix_fails,
        "s_d64_does_not_hold": lambda: holders(reports["S"]["64"]) < MIN_HOLDERS,
        "real_books_do_not_replicate": lambda: any(holders(reports[arm][d]) < MIN_HOLDERS for arm in ARMS for d in WIDE),
    }


def score(reports: dict, predictions: dict) -> list[dict]:
    """reports: {"W": the arm W analysis, "S": the arm S analysis}, each keyed by width; predictions: the locked file."""
    check_inputs(reports, predictions)
    rows = []

    def add(q, what, predicted, actual, result, note=""):
        rows.append({"q": q, "what": what, "predicted": predicted, "actual": actual, "result": result, "note": note})

    def lines(q, key, what, read, optional=False, note=None):
        """One scored line per arm and width: read(entry) is the actual value; a count or a budget is a closed range."""
        for arm in ARMS:
            for d in WIDTHS:
                if optional and f"d{d}" not in predictions[key][arm] and not spec_for(predictions, "q2_rung_passes_at_4000", arm, d):
                    continue                                                        # a rung predicted not to pass has no such line
                spec, entry = spec_for(predictions, key, arm, d), reports[arm][d]
                actual = read(entry)
                add(q, f"{what}, arm {arm}, d={d}", shown(spec), actual,
                    verdict(in_range(actual, spec["low"], spec["high"]), actual == spec["point"]), note(entry) if note else "")

    lines("Q1", "q1_seeds_perfect_by_4000", "seeds perfect by 4000 epochs", lambda e: e["perfect_by"][LARGEST])

    for arm in ARMS:
        for d in WIDTHS:
            predicted, actual = spec_for(predictions, "q2_rung_passes_at_4000", arm, d), reports[arm][d]["passes"][LARGEST]
            add("Q2", f"rung passes at 4000 epochs, arm {arm}, d={d}", predicted, actual, verdict(actual == predicted))

    lines("Q3", "q3_first_passing_budget", "first passing budget", lambda e: e["first_pass_budget"], optional=True)
    lines("Q4", "q4_seeds_that_hold_100_percent", "seeds that hold 100 percent", holders, note=late_counts)
    lines("Q5", "q5_seeds_perfect_at_epoch_4000", "seeds perfect at epoch 4000", lambda e: e["perfect_at_end"], optional=True)

    q = predictions["q6_d64_windows"]
    windows = (("W", "mean_accuracy_1001_2000", "mean accuracy, epochs 1001-2000", lambda e: e["mean_accuracy"][EARLY]),
               ("W", "mean_accuracy_3001_4000", "mean accuracy, epochs 3001-4000", lambda e: e["mean_accuracy"][LATE]),
               ("W", "gain_vs_1001_2000", "gain over epochs 1001-2000", lambda e: e["gain"]),
               ("S", "mean_accuracy_3001_4000", "mean accuracy, epochs 3001-4000", lambda e: e["mean_accuracy"][LATE]),
               ("S", "gain_vs_1001_2000", "gain over epochs 1001-2000", lambda e: e["gain"]))
    for arm, key, what, read in windows:
        spec, actual = q[arm][key], read(reports[arm]["64"])
        add("Q6", f"d=64 arm {arm} {what}", shown(spec), round(actual, 4),
            verdict(in_range(actual, spec["low"], spec["high"]), actual == spec["point"]))

    e6 = predictions["e6_control"]
    for arm in ARMS:
        for d in WIDTHS:
            mean, e6_mean = reports[arm][d]["mean_accuracy"][LATE], e6["mean_accuracy_3001_4000"][f"d{d}"]
            predicted, actual = spec_for(predictions, "q7_late_mean_beats_e6", arm, d)["answer"], gap(mean, e6_mean) >= e6["margin_for_beating_e6"]
            add("Q7", f"3001-4000 mean beats E6 by the margin, arm {arm}, d={d}", predicted, actual, verdict(actual == predicted),
                f"{mean:.4f} against E6 {e6_mean:.4f}")

    q = predictions["q8_control_book"]
    violations = sum(reports[arm][d]["control_violations"] for arm in ARMS for d in WIDTHS)
    add("Q8", "runs that recalled any control entry (of 18)", q["max_control_violations"], violations,
        verdict(violations <= q["max_control_violations"], violations == q["max_control_violations"]))

    q = predictions["q9_contrasts"]
    w_late, s_late = reports["W"]["64"]["mean_accuracy"][LATE], reports["S"]["64"]["mean_accuracy"][LATE]
    higher = "W" if w_late > s_late else "S" if s_late > w_late else "tie"
    add("Q9", "d=64: arm with the higher 3001-4000 mean", q["d64_higher_3001_4000_mean"]["answer"], higher,
        verdict(higher == q["d64_higher_3001_4000_mean"]["answer"]))
    w_hold, s_hold = holders(reports["W"]["16"]), holders(reports["S"]["16"])
    more = "W" if w_hold > s_hold else "S" if s_hold > w_hold else "tie"
    add("Q9", "d=16: arm with more holding seeds", q["d16_more_holding_seeds"]["answer"], more,
        verdict(more == q["d16_more_holding_seeds"]["answer"]))

    matches = [reports["S"][d][PREFIX_KEY] for d in WIDTHS]
    check = "CHECK FAILED" if any(m is False for m in matches) else "CHECK NOT RUN" if any(m is None for m in matches) else "CHECK PASSED"
    add("CHECK", "first 2000 epochs of every arm S run equal the E6 run", "all equal", dict(zip(WIDTHS, matches)), check)

    for arm in ARMS:
        for d in WIDTHS:
            entry = reports[arm][d]
            add("TREND", f"trend, arm {arm}, d={d}", "", f"{entry['trend']} (gain {entry['gain']:+.4f})", "REPORTED")

    rules = tripwire_rules(reports, e6, check)
    for name, text in predictions["tripwires"].items():
        if name not in rules:
            raise ValueError(f"the scorer has no rule for the tripwire {name!r}")
        fires = rules[name]()
        add("TRIPWIRE", name, "", fires, {True: "TRIPWIRE FIRED", False: "TRIPWIRE NOT FIRED", None: "TRIPWIRE NOT RUN"}[fires], text)
    return rows


def render(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        line = f"{r['q']:9} {r['what']:64} predicted {str(r['predicted']):22} actual {str(r['actual']):30} {r['result']}"
        if r["note"] and (r["q"] != "TRIPWIRE" or r["result"] == "TRIPWIRE FIRED"):          # a tripwire's reading is shown when it fired
            line += f"   [{r['note']}]"
        lines.append(line)
    hits = sum(r["result"].startswith("HIT") for r in rows)
    scored = sum(r["result"].startswith(("HIT", "MISS")) for r in rows)
    lines.append(f"\n{hits} hits of {scored} scored lines (the check, the trends and the tripwires are reported, not scored)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w", type=Path, required=True, help="analyze_extension.py output for arm W")
    parser.add_argument("--s", type=Path, required=True, help="analyze_extension.py output for arm S")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="also write the scored lines as JSON")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    rows = score({"W": load(args.w), "S": load(args.s)}, load(args.predictions))
    print(render(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
