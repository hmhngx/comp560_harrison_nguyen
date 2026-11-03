"""Score the locked E5 predictions against the E5 analysis, mechanically.

The rules were written down before any E5 result was read: a line is a HIT when the actual value lies inside the
locked [low, high] (or equals the locked answer), and nothing else counts - no rounding, no "close". Ties between
k values are broken toward the smaller k, the same rule the analysis uses. A censored (null) median counts as
infinitely slow.

    python phonebook/score_e5_predictions.py --primary phonebook/results/e5_analysis.json \\
        --robust phonebook/results/e5_robust_analysis.json --cells phonebook/results/e5_cells.csv \\
        --predictions phonebook/preregistration/e5_predictions.json [--json scores.json]

--primary is analyze_replay_k.py at lr 0.001 (k 1 2 3 4 5 9); --robust is the same tool at lr 0.003 and 0.0003
(k 1 2 3 5), run separately because k differs by learning rate; --cells is the cells CSV either call writes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

PRIMARY_LR, LOW_LR = "0.001", "0.0003"
MAIN_RUNS = 270


def _num(value):
    return math.inf if value is None else value


def argmin_k(values: dict) -> int:
    """The k with the smallest value; ties go to the smaller k, a null value is infinitely slow."""
    return min((int(k) for k in values), key=lambda k: (_num(values[str(k)]), k))


def in_range(value, low, high) -> bool:
    return value is not None and low <= value <= high


def verdict(ok: bool, exact: bool = False) -> str:
    if not ok:
        return "MISS"
    return "HIT (exact)" if exact else "HIT"


def count_censored_main(cells_csv: Path, main_seed: int = 42) -> tuple[int, int]:
    """(runs without a stable joint recall, runs counted) among the main-grid cells; the noise arm uses other seeds."""
    total = censored = 0
    with Path(cells_csv).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if int(row["seed"]) != main_seed:
                continue
            total += 1
            censored += row["stable_joint"] == ""
    return censored, total


def score(primary: dict, robust: dict, cells_csv: Path, predictions: dict) -> list[dict]:
    p = primary["lr"][PRIMARY_LR]
    band = primary["noise"]["band"]
    rows = []

    def add(q, what, predicted, actual, result, note=""):
        rows.append({"q": q, "what": what, "predicted": predicted, "actual": actual, "result": result, "note": note})

    q = predictions["q1_k1_median_updates"]
    actual = p["median_stable_joint"]["1"]
    add("Q1", "median stable_joint at k=1", f"{q['point']} [{q['low']}, {q['high']}]", actual,
        verdict(in_range(actual, q["low"], q["high"]), actual == q["point"]))

    q = predictions["q2_phonebooks_faster_than_k1_of_6"]
    for k in ("2", "3", "4", "5", "9"):
        spec, entry = q[f"k{k}"], p["paired"][k]
        add("Q2", f"phonebooks faster than k=1, k={k}", f"{spec['point']} [{spec['low']}, {spec['high']}]", entry["faster"],
            verdict(in_range(entry["faster"], spec["low"], spec["high"]), entry["faster"] == spec["point"]),
            f"tied {entry['tied']}, slower {entry['slower']}, median gain {entry['median_gain']}")

    q = predictions["q3_noise_band_updates"]
    add("Q3", "noise band", f"{q['point']} [{q['low']}, {q['high']}]", band,
        verdict(in_range(band, q["low"], q["high"]), band == q["point"]))

    q = predictions["q4_any_k_beats_k1"]
    winners = sorted(int(k) for k in ("2", "3", "4", "5", "9") if p["paired"][k]["beats"])
    add("Q4", "k that beat k=1 under the rule", f"{q['answer']}, {q['which']}", winners or "none",
        verdict(q["answer"] == "yes" and winners == sorted(q["which"])), f"band {band}")

    q = predictions["q5_plateau"]
    plateau = p["plateau"]
    expected = q["plateau"] if band == 0 else q["if_noise_band_is_1"] if band == 1 else None
    ok = expected is not None and plateau == expected     # equality already rules out the locked "excluded" ks
    add("Q5", "plateau", f"{q['plateau']} (band 0) / {q['if_noise_band_is_1']} (band 1)", plateau, verdict(ok),
        f"actual band {band}" + ("" if expected is not None else "; band outside the predicted cases"))

    q = predictions["q6_cheapest_and_fastest"]
    cheapest, fastest = p["cheapest_k_in_work"], argmin_k(p["median_stable_joint"])
    add("Q6", "cheapest k in work", q["cheapest_k_in_work"], cheapest, verdict(cheapest == q["cheapest_k_in_work"]))
    add("Q6", "fastest k in updates", q["fastest_k_in_updates"], fastest, verdict(fastest == q["fastest_k_in_updates"]),
        f"medians {p['median_stable_joint']}")

    q = predictions["q7_dip_fewest_a_recalled"]
    for k in ("1", "9"):
        spec, actual = q[f"k{k}"], p["median_min_a"][k]
        add("Q7", f"median dip at k={k}", f"{spec['point']} [{spec['low']}, {spec['high']}]", actual,
            verdict(in_range(actual, spec["low"], spec["high"]), actual == spec["point"]))

    q = predictions["q8_runs_never_joint_of_270"]
    censored, total = count_censored_main(cells_csv)
    add("Q8", "main-spec runs with no stable joint recall", f"{q['point']} [{q['low']}, {q['high']}]", censored,
        verdict(total == MAIN_RUNS and in_range(censored, q["low"], q["high"]), censored == q["point"]),
        f"main-spec cells counted: {total}")

    q = predictions["q9_best_k_at_lr_0.0003_vs_0.001"]
    best_low, best_primary = argmin_k(robust["lr"][LOW_LR]["median_stable_joint"]), argmin_k(p["median_stable_joint"])
    relation = "larger" if best_low > best_primary else "smaller" if best_low < best_primary else "same"
    add("Q9", "best k at lr 0.0003 vs 0.001", q["answer"], relation, verdict(relation == q["answer"]),
        f"best k {best_low} at 0.0003, {best_primary} at 0.001")

    q = predictions["q10_result_that_would_change_my_mind"]
    tripwire = p["paired"]["4"]["faster"] >= 3 or p["paired"]["5"]["faster"] >= 3
    noisy = band >= 3
    add("Q10", "tripwires fired", "none", {"k4_or_k5_faster_in_3_plus": tripwire, "noise_band_3_plus": noisy},
        "STORY CHALLENGED" if tripwire or noisy else "STORY SURVIVES", q["tripwire"])
    return rows


def render(rows: list[dict]) -> str:
    lines = [f"{r['q']:4} {r['what']:46} predicted {str(r['predicted']):28} actual {str(r['actual']):22} {r['result']}"
             + (f"   [{r['note']}]" if r["note"] else "") for r in rows]
    hits = sum(r["result"].startswith("HIT") for r in rows)
    scored = sum(r["result"].startswith(("HIT", "MISS")) for r in rows)
    lines.append(f"\n{hits} hits of {scored} scored lines (Q10 is a tripwire report, not scored)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--robust", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="also write the scored lines as JSON")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    rows = score(load(args.primary), load(args.robust), args.cells, load(args.predictions))
    print(render(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
