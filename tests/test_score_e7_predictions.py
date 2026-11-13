"""The E7 scorer applies rules fixed before any result existed; each rule is pinned here by a hit and a miss.

PREDICTIONS and the reports below are made up, with different numbers in every cell so that a line that reads the wrong arm or
width is caught; the last section pins the adopted file itself.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from phonebook import score_e7_predictions as sp
from phonebook.predictions import load_predictions

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "phonebook" / "preregistration" / "e7_predictions.json"
ARMS = ("W", "S")
WIDTHS = ("16", "32", "64")
RUNGS = {"16": 160, "32": 320, "64": 640}
TRIPWIRES = ("w_d64_passes", "w_d64_fails_to_pass", "neither_arm_lifts_d64", "w_null_s_large", "w_large_s_null", "both_large",
             "w_hurts", "s_prefix_fails", "s_d64_does_not_hold", "real_books_do_not_replicate")
NOT_SCORED = "(the check, the trends and the tripwires are reported, not scored)"


def rng(point, low, high):
    return {"point": point, "low": low, "high": high}


def answer(value, confidence=0.9):
    return {"answer": value, "confidence": confidence}


def per_arm(w16, w32, w64, s16, s32, s64):
    return {"W": {"d16": w16, "d32": w32, "d64": w64}, "S": {"d16": s16, "d32": s32, "d64": s64}}


PREDICTIONS = {
    "schema": "e7-predictions-v1",
    "rungs": {"d16": 160, "d32": 320, "d64": 640},
    "e6_control": {"mean_accuracy_3001_4000": {"d16": 0.95, "d32": 0.97, "d64": 0.975},
                   "seeds_that_hold": {"d16": 0, "d32": 0, "d64": 0}, "margin_for_beating_e6": 0.005},
    "q1_seeds_perfect_by_4000": per_arm(rng(3, 3, 3), rng(2, 2, 3), rng(2, 1, 3), rng(3, 2, 3), rng(3, 2, 3), rng(3, 3, 3)),
    "q2_rung_passes_at_4000": {**per_arm(True, True, True, True, True, True),
                               "confidence_of_this_answer": per_arm(0.99, 0.99, 0.97, 0.99, 0.99, 0.98)},
    "q3_first_passing_budget": per_arm(rng(2000, 2000, 3000), rng(2000, 2000, 3000), rng(3000, 3000, 4000),
                                       rng(3000, 3000, 3000), rng(3000, 3000, 4000), rng(4000, 3000, 4000)),
    "q4_seeds_that_hold_100_percent": per_arm(rng(3, 3, 3), rng(3, 2, 3), rng(3, 3, 3), rng(3, 3, 3), rng(3, 2, 3), rng(2, 2, 3)),
    "q5_seeds_perfect_at_epoch_4000": per_arm(rng(3, 3, 3), rng(2, 2, 3), rng(2, 2, 3), rng(3, 3, 3), rng(3, 2, 3), rng(3, 3, 3)),
    "q6_d64_windows": {"W": {"mean_accuracy_1001_2000": rng(0.991, 0.98, 0.998), "mean_accuracy_3001_4000": rng(0.995, 0.99, 1.0),
                             "gain_vs_1001_2000": rng(0.007, 0.001, 0.018)},
                       "S": {"mean_accuracy_3001_4000": rng(1.0, 0.99, 1.0), "gain_vs_1001_2000": rng(0.03, 0.0216, 0.0316)}},
    "q7_late_mean_beats_e6": per_arm(answer(True), answer(True), answer(True), answer(True), answer(False), answer(True)),
    "q8_control_book": {"max_control_violations": 0},
    "q9_contrasts": {"d64_higher_3001_4000_mean": answer("S", 0.6), "d16_more_holding_seeds": answer("tie", 0.8)},
    "tripwires": {name: f"Fires when {name} is true. Reading: the reading of {name}." for name in TRIPWIRES},
}


def held(k: int, *, late: int = 100, short: int = 3, logged: int = 100) -> dict:
    """epochs_at_100 of three seeds: the first k are at 100 percent on `late` of the `logged` late epochs, the others on `short`."""
    return {"logged": {"all": 400, "3001_4000": logged},
            "by_seed": {str(s): {"all": 250 if s <= k else 9, "3001_4000": late if s <= k else short} for s in (1, 2, 3)}}


def width_report(d, *, by_4000, first_pass, at_end, holders, mean, early=None, gain=0.0, trend="flat", violations=0, match=None):
    """One width of one arm, shaped like analyze_extension.py --epochs-at-100 --prefix-epochs 2000 (match is None without --earlier-root)."""
    return {"n": RUNGS[d], "seeds": 3,
            "perfect_by": {"1000": 0, "2000": 0, "3000": 0, "4000": by_4000},
            "passes": {"1000": False, "2000": False, "3000": False, "4000": by_4000 >= 2},
            "first_pass_budget": first_pass, "reached_perfect": by_4000, "perfect_at_end": at_end,
            "mean_accuracy": {"1001_2000": mean - gain if early is None else early, "3001_4000": mean},
            "gain": gain, "trend": trend, "control_violations": violations,
            "first_2000_match_by_seed": None if match is None else {"1": match, "2": match, "3": match},
            "first_2000_matches_earlier": match, "epochs_at_100": held(holders)}


@pytest.fixture
def reports() -> dict:
    """Every scored line lands on its prediction (19 of them exactly); the six cells differ so that a wrong cell is noticed."""
    return {
        "W": {"16": width_report("16", by_4000=3, first_pass=2000, at_end=3, holders=3, mean=0.99, gain=0.02, trend="rising"),
              "32": width_report("32", by_4000=3, first_pass=3000, at_end=2, holders=2, mean=0.98),
              "64": width_report("64", by_4000=2, first_pass=3000, at_end=3, holders=3, mean=0.998, early=0.991, gain=0.007,
                                 trend="rising")},
        "S": {"16": width_report("16", by_4000=3, first_pass=3000, at_end=3, holders=3, mean=0.96, gain=-0.01, trend="falling",
                                 match=True),
              "32": width_report("32", by_4000=2, first_pass=4000, at_end=2, holders=3, mean=0.97, gain=0.001, match=True),
              "64": width_report("64", by_4000=3, first_pass=3000, at_end=3, holders=3, mean=1.0, early=0.9684, gain=0.0316,
                                 trend="rising", match=True)}}


def scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["result"].startswith(("HIT", "MISS"))]


def row(rows: list[dict], q: str, what_contains: str = "") -> dict:
    return next(r for r in rows if r["q"] == q and what_contains in r["what"])


def line(rows: list[dict], q: str, what_contains: str = "") -> str:
    return row(rows, q, what_contains)["result"]


def fired(rows: list[dict]) -> set:
    return {r["what"] for r in rows if r["q"] == "TRIPWIRE" and r["result"] == "TRIPWIRE FIRED"}


def flips(before: list[dict], after: list[dict]) -> set:
    """The scored lines whose verdict changed between hit and miss."""
    verdicts = lambda rows: {(r["q"], r["what"]): r["result"].startswith("HIT") for r in scored(rows)}
    b, a = verdicts(before), verdicts(after)
    assert b.keys() == a.keys()
    return {key for key in b if b[key] != a[key]}


def put_late(reports: dict, arm: str, d: str, value: float) -> None:
    reports[arm][d]["mean_accuracy"]["3001_4000"] = value


def put_holders(reports: dict, arm: str, d: str, k: int) -> None:
    reports[arm][d]["epochs_at_100"] = held(k)


def put_perfect(reports: dict, arm: str, d: str, seeds: int, budget: str = "4000") -> None:
    """Seeds perfect by a budget; `passes` follows, because a rung passes with at least 2 of 3 seeds perfect (the scorer refuses a report that says otherwise)."""
    reports[arm][d]["perfect_by"][budget] = seeds
    reports[arm][d]["passes"][budget] = seeds >= 2


def put_passes(reports: dict, arm: str, d: str, flag: bool) -> None:
    """Whether the rung passes at 4000, with the seeds perfect by then made to agree (3 if it passes, 0 if not)."""
    put_perfect(reports, arm, d, 3 if flag else 0)


# ------------------------------------------------------------------ the scored lines

def test_a_report_that_lands_on_every_prediction_is_all_hits(reports) -> None:
    rows = sp.score(reports, PREDICTIONS)
    assert len(scored(rows)) == 44 and all(r["result"].startswith("HIT") for r in scored(rows))
    assert sum(r["result"] == "HIT (exact)" for r in rows) == 19                  # Q1 4, Q3 3, Q4 4, Q5 4, Q6 3, Q8 1
    assert len(rows) == 44 + 1 + 6 + 10
    assert sp.render(rows).splitlines()[-1] == f"44 hits of 44 scored lines {NOT_SCORED}"


def test_the_lines_come_by_question_then_arm_then_width(reports) -> None:
    rows = sp.score(reports, PREDICTIONS)
    assert [r["q"] for r in rows] == (["Q1"] * 6 + ["Q2"] * 6 + ["Q3"] * 6 + ["Q4"] * 6 + ["Q5"] * 6 + ["Q6"] * 5 + ["Q7"] * 6
                                      + ["Q8", "Q9", "Q9", "CHECK"] + ["TREND"] * 6 + ["TRIPWIRE"] * 10)
    assert [r["what"] for r in rows if r["q"] == "Q1"] == [f"seeds perfect by 4000 epochs, arm {a}, d={d}" for a in ARMS for d in WIDTHS]
    assert [r["what"] for r in rows if r["q"] == "TRIPWIRE"] == list(TRIPWIRES)       # the order of the file


def test_a_scored_row_carries_the_prediction_and_the_actual_value(reports) -> None:
    assert row(sp.score(reports, PREDICTIONS), "Q1", "arm W, d=32") == {
        "q": "Q1", "what": "seeds perfect by 4000 epochs, arm W, d=32", "predicted": "2 [2, 3]", "actual": 3, "result": "HIT", "note": ""}


@pytest.mark.parametrize("seeds, expected", [(2, "HIT (exact)"), (3, "HIT"), (1, "MISS"), (0, "MISS")])
def test_q1_the_count_of_perfect_seeds_is_a_closed_range(reports, seeds, expected) -> None:
    put_perfect(reports, "W", "32", seeds)                                         # predicted 2 [2, 3]
    assert line(sp.score(reports, PREDICTIONS), "Q1", "arm W, d=32") == expected


def test_q1_reads_the_4000_epoch_column_only(reports) -> None:
    reports["S"]["64"]["perfect_by"].update({"1000": 1, "2000": 1, "3000": 1})      # predicted 3 [3, 3]
    assert line(sp.score(reports, PREDICTIONS), "Q1", "arm S, d=64") == "HIT (exact)"
    put_perfect(reports, "S", "64", 3, "3000")
    put_perfect(reports, "S", "64", 2)
    assert line(sp.score(reports, PREDICTIONS), "Q1", "arm S, d=64") == "MISS"


@pytest.mark.parametrize("predicted, actual, expected", [(True, True, "HIT"), (True, False, "MISS"), (False, False, "HIT"), (False, True, "MISS")])
def test_q2_the_rung_either_passes_or_not_as_predicted(reports, predicted, actual, expected) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["q2_rung_passes_at_4000"]["S"]["d32"] = predicted
    put_passes(reports, "S", "32", actual)
    assert line(sp.score(reports, predictions), "Q2", "arm S, d=32") == expected


def test_q2_reads_the_4000_epoch_column_only(reports) -> None:
    reports["W"]["16"]["perfect_by"].update({"1000": 3, "2000": 3, "3000": 3, "4000": 1})
    reports["W"]["16"]["passes"].update({"1000": True, "2000": True, "3000": True, "4000": False})
    assert line(sp.score(reports, PREDICTIONS), "Q2", "arm W, d=16") == "MISS"


@pytest.mark.parametrize("budget, expected", [(3000, "HIT (exact)"), (4000, "HIT"), (2000, "MISS"), (1000, "MISS"), (None, "MISS")])
def test_q3_the_first_passing_budget_is_a_closed_range_and_never_passing_is_a_miss(reports, budget, expected) -> None:
    reports["S"]["32"]["first_pass_budget"] = budget                               # predicted 3000 [3000, 4000]
    assert line(sp.score(reports, PREDICTIONS), "Q3", "arm S, d=32") == expected


def test_a_rung_predicted_not_to_pass_has_no_q3_and_q5_line(reports) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["q2_rung_passes_at_4000"]["S"]["d64"] = False
    predictions["q1_seeds_perfect_by_4000"]["S"]["d64"] = rng(1, 0, 1)
    del predictions["q3_first_passing_budget"]["S"]["d64"], predictions["q5_seeds_perfect_at_epoch_4000"]["S"]["d64"]
    put_perfect(reports, "S", "64", 1)
    rows = sp.score(reports, predictions)
    assert not [r for r in rows if r["q"] in ("Q3", "Q5") and "arm S, d=64" in r["what"]]
    assert len(scored(rows)) == 42 and all(r["result"].startswith("HIT") for r in scored(rows))


@pytest.mark.parametrize("holders, expected", [(2, "HIT (exact)"), (3, "HIT"), (1, "MISS"), (0, "MISS")])
def test_q4_counts_the_seeds_that_hold(reports, holders, expected) -> None:
    put_holders(reports, "S", "64", holders)                                       # predicted 2 [2, 3]
    assert line(sp.score(reports, PREDICTIONS), "Q4", "arm S, d=64") == expected


@pytest.mark.parametrize("late, holds", [(100, True), (50, True), (49, False), (0, False)])
def test_a_seed_holds_when_half_of_the_late_logged_epochs_are_at_100_percent(reports, late, holds) -> None:
    every_seed = {str(s): {"all": 0 if holds else 400, "3001_4000": late} for s in (1, 2, 3)}     # the whole-run count must not matter
    reports["W"]["16"]["epochs_at_100"] = {"logged": {"all": 400, "3001_4000": 100}, "by_seed": every_seed}
    assert row(sp.score(reports, PREDICTIONS), "Q4", "arm W, d=16")["actual"] == (3 if holds else 0)


def test_q4_shows_the_late_counts_of_every_seed(reports) -> None:
    assert row(sp.score(reports, PREDICTIONS), "Q4", "arm W, d=32")["note"] == (
        "epochs at 100% of the 100 logged in 3001-4000, per seed: 100, 100, 3")


def test_q4_lists_the_seeds_in_seed_order_whatever_the_order_of_the_report(reports) -> None:
    counts = reports["W"]["32"]["epochs_at_100"]["by_seed"]
    reports["W"]["32"]["epochs_at_100"]["by_seed"] = {"3": counts["3"], "1": counts["1"], "2": counts["2"]}
    assert row(sp.score(reports, PREDICTIONS), "Q4", "arm W, d=32")["note"].endswith("per seed: 100, 100, 3")


@pytest.mark.parametrize("at_end, expected", [(2, "HIT (exact)"), (3, "HIT"), (1, "MISS"), (0, "MISS")])
def test_q5_counts_the_seeds_perfect_at_the_last_epoch(reports, at_end, expected) -> None:
    reports["W"]["64"]["perfect_at_end"] = at_end                                  # predicted 2 [2, 3]
    assert line(sp.score(reports, PREDICTIONS), "Q5", "arm W, d=64") == expected


Q6_WHAT = {("W", "early"): "d=64 arm W mean accuracy, epochs 1001-2000", ("W", "late"): "d=64 arm W mean accuracy, epochs 3001-4000",
           ("W", "gain"): "d=64 arm W gain over epochs 1001-2000", ("S", "late"): "d=64 arm S mean accuracy, epochs 3001-4000",
           ("S", "gain"): "d=64 arm S gain over epochs 1001-2000"}


def put_q6(reports: dict, arm: str, kind: str, value: float) -> None:
    entry = reports[arm]["64"]
    if kind == "gain":
        entry["gain"] = value
    else:
        entry["mean_accuracy"]["1001_2000" if kind == "early" else "3001_4000"] = value


@pytest.mark.parametrize("arm, kind, value, expected", [
    ("W", "early", 0.991, "HIT (exact)"), ("W", "early", 0.98, "HIT"), ("W", "early", 0.998, "HIT"), ("W", "early", 0.9799, "MISS"),
    ("W", "early", 0.9981, "MISS"),
    ("W", "late", 0.995, "HIT (exact)"), ("W", "late", 0.99, "HIT"), ("W", "late", 1.0, "HIT"), ("W", "late", 0.9899, "MISS"),
    ("W", "late", 1.0001, "MISS"),
    ("W", "gain", 0.007, "HIT (exact)"), ("W", "gain", 0.001, "HIT"), ("W", "gain", 0.018, "HIT"), ("W", "gain", 0.0009, "MISS"),
    ("W", "gain", 0.0181, "MISS"), ("W", "gain", -0.004, "MISS"),
    ("S", "late", 1.0, "HIT (exact)"), ("S", "late", 0.99, "HIT"), ("S", "late", 0.9899, "MISS"), ("S", "late", 1.0001, "MISS"),
    ("S", "gain", 0.03, "HIT (exact)"), ("S", "gain", 0.0216, "HIT"), ("S", "gain", 0.0316, "HIT"), ("S", "gain", 0.0215, "MISS"),
    ("S", "gain", 0.0317, "MISS")])
def test_q6_each_width_64_window_line_is_a_closed_range(reports, arm, kind, value, expected) -> None:
    put_q6(reports, arm, kind, value)
    assert line(sp.score(reports, PREDICTIONS), "Q6", Q6_WHAT[arm, kind]) == expected


def test_q6_shows_the_value_to_four_places_and_scores_the_exact_one(reports) -> None:
    put_q6(reports, "W", "gain", 0.0123456)
    r = row(sp.score(reports, PREDICTIONS), "Q6", Q6_WHAT["W", "gain"])
    assert (r["predicted"], r["actual"], r["result"]) == ("0.007 [0.001, 0.018]", 0.0123, "HIT")
    put_q6(reports, "W", "gain", 0.0180001)                                        # outside by a hair: the shown value rounds into the range
    r = row(sp.score(reports, PREDICTIONS), "Q6", Q6_WHAT["W", "gain"])
    assert (r["actual"], r["result"]) == (0.018, "MISS")


@pytest.mark.parametrize("mean, expected", [(0.955, "HIT"), (0.96, "HIT"), (0.9549, "MISS"), (0.95, "MISS"), (0.94, "MISS")])
def test_q7_a_late_mean_beats_e6_by_at_least_the_margin(reports, mean, expected) -> None:
    put_late(reports, "W", "16", mean)                                             # E6 0.95, margin 0.005, predicted yes
    assert line(sp.score(reports, PREDICTIONS), "Q7", "arm W, d=16") == expected


def test_q7_a_difference_of_exactly_the_margin_counts_even_when_the_float_subtraction_lands_just_under_it(reports) -> None:
    assert 0.9128 - 0.9078 < 0.005
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["e6_control"]["mean_accuracy_3001_4000"]["d16"] = 0.9078
    put_late(reports, "W", "16", 0.9128)
    assert line(sp.score(reports, predictions), "Q7", "arm W, d=16") == "HIT"


def test_q7_takes_the_e6_value_and_the_margin_from_the_predictions_file(reports) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["e6_control"]["margin_for_beating_e6"] = 0.05                      # W16's 0.99 is 0.04 above E6's 0.95
    assert line(sp.score(reports, predictions), "Q7", "arm W, d=16") == "MISS"
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["e6_control"]["mean_accuracy_3001_4000"]["d16"] = 0.99             # W16 is 0.99
    assert line(sp.score(reports, predictions), "Q7", "arm W, d=16") == "MISS"


def test_q7_a_predicted_no_is_a_hit_when_the_arm_does_not_beat_e6(reports) -> None:
    assert line(sp.score(reports, PREDICTIONS), "Q7", "arm S, d=32") == "HIT"       # predicted no, mean equals E6's
    put_late(reports, "S", "32", 0.99)
    assert line(sp.score(reports, PREDICTIONS), "Q7", "arm S, d=32") == "MISS"


def test_q7_shows_the_mean_against_the_e6_mean(reports) -> None:
    r = row(sp.score(reports, PREDICTIONS), "Q7", "arm W, d=16")
    assert (r["predicted"], r["actual"], r["note"]) == (True, True, "0.9900 against E6 0.9500")


@pytest.mark.parametrize("violations, allowed, expected", [(0, 0, "HIT (exact)"), (1, 0, "MISS"), (1, 1, "HIT (exact)"), (0, 1, "HIT"), (2, 1, "MISS")])
def test_q8_the_control_book_violations_are_summed_over_both_arms_and_all_widths(reports, violations, allowed, expected) -> None:
    reports["W"]["16"]["control_violations"] = violations // 2
    reports["S"]["64"]["control_violations"] = violations - violations // 2
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["q8_control_book"]["max_control_violations"] = allowed
    assert line(sp.score(reports, predictions), "Q8") == expected


@pytest.mark.parametrize("w_mean, s_mean, higher", [(0.998, 1.0, "S"), (1.0, 0.998, "W"), (1.0, 1.0, "tie")])
def test_q9_d64_names_the_arm_with_the_higher_late_mean_or_a_tie(reports, w_mean, s_mean, higher) -> None:
    put_late(reports, "W", "64", w_mean)
    put_late(reports, "S", "64", s_mean)
    for predicted in ("W", "S", "tie"):
        predictions = copy.deepcopy(PREDICTIONS)
        predictions["q9_contrasts"]["d64_higher_3001_4000_mean"]["answer"] = predicted
        r = row(sp.score(reports, predictions), "Q9", "d=64")
        assert (r["actual"], r["result"]) == (higher, "HIT" if predicted == higher else "MISS")


@pytest.mark.parametrize("w_holders, s_holders, more", [(3, 3, "tie"), (3, 1, "W"), (1, 3, "S"), (0, 0, "tie"), (2, 2, "tie")])
def test_q9_d16_names_the_arm_with_more_holding_seeds_or_a_tie(reports, w_holders, s_holders, more) -> None:
    put_holders(reports, "W", "16", w_holders)
    put_holders(reports, "S", "16", s_holders)
    for predicted in ("W", "S", "tie"):
        predictions = copy.deepcopy(PREDICTIONS)
        predictions["q9_contrasts"]["d16_more_holding_seeds"]["answer"] = predicted
        r = row(sp.score(reports, predictions), "Q9", "d=16")
        assert (r["actual"], r["result"]) == (more, "HIT" if predicted == more else "MISS")


# Each case changes one cell of the reports so that its own line misses; only the lines that read that cell may change.
CELL_WHAT = {"Q1": "seeds perfect by 4000 epochs", "Q2": "rung passes at 4000 epochs", "Q3": "first passing budget",
             "Q4": "seeds that hold 100 percent", "Q5": "seeds perfect at epoch 4000", "Q7": "3001-4000 mean beats E6 by the margin"}
Q9_D16 = ("Q9", "d=16: arm with more holding seeds")
Q9_D64 = ("Q9", "d=64: arm with the higher 3001-4000 mean")

def cell(q, arm, d):
    return (q, f"{CELL_WHAT[q]}, arm {arm}, d={d}")


def make_cases() -> list:
    cases = []
    def zero_perfect(r, arm, d): put_perfect(r, arm, d, 0)                         # passes follows: the Q1 and the Q2 line of the cell both miss
    def zero_passes(r, arm, d): put_passes(r, arm, d, False)
    def never_passes(r, arm, d): r[arm][d]["first_pass_budget"] = None
    def zero_holders(r, arm, d): put_holders(r, arm, d, 0)
    def zero_at_end(r, arm, d): r[arm][d]["perfect_at_end"] = 0
    def not_beating(r, arm, d):                                                    # a mean that flips the yes/no of its Q7 line
        beats = PREDICTIONS["q7_late_mean_beats_e6"][arm][f"d{d}"]["answer"]
        put_late(r, arm, d, PREDICTIONS["e6_control"]["mean_accuracy_3001_4000"][f"d{d}"] if beats else 0.99)
    for arm in ARMS:
        for d in WIDTHS:
            holders_also = {Q9_D16} if d == "16" else set()
            mean_also = {("Q6", Q6_WHAT[arm, "late"])} | ({Q9_D64} if (arm, d) == ("S", "64") else set()) if d == "64" else set()
            cases.extend([
                (f"Q1 {arm}{d}", zero_perfect, arm, d, {cell("Q1", arm, d), cell("Q2", arm, d)}),
                (f"Q2 {arm}{d}", zero_passes, arm, d, {cell("Q1", arm, d), cell("Q2", arm, d)}),
                (f"Q3 {arm}{d}", never_passes, arm, d, {cell("Q3", arm, d)}),
                (f"Q4 {arm}{d}", zero_holders, arm, d, {cell("Q4", arm, d)} | holders_also),
                (f"Q5 {arm}{d}", zero_at_end, arm, d, {cell("Q5", arm, d)}),
                (f"Q7 {arm}{d}", not_beating, arm, d, {cell("Q7", arm, d)} | mean_also)])
    return cases


CASES = make_cases()


@pytest.mark.parametrize("name, change, arm, d, expected", CASES, ids=[case[0] for case in CASES])
def test_changing_one_cell_changes_only_the_lines_that_read_it(reports, name, change, arm, d, expected) -> None:
    changed = copy.deepcopy(reports)
    change(changed, arm, d)
    assert flips(sp.score(reports, PREDICTIONS), sp.score(changed, PREDICTIONS)) == expected


@pytest.mark.parametrize("name, arm, kind, expected", [
    ("W early", "W", "early", {("Q6", Q6_WHAT["W", "early"])}), ("W gain", "W", "gain", {("Q6", Q6_WHAT["W", "gain"])}),
    ("S gain", "S", "gain", {("Q6", Q6_WHAT["S", "gain"])})])
def test_changing_one_q6_window_changes_only_its_own_line(reports, name, arm, kind, expected) -> None:
    changed = copy.deepcopy(reports)
    put_q6(changed, arm, kind, 0.5)
    assert flips(sp.score(reports, PREDICTIONS), sp.score(changed, PREDICTIONS)) == expected


@pytest.mark.parametrize("arm, d", [(arm, d) for arm in ARMS for d in WIDTHS])
def test_a_control_book_violation_in_any_run_breaks_the_q8_line_and_nothing_else(reports, arm, d) -> None:
    changed = copy.deepcopy(reports)
    changed[arm][d]["control_violations"] = 1
    assert flips(sp.score(reports, PREDICTIONS), sp.score(changed, PREDICTIONS)) == {("Q8", "runs that recalled any control entry (of 18)")}


# ------------------------------------------------------------------ lines that are reported but never scored

@pytest.mark.parametrize("matches, expected", [([True] * 3, "CHECK PASSED"), ([True, False, True], "CHECK FAILED"),
                                                ([True, None, True], "CHECK NOT RUN"), ([None] * 3, "CHECK NOT RUN"),
                                                ([False, None, True], "CHECK FAILED")])
def test_the_first_2000_epochs_check_needs_every_arm_s_width_to_match(reports, matches, expected) -> None:
    for d, value in zip(WIDTHS, matches):
        reports["S"][d]["first_2000_matches_earlier"] = value
    rows = sp.score(reports, PREDICTIONS)
    assert line(rows, "CHECK") == expected
    assert (row(rows, "CHECK")["what"], row(rows, "CHECK")["predicted"], row(rows, "CHECK")["actual"]) == (
        "first 2000 epochs of every arm S run equal the E6 run", "all equal", dict(zip(WIDTHS, matches)))
    assert not any(r["result"].startswith(("HIT", "MISS")) for r in rows if r["q"] == "CHECK")


def test_arm_w_has_no_first_2000_epochs_check(reports) -> None:
    reports["W"]["16"]["first_2000_matches_earlier"] = False                       # arm W departs from E6 at once; its value is not read
    assert line(sp.score(reports, PREDICTIONS), "CHECK") == "CHECK PASSED"


def test_a_miss_is_a_scored_line_that_is_not_a_hit(reports) -> None:
    put_perfect(reports, "W", "16", 0)                                             # misses its Q1 and its Q2 line
    reports["S"]["64"]["perfect_at_end"] = 0
    assert sp.render(sp.score(reports, PREDICTIONS)).splitlines()[-1] == f"41 hits of 44 scored lines {NOT_SCORED}"


def test_a_failed_check_does_not_change_the_hit_count(reports) -> None:
    reports["S"]["16"]["first_2000_matches_earlier"] = False
    assert sp.render(sp.score(reports, PREDICTIONS)).splitlines()[-1].startswith("44 hits of 44 scored lines")


def test_the_trend_of_every_width_in_both_arms_is_reported_with_its_gain(reports) -> None:
    rows = [r for r in sp.score(reports, PREDICTIONS) if r["q"] == "TREND"]
    assert [(r["what"], r["actual"], r["result"]) for r in rows] == [
        ("trend, arm W, d=16", "rising (gain +0.0200)", "REPORTED"), ("trend, arm W, d=32", "flat (gain +0.0000)", "REPORTED"),
        ("trend, arm W, d=64", "rising (gain +0.0070)", "REPORTED"), ("trend, arm S, d=16", "falling (gain -0.0100)", "REPORTED"),
        ("trend, arm S, d=32", "flat (gain +0.0010)", "REPORTED"), ("trend, arm S, d=64", "rising (gain +0.0316)", "REPORTED")]


# ------------------------------------------------------------------ the tripwires

def near_e6(reports: dict, arm: str) -> None:
    """The arm's late means equal E6's in all three widths."""
    for d in WIDTHS:
        put_late(reports, arm, d, PREDICTIONS["e6_control"]["mean_accuracy_3001_4000"][f"d{d}"])


def test_the_tripwires_of_the_realistic_outcome_are_the_two_that_fire(reports) -> None:
    assert fired(sp.score(reports, PREDICTIONS)) == {"w_d64_passes", "both_large"}


def test_a_tripwire_row_has_no_prediction_and_carries_the_text_of_the_file(reports) -> None:
    r = row(sp.score(reports, PREDICTIONS), "TRIPWIRE", "w_hurts")
    assert r == {"q": "TRIPWIRE", "what": "w_hurts", "predicted": "", "actual": False, "result": "TRIPWIRE NOT FIRED",
                 "note": PREDICTIONS["tripwires"]["w_hurts"]}


@pytest.mark.parametrize("passes, w_passes, w_fails", [(True, True, False), (False, False, True)])
def test_the_two_tripwires_about_arm_w_at_width_64_are_complements(reports, passes, w_passes, w_fails) -> None:
    put_passes(reports, "W", "64", passes)
    put_passes(reports, "W", "16", True)                                           # other widths do not count
    put_passes(reports, "S", "64", True)                                           # nor does the other arm
    names = fired(sp.score(reports, PREDICTIONS))
    assert ("w_d64_passes" in names) is w_passes and ("w_d64_fails_to_pass" in names) is w_fails


def test_a_failure_to_pass_elsewhere_does_not_fire_the_width_64_tripwire(reports) -> None:
    put_passes(reports, "W", "16", False)
    put_passes(reports, "S", "64", False)
    assert "w_d64_fails_to_pass" not in fired(sp.score(reports, PREDICTIONS))


@pytest.mark.parametrize("w_mean, s_mean, expected", [(0.979, 0.979, True), (0.98, 0.979, False), (0.979, 0.98, False), (0.5, 0.99, False)])
def test_neither_arm_lifts_d64_needs_both_means_below_0_98(reports, w_mean, s_mean, expected) -> None:
    put_late(reports, "W", "64", w_mean)
    put_late(reports, "S", "64", s_mean)
    assert ("neither_arm_lifts_d64" in fired(sp.score(reports, PREDICTIONS))) is expected


def test_neither_arm_lifts_d64_ignores_the_other_widths(reports) -> None:
    for d in ("16", "32"):
        put_late(reports, "W", d, 0.5)
        put_late(reports, "S", d, 0.5)
    assert "neither_arm_lifts_d64" not in fired(sp.score(reports, PREDICTIONS))


@pytest.mark.parametrize("w16, w32, w64, expected", [
    (0.95, 0.97, 0.975, True), (0.953, 0.97, 0.975, True), (0.947, 0.97, 0.975, True), (0.954, 0.97, 0.975, False),
    (0.95, 0.974, 0.975, False), (0.95, 0.97, 0.9781, False), (0.95, 0.97, 0.972, True), (0.95, 0.97, 0.9719, False)])
def test_w_null_s_large_needs_arm_w_within_0_003_of_e6_at_all_three_widths(reports, w16, w32, w64, expected) -> None:
    for d, value in zip(WIDTHS, (w16, w32, w64)):
        put_late(reports, "W", d, value)
    assert ("w_null_s_large" in fired(sp.score(reports, PREDICTIONS))) is expected


@pytest.mark.parametrize("s16, s32, expected", [(1, 1, True), (3, 3, True), (0, 3, False), (3, 0, False), (0, 0, False)])
def test_w_null_s_large_needs_arm_s_to_hold_at_both_widths_16_and_32(reports, s16, s32, expected) -> None:
    near_e6(reports, "W")
    put_holders(reports, "S", "16", s16)
    put_holders(reports, "S", "32", s32)
    assert ("w_null_s_large" in fired(sp.score(reports, PREDICTIONS))) is expected


def test_w_null_s_large_does_not_look_at_arm_s_at_width_64(reports) -> None:
    near_e6(reports, "W")
    put_holders(reports, "S", "64", 0)
    assert "w_null_s_large" in fired(sp.score(reports, PREDICTIONS))


def test_a_difference_of_exactly_0_003_is_within_even_when_the_float_difference_lands_just_over_it(reports) -> None:
    assert abs(0.903 - 0.9) > 0.003
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["e6_control"]["mean_accuracy_3001_4000"]["d16"] = 0.9
    near_e6(reports, "W")
    put_late(reports, "W", "16", 0.903)
    put_late(reports, "W", "32", 0.97)
    assert "w_null_s_large" in fired(sp.score(reports, predictions))
    put_late(reports, "W", "16", 0.9031)
    assert "w_null_s_large" not in fired(sp.score(reports, predictions))


@pytest.mark.parametrize("s16, s32, expected", [(0.95, 0.97, True), (0.953, 0.973, True), (0.947, 0.967, True), (0.954, 0.97, False),
                                                 (0.95, 0.974, False), (0.99, 0.97, False)])
def test_w_large_s_null_needs_arm_s_within_0_003_of_e6_at_widths_16_and_32(reports, s16, s32, expected) -> None:
    put_late(reports, "S", "16", s16)
    put_late(reports, "S", "32", s32)
    assert ("w_large_s_null" in fired(sp.score(reports, PREDICTIONS))) is expected


@pytest.mark.parametrize("w16, w32, expected", [(1, 1, True), (3, 2, True), (0, 3, False), (3, 0, False)])
def test_w_large_s_null_needs_arm_w_to_hold_at_both_widths_16_and_32(reports, w16, w32, expected) -> None:
    put_late(reports, "S", "16", 0.95)
    put_late(reports, "S", "32", 0.97)
    put_holders(reports, "W", "16", w16)
    put_holders(reports, "W", "32", w32)
    assert ("w_large_s_null" in fired(sp.score(reports, PREDICTIONS))) is expected


def test_w_large_s_null_does_not_look_at_width_64(reports) -> None:
    put_late(reports, "S", "16", 0.95)
    put_late(reports, "S", "32", 0.97)
    put_late(reports, "S", "64", 0.5)
    put_holders(reports, "W", "64", 0)
    assert "w_large_s_null" in fired(sp.score(reports, PREDICTIONS))


@pytest.mark.parametrize("arm, d, k, expected", [
    ("W", "16", 0, False), ("W", "32", 0, False), ("S", "16", 0, False), ("S", "32", 0, False),
    ("W", "16", 1, True), ("S", "32", 1, True), ("W", "64", 0, True), ("S", "64", 0, True)])
def test_both_large_needs_both_arms_to_hold_at_both_widths_16_and_32_and_ignores_width_64(reports, arm, d, k, expected) -> None:
    put_holders(reports, arm, d, k)
    assert ("both_large" in fired(sp.score(reports, PREDICTIONS))) is expected


def test_more_seeds_than_e6_means_more_than_the_e6_count_in_the_file(reports) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["e6_control"]["seeds_that_hold"]["d16"] = 3                        # W16 and S16 hold in 3 seeds, not more
    names = fired(sp.score(reports, predictions))
    assert "both_large" not in names and "w_large_s_null" not in names
    put_late(reports, "S", "16", 0.95)
    put_late(reports, "S", "32", 0.97)
    near_e6(reports, "W")
    assert "w_null_s_large" not in fired(sp.score(reports, predictions))
    predictions["e6_control"]["seeds_that_hold"]["d16"] = 2                        # now 3 is more
    assert "w_null_s_large" in fired(sp.score(reports, predictions))


@pytest.mark.parametrize("d, value, expected", [("16", 0.949, True), ("16", 0.95, False), ("32", 0.96, True), ("64", 0.97, True),
                                                 ("64", 0.975, False), ("64", 0.976, False)])
def test_w_hurts_when_arm_w_is_below_e6_at_any_width(reports, d, value, expected) -> None:
    put_late(reports, "W", d, value)
    assert ("w_hurts" in fired(sp.score(reports, PREDICTIONS))) is expected


def test_w_hurts_does_not_look_at_arm_s(reports) -> None:
    near_e6(reports, "S")
    put_late(reports, "S", "16", 0.5)
    assert "w_hurts" not in fired(sp.score(reports, PREDICTIONS))


@pytest.mark.parametrize("matches, tripwire", [([True] * 3, "TRIPWIRE NOT FIRED"), ([True, False, True], "TRIPWIRE FIRED"),
                                               ([False, True, True], "TRIPWIRE FIRED"), ([True, True, False], "TRIPWIRE FIRED"),
                                               ([True, None, True], "TRIPWIRE NOT RUN"), ([None] * 3, "TRIPWIRE NOT RUN"),
                                               ([False, None, None], "TRIPWIRE FIRED")])
def test_s_prefix_fails_follows_the_check_line(reports, matches, tripwire) -> None:
    for d, value in zip(WIDTHS, matches):
        reports["S"][d]["first_2000_matches_earlier"] = value
    assert line(sp.score(reports, PREDICTIONS), "TRIPWIRE", "s_prefix_fails") == tripwire


@pytest.mark.parametrize("k, expected", [(0, True), (1, True), (2, False), (3, False)])
def test_s_d64_does_not_hold_when_fewer_than_two_seeds_hold(reports, k, expected) -> None:
    put_holders(reports, "S", "64", k)
    assert ("s_d64_does_not_hold" in fired(sp.score(reports, PREDICTIONS))) is expected


def test_s_d64_does_not_hold_ignores_arm_w_and_the_other_widths(reports) -> None:
    put_holders(reports, "W", "64", 0)
    put_holders(reports, "S", "16", 0)
    assert "s_d64_does_not_hold" not in fired(sp.score(reports, PREDICTIONS))


@pytest.mark.parametrize("arm, d, k, expected", [
    ("W", "16", 1, True), ("W", "32", 1, True), ("S", "16", 1, True), ("S", "32", 1, True), ("W", "16", 2, False),
    ("S", "32", 2, False), ("W", "64", 0, False), ("S", "64", 0, False)])
def test_real_books_do_not_replicate_when_either_arm_holds_in_fewer_than_two_seeds_at_16_or_32(reports, arm, d, k, expected) -> None:
    put_holders(reports, arm, d, k)
    assert ("real_books_do_not_replicate" in fired(sp.score(reports, PREDICTIONS))) is expected


def test_the_text_of_a_tripwire_is_shown_only_when_it_fired(reports) -> None:
    text = sp.render(sp.score(reports, PREDICTIONS))
    assert "Reading: the reading of w_d64_passes." in text and "Reading: the reading of both_large." in text
    assert "the reading of w_hurts" not in text and "the reading of s_prefix_fails" not in text


def test_a_tripwire_the_scorer_has_no_rule_for_is_refused(reports) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["tripwires"]["something_new"] = "Fires when something new. Reading: nothing."
    with pytest.raises(ValueError, match="something_new"):
        sp.score(reports, predictions)


def test_a_tripwire_left_out_of_the_file_is_not_reported(reports) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    del predictions["tripwires"]["w_hurts"]
    rows = sp.score(reports, predictions)
    assert [r["what"] for r in rows if r["q"] == "TRIPWIRE"] == [t for t in TRIPWIRES if t != "w_hurts"]


# ------------------------------------------------------------------ what the scorer refuses to score

def test_a_wrong_schema_is_refused(reports) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["schema"] = "e6-predictions-v1"
    with pytest.raises(ValueError, match="schema"):
        sp.score(reports, predictions)


@pytest.mark.parametrize("arm, d", [("W", None), ("S", None), ("W", "16"), ("S", "32"), ("W", "64")])
def test_a_missing_arm_or_width_is_refused(reports, arm, d) -> None:
    if d is None:
        del reports[arm]
    else:
        del reports[arm][d]
    with pytest.raises(ValueError, match=f"arm {arm}" if d is None else f"arm {arm}, d={d}"):
        sp.score(reports, PREDICTIONS)


def test_a_width_with_fewer_than_three_seeds_is_refused(reports) -> None:
    reports["S"]["32"]["seeds"] = 2
    with pytest.raises(ValueError, match=r"arm S, d=32.*3 seeds"):
        sp.score(reports, PREDICTIONS)


def test_a_report_of_another_rung_is_refused(reports) -> None:
    reports["W"]["16"]["n"] = 20
    with pytest.raises(ValueError, match=r"arm W, d=16.*160"):
        sp.score(reports, PREDICTIONS)


def test_a_report_without_the_epochs_at_100_counts_is_refused(reports) -> None:
    del reports["S"]["64"]["epochs_at_100"]
    with pytest.raises(ValueError, match="--epochs-at-100"):
        sp.score(reports, PREDICTIONS)


@pytest.mark.parametrize("logged", [50, 99, 101, 200])
def test_holding_is_defined_on_100_late_logged_epochs_and_nothing_else_is_scored(reports, logged) -> None:
    reports["W"]["32"]["epochs_at_100"] = held(2, logged=logged)
    with pytest.raises(ValueError, match=rf"arm W, d=32: holding is defined on 100 logged epochs in 3001-4000, the report has {logged}"):
        sp.score(reports, PREDICTIONS)


def test_the_s_report_must_carry_the_2000_epoch_check(reports) -> None:
    del reports["S"]["16"]["first_2000_matches_earlier"]
    with pytest.raises(ValueError, match="--prefix-epochs 2000"):
        sp.score(reports, PREDICTIONS)


@pytest.mark.parametrize("key", ["q1_seeds_perfect_by_4000", "q4_seeds_that_hold_100_percent"])
def test_q1_and_q4_lines_are_never_optional_even_for_a_rung_predicted_not_to_pass(reports, key) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["q2_rung_passes_at_4000"]["S"]["d32"] = False
    del predictions[key]["S"]["d32"]
    with pytest.raises(ValueError, match=rf"{key} line for arm S, d=32"):
        sp.score(reports, predictions)


@pytest.mark.parametrize("budget, seeds, flag", [("4000", 1, True), ("4000", 2, False), ("2000", 0, True), ("1000", 3, False)])
def test_a_report_whose_passes_disagrees_with_its_perfect_seeds_is_refused(reports, budget, seeds, flag) -> None:
    reports["W"]["32"]["perfect_by"][budget] = seeds                               # as if analyze_extension.py had been run with another --min-seeds
    reports["W"]["32"]["passes"][budget] = flag
    with pytest.raises(ValueError, match=rf"arm W, d=32: .*passes at {budget} epochs.*without --min-seeds"):
        sp.score(reports, PREDICTIONS)


@pytest.mark.parametrize("seeds_listed", [2, 4])
def test_epochs_at_100_must_list_exactly_three_seeds(reports, seeds_listed) -> None:
    by_seed = reports["S"]["16"]["epochs_at_100"]["by_seed"]
    reports["S"]["16"]["epochs_at_100"]["by_seed"] = {str(s): by_seed["1"] for s in range(1, seeds_listed + 1)}
    with pytest.raises(ValueError, match=rf"arm S, d=16: epochs_at_100 lists {seeds_listed} seeds"):
        sp.score(reports, PREDICTIONS)


@pytest.mark.parametrize("key", ["seeds", "n", "perfect_by", "passes", "first_pass_budget", "perfect_at_end", "mean_accuracy", "gain", "trend",
                                 "control_violations"])
def test_a_report_that_lacks_a_field_is_refused_by_name(reports, key) -> None:
    del reports["W"]["64"][key]
    with pytest.raises(ValueError, match=rf"arm W, d=64: the report lacks {key}"):
        sp.score(reports, PREDICTIONS)


@pytest.mark.parametrize("key", ["mean_accuracy_3001_4000", "seeds_that_hold", "margin_for_beating_e6"])
def test_predictions_without_the_e6_control_values_are_refused_by_name(reports, key) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    del predictions["e6_control"][key]
    with pytest.raises(ValueError, match=rf"e6_control lacks {key}"):
        sp.score(reports, predictions)


@pytest.mark.parametrize("key", ["rungs", "e6_control", "tripwires"])
def test_predictions_without_a_whole_section_are_refused_by_name(reports, key) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    del predictions[key]
    with pytest.raises(ValueError, match=rf"the predictions have no {key}"):
        sp.score(reports, predictions)


def test_the_per_seed_counts_are_listed_in_seed_order_by_number(reports) -> None:
    by_seed = reports["W"]["32"]["epochs_at_100"]["by_seed"]                        # seeds 1, 2, 3 hold 100, 100, 3
    reports["W"]["32"]["epochs_at_100"]["by_seed"] = {"10": by_seed["3"], "2": by_seed["2"], "3": by_seed["1"]}
    assert row(sp.score(reports, PREDICTIONS), "Q4", "arm W, d=32")["note"].endswith("per seed: 100, 100, 3")


def test_arm_w_needs_no_first_2000_epochs_key(reports) -> None:
    for d in WIDTHS:
        del reports["W"][d]["first_2000_matches_earlier"]
        del reports["W"][d]["first_2000_match_by_seed"]
    assert len(scored(sp.score(reports, PREDICTIONS))) == 44


@pytest.mark.parametrize("key", ["q1_seeds_perfect_by_4000", "q2_rung_passes_at_4000", "q4_seeds_that_hold_100_percent",
                                 "q7_late_mean_beats_e6", "q3_first_passing_budget", "q5_seeds_perfect_at_epoch_4000"])
def test_a_missing_line_is_an_error_unless_the_rung_is_predicted_not_to_pass(reports, key) -> None:
    predictions = copy.deepcopy(PREDICTIONS)
    del predictions[key]["S"]["d32"]
    with pytest.raises(ValueError, match=rf"{key} line for arm S, d=32"):
        sp.score(reports, predictions)


def test_a_rendered_line_has_fixed_columns_and_a_note_in_brackets(reports) -> None:
    lines = sp.render(sp.score(reports, PREDICTIONS)).splitlines()
    assert lines[0] == ("Q1".ljust(9) + " " + "seeds perfect by 4000 epochs, arm W, d=16".ljust(64) + " predicted " + "3 [3, 3]".ljust(22)
                        + " actual " + "3".ljust(30) + " HIT (exact)")
    q7 = next(text for text in lines if text.startswith("Q7") and "arm W, d=16" in text)
    assert q7.endswith(" HIT   [0.9900 against E6 0.9500]")


def test_the_scorer_does_not_change_what_it_is_given(reports) -> None:
    before, before_predictions = copy.deepcopy(reports), copy.deepcopy(PREDICTIONS)
    sp.score(reports, PREDICTIONS)
    assert reports == before and PREDICTIONS == before_predictions


# ------------------------------------------------------------------ what analyze_extension.py writes

def curve(n: int, perfect_from: int) -> list[dict]:
    """A run logged every 10 epochs: 5 entries short of n until `perfect_from`, all n from then on."""
    return [{"epoch": epoch, "A": n if epoch >= perfect_from else n - 5, "B": 0} for epoch in range(10, 4001, 10)]


def analysis(perfect_from: int, earlier: dict | None = None) -> dict:
    """The report of one arm, from the analysis code itself (three widths of three seeds) and through a JSON round trip."""
    from phonebook import analyze_extension as ae
    runs = {int(d): [{"seed": seed, "n": RUNGS[d], "points": curve(RUNGS[d], perfect_from)} for seed in (1, 2, 3)] for d in WIDTHS}
    report = ae.summarize_extension(runs, (1000, 2000, 3000, 4000), earlier=earlier, prefix=2000, held=True)
    return json.loads(json.dumps(report))


def e6_runs(a_short: int = 5) -> dict:
    """E6 as the analysis wants it: the runs of every width and seed, 5 entries short of n for all 4000 epochs."""
    return {(int(d), seed, RUNGS[d]): [{"epoch": e, "A": RUNGS[d] - a_short, "B": 0} for e in range(10, 4001, 10)]
            for d in WIDTHS for seed in (1, 2, 3)}


def test_the_scorer_reads_what_the_analysis_writes() -> None:
    rows = sp.score({"W": analysis(1500), "S": analysis(2010, earlier=e6_runs())}, PREDICTIONS)
    actual = lambda q, arm: [r["actual"] for r in rows if r["q"] == q and f"arm {arm}," in r["what"]]
    assert actual("Q1", "W") == actual("Q1", "S") == actual("Q4", "W") == actual("Q4", "S") == actual("Q5", "W") == actual("Q5", "S") == [3, 3, 3]
    assert actual("Q2", "W") == actual("Q2", "S") == [True] * 3
    assert actual("Q3", "W") == [2000] * 3 and actual("Q3", "S") == [3000] * 3          # arm S is perfect from epoch 2010
    assert actual("Q7", "W") == actual("Q7", "S") == [True] * 3 and row(rows, "Q8")["actual"] == 0
    assert line(rows, "CHECK") == "CHECK PASSED"
    assert len(scored(rows)) == 44


@pytest.mark.parametrize("earlier, check", [(None, "CHECK NOT RUN"), (e6_runs(6), "CHECK FAILED"), (e6_runs(5), "CHECK PASSED")])
def test_the_check_follows_the_comparison_with_the_e6_runs_that_the_analysis_made(earlier, check) -> None:
    rows = sp.score({"W": analysis(1500), "S": analysis(2010, earlier=earlier)}, PREDICTIONS)
    assert line(rows, "CHECK") == check
    assert (fired(rows) >= {"s_prefix_fails"}) is (check == "CHECK FAILED")


# ------------------------------------------------------------------ the adopted file and the command line

# ---- PINNED NUMBERS (written by pin_predictions.py from the adopted file; regenerate, do not edit by hand) ----
LOCKED_SHA256 = "8876264b8ca03e7f3c589af2162feebe5a1a29a6e0e076805362e52b954ea7e9"
PINNED = {'schema': 'e7-predictions-v1',
 'rungs': {'d16': 160, 'd32': 320, 'd64': 640},
 'e6_control': {'mean_accuracy_3001_4000': {'d16': 0.9525625, 'd32': 0.9733020833333333, 'd64': 0.9754427083333334},
                'seeds_that_hold': {'d16': 0, 'd32': 0, 'd64': 0},
                'margin_for_beating_e6': 0.005},
 'q1_seeds_perfect_by_4000': {'W': {'d16': {'point': 3, 'low': 3, 'high': 3},
                                    'd32': {'point': 3, 'low': 3, 'high': 3},
                                    'd64': {'point': 3, 'low': 3, 'high': 3}},
                              'S': {'d16': {'point': 3, 'low': 3, 'high': 3},
                                    'd32': {'point': 3, 'low': 3, 'high': 3},
                                    'd64': {'point': 3, 'low': 3, 'high': 3}}},
 'q2_rung_passes_at_4000': {'confidence_of_this_answer': {'W': {'d16': 0.99, 'd32': 0.99, 'd64': 0.99},
                                                          'S': {'d16': 0.99, 'd32': 0.99, 'd64': 0.99}},
                            'W': {'d16': True, 'd32': True, 'd64': True},
                            'S': {'d16': True, 'd32': True, 'd64': True}},
 'q3_first_passing_budget': {'W': {'d16': {'point': 2000, 'low': 2000, 'high': 3000},
                                   'd32': {'point': 2000, 'low': 2000, 'high': 3000},
                                   'd64': {'point': 2000, 'low': 2000, 'high': 2000}},
                             'S': {'d16': {'point': 3000, 'low': 3000, 'high': 3000},
                                   'd32': {'point': 3000, 'low': 3000, 'high': 3000},
                                   'd64': {'point': 3000, 'low': 3000, 'high': 3000}}},
 'q4_seeds_that_hold_100_percent': {'W': {'d16': {'point': 3, 'low': 3, 'high': 3},
                                          'd32': {'point': 3, 'low': 3, 'high': 3},
                                          'd64': {'point': 3, 'low': 3, 'high': 3}},
                                    'S': {'d16': {'point': 3, 'low': 3, 'high': 3},
                                          'd32': {'point': 3, 'low': 3, 'high': 3},
                                          'd64': {'point': 3, 'low': 3, 'high': 3}}},
 'q5_seeds_perfect_at_epoch_4000': {'W': {'d16': {'point': 3, 'low': 3, 'high': 3},
                                          'd32': {'point': 3, 'low': 3, 'high': 3},
                                          'd64': {'point': 3, 'low': 3, 'high': 3}},
                                    'S': {'d16': {'point': 3, 'low': 3, 'high': 3},
                                          'd32': {'point': 3, 'low': 3, 'high': 3},
                                          'd64': {'point': 3, 'low': 3, 'high': 3}}},
 'q6_d64_windows': {'W': {'mean_accuracy_1001_2000': {'point': 0.991, 'low': 0.98, 'high': 0.998},
                          'mean_accuracy_3001_4000': {'point': 0.9988, 'low': 0.9934, 'high': 1.0},
                          'gain_vs_1001_2000': {'point': 0.0068, 'low': -0.0019, 'high': 0.0164}},
                    'S': {'mean_accuracy_3001_4000': {'point': 1.0, 'low': 0.995, 'high': 1.0},
                          'gain_vs_1001_2000': {'point': 0.0316, 'low': 0.0266, 'high': 0.0316}}},
 'q7_late_mean_beats_e6': {'W': {'d16': {'answer': True, 'confidence': 0.99},
                                 'd32': {'answer': True, 'confidence': 0.99},
                                 'd64': {'answer': True, 'confidence': 0.97}},
                           'S': {'d16': {'answer': True, 'confidence': 0.99},
                                 'd32': {'answer': True, 'confidence': 0.98},
                                 'd64': {'answer': True, 'confidence': 0.98}}},
 'q8_control_book': {'max_control_violations': 0},
 'q9_contrasts': {'d64_higher_3001_4000_mean': {'answer': 'S', 'confidence': 0.66},
                  'd16_more_holding_seeds': {'answer': 'tie', 'confidence': 0.9}},
 'tripwires': ['w_d64_passes',
               'w_d64_fails_to_pass',
               'neither_arm_lifts_d64',
               'w_null_s_large',
               'w_large_s_null',
               'both_large',
               'w_hurts',
               's_prefix_fails',
               's_d64_does_not_hold',
               'real_books_do_not_replicate']}
# ---- END PINNED NUMBERS ----

# Written by hand. If the adopted wording of a tripwire's first sentence differs, the scorer may no longer implement it:
# change the scorer and its tests first, then this table.
RULE_SENTENCES = {
    "w_d64_passes": "Fires when arm W passes at width 64 (at least 2 seeds perfect by 4000)",
    "w_d64_fails_to_pass": "Fires when arm W does not pass at width 64 (fewer than 2 seeds perfect by 4000)",
    "neither_arm_lifts_d64": "Fires when both arms have a width-64 3001-4000 mean accuracy below 0.980",
    "w_null_s_large": ("Fires when arm W's 3001-4000 mean is within 0.003 of E6's at all three widths and arm S holds in more seeds "
                       "than E6 (E6 held in none, so at least 1) at both widths 16 and 32"),
    "w_large_s_null": ("Fires when arm W holds in more seeds than E6 (E6 held in none, so at least 1) at both widths 16 and 32 and "
                       "arm S's 3001-4000 mean is within 0.003 of E6's at both"),
    "both_large": "Fires when both arms hold in more seeds than E6 (E6 held in none, so at least 1) at both widths 16 and 32",
    "w_hurts": "Fires when arm W's 3001-4000 mean accuracy is below E6's at any width",
    "s_prefix_fails": "Fires when the first 2000 epochs of an arm S run differ from the E6 run at any width (the CHECK line of the scorer)",
    "s_d64_does_not_hold": "Fires when arm S holds in fewer than 2 seeds at width 64",
    "real_books_do_not_replicate": "Fires when either arm holds in fewer than 2 seeds at width 16 or at width 32",
}


def strip(value):
    """Drop the free-text reasons, keeping every number and answer."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k != "why"}
    return value


def test_the_locked_predictions_file_is_unchanged() -> None:
    assert hashlib.sha256(LOCKED.read_bytes()).hexdigest() == LOCKED_SHA256


def test_the_locked_predictions_hold_exactly_the_numbers_pinned_here() -> None:
    committed = load_predictions(LOCKED)
    assert {key for key in committed if key in PINNED} == set(PINNED) and "tripwires" in PINNED
    for key, expected in PINNED.items():
        if key == "tripwires":
            assert list(committed[key]) == expected                                # the wording is free; the ids and their order are fixed
        else:
            assert strip(committed[key]) == expected, key
    assert PINNED["schema"] == "e7-predictions-v1"


def test_the_rule_sentence_of_every_locked_tripwire_is_the_one_the_scorer_implements() -> None:
    committed = load_predictions(LOCKED)
    assert {name: text.split(". ")[0] for name, text in committed["tripwires"].items()} == RULE_SENTENCES
    assert list(RULE_SENTENCES) == list(TRIPWIRES)


def test_the_locked_predictions_can_be_scored(reports) -> None:
    rows = sp.score(reports, load_predictions(LOCKED))
    assert len(scored(rows)) == 44
    assert [r["what"] for r in rows if r["q"] == "TRIPWIRE"] == list(TRIPWIRES)


@pytest.fixture
def files(reports, tmp_path: Path) -> dict:
    """The two analyses and the predictions as the command line wants them: three JSON files."""
    paths = {}
    for name, content in {"--w": reports["W"], "--s": reports["S"], "--predictions": PREDICTIONS}.items():
        paths[name] = tmp_path / f"{name.strip('-')}.json"
        paths[name].write_text(json.dumps(content), encoding="utf-8")
    return paths


def run_scorer(arguments: dict) -> subprocess.CompletedProcess:
    command = [sys.executable, str(ROOT / "phonebook" / "score_e7_predictions.py"), *(str(x) for pair in arguments.items() for x in pair)]
    return subprocess.run(command, capture_output=True, text=True, cwd=ROOT)


def test_the_command_line_prints_the_table_and_writes_json(files, tmp_path: Path) -> None:
    out = tmp_path / "scores.json"
    done = run_scorer({**files, "--json": out})
    assert done.returncode == 0, done.stderr
    assert "44 hits of 44 scored lines" in done.stdout
    text = out.read_text(encoding="utf-8")
    assert len(json.loads(text)) == 61 and text.startswith('[\n {\n  "q": "Q1"') and text.endswith("}\n]\n")
    plain = run_scorer(files)
    assert plain.returncode == 0 and plain.stdout == done.stdout                   # --json only adds a file


@pytest.mark.parametrize("missing", ["--w", "--s", "--predictions"])
def test_the_command_line_needs_both_analyses_and_the_predictions(files, missing) -> None:
    done = run_scorer({name: path for name, path in files.items() if name != missing})
    assert done.returncode == 2 and missing in done.stderr and "required" in done.stderr
