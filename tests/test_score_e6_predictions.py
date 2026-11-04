"""The E6 scorer applies rules fixed before any result existed; each rule is pinned here by a hit and a miss."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from phonebook import score_e6_predictions as sp
from phonebook.predictions import load_predictions

ROOT = Path(__file__).resolve().parents[1]


def rng(point, low, high):
    return {"point": point, "low": low, "high": high}


PREDICTIONS = {
    "q1_seeds_perfect_by_4000": {"d8": rng(3, 2, 3), "d16": rng(2, 1, 3), "d32": rng(2, 1, 3), "d64": rng(0, 0, 1)},
    "q2_rung_passes_at_4000": {"d8": True, "d16": True, "d32": True, "d64": False},
    "q3_first_passing_budget": {"d8": rng(2000, 2000, 3000), "d16": rng(3000, 2000, 4000), "d32": rng(3000, 2000, 4000)},
    "q4_widths_passing_at_4000": rng(3, 2, 3),
    "q5_d64_last_windows": {"mean_accuracy_3001_4000": rng(0.975, 0.965, 0.985), "gain_vs_1001_2000": rng(0.007, 0.003, 0.012)},
    "q6_still_perfect_at_epoch_4000": {"d16_plus_d32_runs": rng(0, 0, 1), "d8_at_least": 2},
    "q7_control_book": {"max_control_violations": 0},
    "q8_results_that_would_change_my_mind": {
        "tripwire_d64_passes": "d=64 passes at 4000 epochs",
        "tripwire_d16_and_d32_both_fail": "d=16 and d=32 both fail at 4000 epochs"},
}


def width(n, *, by_4000, first_pass, at_end, mean=0.97, gain=0.0, trend="flat", violations=0, match=True):
    passes = by_4000 >= 2
    return {"n": n, "seeds": 3, "perfect_by": {"1000": 0, "2000": 0, "3000": 0, "4000": by_4000},
            "passes": {"1000": False, "2000": False, "3000": False, "4000": passes},
            "first_pass_budget": first_pass, "reached_perfect": by_4000, "perfect_at_end": at_end,
            "mean_accuracy": {"1001_2000": mean - gain, "3001_4000": mean}, "gain": gain, "trend": trend,
            "control_violations": violations, "first_1000_matches_earlier": match}


@pytest.fixture
def report() -> dict:
    """Every scored line lands exactly on its prediction."""
    return {"8": width(20, by_4000=3, first_pass=2000, at_end=3),
            "16": width(160, by_4000=2, first_pass=3000, at_end=0),
            "32": width(320, by_4000=2, first_pass=3000, at_end=0),
            "64": width(640, by_4000=0, first_pass=None, at_end=0, mean=0.975, gain=0.007, trend="rising")}


def line(rows: list[dict], q: str, what_contains: str = "") -> str:
    return next(r["result"] for r in rows if r["q"] == q and what_contains in r["what"])


def test_a_report_that_lands_on_every_prediction_is_all_hits(report) -> None:
    rows = sp.score(report, PREDICTIONS)
    scored = [r for r in rows if r["result"].startswith(("HIT", "MISS"))]
    assert len(scored) == 4 + 4 + 3 + 1 + 2 + 2 + 1 and all(r["result"].startswith("HIT") for r in scored)
    assert sum(r["result"] == "HIT (exact)" for r in rows) == 4 + 3 + 1 + 2 + 1 + 1       # Q1, Q3, Q4, Q5, Q6 wide, Q7
    assert len(rows) == 17 + 1 + 4 + 2
    assert sp.render(rows).splitlines()[-1] == ("17 hits of 17 scored lines "
                                                "(the check, the trends and the tripwires are reported, not scored)")


@pytest.mark.parametrize("seeds, expected", [(2, "HIT"), (3, "HIT (exact)"), (1, "MISS"), (0, "MISS")])
def test_q1_the_count_of_perfect_seeds_is_a_closed_range(report, seeds, expected) -> None:
    changed = copy.deepcopy(report)
    changed["8"]["perfect_by"]["4000"] = seeds
    assert line(sp.score(changed, PREDICTIONS), "Q1", "d=8") == expected


def test_q1_reads_the_4000_epoch_column_only(report) -> None:
    changed = copy.deepcopy(report)
    changed["64"]["perfect_by"]["2000"] = 3                                # perfect early does not matter for the 4000 line
    assert line(sp.score(changed, PREDICTIONS), "Q1", "d=64") == "HIT (exact)"
    changed["64"]["perfect_by"]["4000"] = 2
    assert line(sp.score(changed, PREDICTIONS), "Q1", "d=64") == "MISS"


@pytest.mark.parametrize("d, passes, expected", [("8", True, "HIT"), ("8", False, "MISS"), ("64", False, "HIT"), ("64", True, "MISS")])
def test_q2_the_rung_either_passes_or_not_as_predicted(report, d, passes, expected) -> None:
    changed = copy.deepcopy(report)
    changed[d]["passes"]["4000"] = passes
    assert line(sp.score(changed, PREDICTIONS), "Q2", f"d={d}") == expected


@pytest.mark.parametrize("budget, expected", [(2000, "HIT (exact)"), (3000, "HIT"), (1000, "MISS"), (4000, "MISS"), (None, "MISS")])
def test_q3_the_first_passing_budget_is_a_closed_range_and_never_passing_is_a_miss(report, budget, expected) -> None:
    changed = copy.deepcopy(report)
    changed["8"]["first_pass_budget"] = budget
    assert line(sp.score(changed, PREDICTIONS), "Q3", "d=8") == expected


def test_q3_has_no_line_for_d64_and_its_exact_grade_is_the_point(report) -> None:
    rows = sp.score(report, PREDICTIONS)
    assert [r["what"] for r in rows if r["q"] == "Q3"] == ["first passing budget, d=8", "first passing budget, d=16",
                                                           "first passing budget, d=32"]
    changed = copy.deepcopy(report)
    changed["16"]["first_pass_budget"] = 2000
    assert line(sp.score(changed, PREDICTIONS), "Q3", "d=16") == "HIT"


@pytest.mark.parametrize("passing, expected", [(3, "HIT (exact)"), (2, "HIT"), (1, "MISS"), (4, "MISS")])
def test_q4_counts_the_widths_that_pass_at_4000(report, passing, expected) -> None:
    changed = copy.deepcopy(report)
    for d, flag in zip(("8", "16", "32", "64"), [True] * passing + [False] * (4 - passing)):
        changed[d]["passes"]["4000"] = flag
    assert line(sp.score(changed, PREDICTIONS), "Q4") == expected


@pytest.mark.parametrize("mean, expected", [(0.975, "HIT (exact)"), (0.965, "HIT"), (0.985, "HIT"), (0.9649, "MISS"), (0.9851, "MISS")])
def test_q5_the_last_window_mean_is_a_closed_range(report, mean, expected) -> None:
    changed = copy.deepcopy(report)
    changed["64"]["mean_accuracy"]["3001_4000"] = mean
    assert line(sp.score(changed, PREDICTIONS), "Q5", "mean accuracy") == expected


@pytest.mark.parametrize("gain, expected", [(0.007, "HIT (exact)"), (0.003, "HIT"), (0.012, "HIT"), (0.0029, "MISS"), (0.0121, "MISS"), (-0.004, "MISS")])
def test_q5_the_gain_is_a_closed_range(report, gain, expected) -> None:
    changed = copy.deepcopy(report)
    changed["64"]["gain"] = gain
    assert line(sp.score(changed, PREDICTIONS), "Q5", "gain") == expected


@pytest.mark.parametrize("d16, d32, expected", [(0, 0, "HIT (exact)"), (1, 0, "HIT"), (0, 1, "HIT"), (1, 1, "MISS"), (2, 0, "MISS")])
def test_q6_counts_the_wide_runs_still_perfect_at_the_end_across_both_widths(report, d16, d32, expected) -> None:
    changed = copy.deepcopy(report)
    changed["16"]["perfect_at_end"], changed["32"]["perfect_at_end"] = d16, d32
    assert line(sp.score(changed, PREDICTIONS), "Q6", "d=16 and d=32") == expected


@pytest.mark.parametrize("at_end, expected", [(3, "HIT"), (2, "HIT"), (1, "MISS"), (0, "MISS")])
def test_q6_needs_at_least_two_of_the_three_d8_runs_to_end_perfect(report, at_end, expected) -> None:
    changed = copy.deepcopy(report)
    changed["8"]["perfect_at_end"] = at_end
    assert line(sp.score(changed, PREDICTIONS), "Q6", "d=8") == expected


def test_q6_does_not_count_d64_or_d8_among_the_wide_runs(report) -> None:
    changed = copy.deepcopy(report)
    changed["64"]["perfect_at_end"] = 3
    changed["8"]["perfect_at_end"] = 3
    assert line(sp.score(changed, PREDICTIONS), "Q6", "d=16 and d=32") == "HIT (exact)"


@pytest.mark.parametrize("violations, allowed, expected", [(0, 0, "HIT (exact)"), (1, 0, "MISS"), (1, 1, "HIT (exact)"), (0, 1, "HIT")])
def test_q7_the_control_book_violations_are_summed_over_all_widths(report, violations, allowed, expected) -> None:
    changed = copy.deepcopy(report)
    changed["32"]["control_violations"] = violations
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["q7_control_book"]["max_control_violations"] = allowed
    assert line(sp.score(changed, predictions), "Q7") == expected


# ------------------------------------------------------------------ lines that are reported but never scored

@pytest.mark.parametrize("matches, expected", [([True] * 4, "CHECK PASSED"), ([True, True, False, True], "CHECK FAILED"),
                                                ([True, None, True, True], "CHECK NOT RUN"), ([None] * 4, "CHECK NOT RUN")])
def test_the_first_1000_epochs_check_needs_every_width_to_match(report, matches, expected) -> None:
    changed = copy.deepcopy(report)
    for d, value in zip(("8", "16", "32", "64"), matches):
        changed[d]["first_1000_matches_earlier"] = value
    rows = sp.score(changed, PREDICTIONS)
    assert line(rows, "CHECK") == expected
    assert not any(r["result"].startswith(("HIT", "MISS")) for r in rows if r["q"] == "CHECK")


def test_a_miss_is_a_scored_line_that_is_not_a_hit(report) -> None:
    changed = copy.deepcopy(report)
    changed["8"]["perfect_by"]["4000"] = 0
    changed["16"]["perfect_by"]["4000"] = 0
    last = sp.render(sp.score(changed, PREDICTIONS)).splitlines()[-1]
    assert last.startswith("15 hits of 17 scored lines")


def test_a_failed_check_does_not_change_the_hit_count(report) -> None:
    changed = copy.deepcopy(report)
    changed["8"]["first_1000_matches_earlier"] = False
    assert sp.render(sp.score(changed, PREDICTIONS)).splitlines()[-1].startswith("17 hits of 17 scored lines")


def test_the_trend_of_every_width_is_reported_with_its_gain(report) -> None:
    rows = [r for r in sp.score(report, PREDICTIONS) if r["q"] == "TREND"]
    assert [(r["what"], r["actual"], r["result"]) for r in rows] == [
        ("trend, d=8", "flat (gain +0.0000)", "REPORTED"), ("trend, d=16", "flat (gain +0.0000)", "REPORTED"),
        ("trend, d=32", "flat (gain +0.0000)", "REPORTED"), ("trend, d=64", "rising (gain +0.0070)", "REPORTED")]


def test_a_d64_pass_fires_the_first_tripwire(report) -> None:
    assert line(sp.score(report, PREDICTIONS), "TRIPWIRE", "d=64 passes") == "TRIPWIRE NOT FIRED"
    changed = copy.deepcopy(report)
    changed["64"]["passes"]["4000"] = True
    rows = sp.score(changed, PREDICTIONS)
    assert line(rows, "TRIPWIRE", "d=64 passes") == "TRIPWIRE FIRED"
    assert line(rows, "TRIPWIRE", "both fail") == "TRIPWIRE NOT FIRED"


@pytest.mark.parametrize("d16, d32, expected", [(True, True, "TRIPWIRE NOT FIRED"), (False, True, "TRIPWIRE NOT FIRED"),
                                                 (True, False, "TRIPWIRE NOT FIRED"), (False, False, "TRIPWIRE FIRED")])
def test_the_second_tripwire_needs_both_d16_and_d32_to_fail(report, d16, d32, expected) -> None:
    changed = copy.deepcopy(report)
    changed["16"]["passes"]["4000"], changed["32"]["passes"]["4000"] = d16, d32
    assert line(sp.score(changed, PREDICTIONS), "TRIPWIRE", "both fail") == expected


# ------------------------------------------------------------------ the committed file and the command line

def strip(value):
    """Drop the free-text fields, keeping only what the scorer reads."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in ("why", "confidence_of_this_answer")}
    return value


def test_the_committed_predictions_hold_exactly_the_numbers_pinned_in_these_tests() -> None:
    committed = load_predictions(ROOT / "phonebook" / "preregistration" / "e6_predictions.json")
    for key, expected in PREDICTIONS.items():
        got = strip(committed[key])
        if key == "q8_results_that_would_change_my_mind":
            assert set(got) == set(expected)                                    # the wording is free; the two tripwires are fixed
        else:
            assert got == expected, key
    assert committed["rungs"] == {"d8": 20, "d16": 160, "d32": 320, "d64": 640}
    assert committed["schema"] == "e6-predictions-v1"


def test_the_committed_predictions_score_a_report_that_lands_on_them(report) -> None:
    committed = load_predictions(ROOT / "phonebook" / "preregistration" / "e6_predictions.json")
    rows = sp.score(report, committed)
    assert sp.render(rows).splitlines()[-1].startswith("17 hits of 17 scored lines")


def test_the_command_line_prints_the_table_and_writes_json(report, tmp_path: Path) -> None:
    paths = {}
    for name, content in {"extension": report, "predictions": PREDICTIONS}.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(content), encoding="utf-8")
    out = tmp_path / "scores.json"
    done = subprocess.run([sys.executable, str(ROOT / "phonebook" / "score_e6_predictions.py"), "--extension", str(paths["extension"]),
                           "--predictions", str(paths["predictions"]), "--json", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    assert "17 hits of 17 scored lines" in done.stdout
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 24
