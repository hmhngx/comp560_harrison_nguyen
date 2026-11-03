"""The E4 scorer applies rules fixed before any result existed; each rule is pinned here by a hit and a miss."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from phonebook import score_e4_predictions as sp

ROOT = Path(__file__).resolve().parents[1]


def rng(point, low, high):
    return {"point": point, "low": low, "high": high}


PREDICTIONS = {
    "q1_capacity_at_1000": {"d8": rng(14, 10, 20), "d16": rng(80, 57, 113), "d32": rng(160, 113, 226), "d64": rng(320, 226, 453)},
    "q2_capacity_at_300": {"d8": rng(0, 0, 0), "d16": rng(20, 10, 40), "d32": rng(160, 80, 226), "d64": rng(320, 160, 320)},
    "q3_capacity_at_100": {"d8": rng(0, 0, 0), "d16": rng(0, 0, 0), "d32": rng(0, 0, 0), "d64": rng(20, 0, 40)},
    "q4_boundary_status_at_1000": {"d8": "moving", "d16": "moving", "d32": "stable", "d64": "stable"},
    "q5_capacity_grows_with_width": {"answer": "strict"},
    "q6_bits_per_param_at_1000": {"d8": rng(0.15, 0.09, 0.25), "d16": rng(0.28, 0.17, 0.4), "d32": rng(0.17, 0.1, 0.3), "d64": rng(0.09, 0.07, 0.2)},
    "q7_no_ladder_ceiling": {"answer": True},
    "q8_control_book": {"max_control_violations": 0},
    "q10_result_that_would_change_my_mind": {"tripwire": "d=64 holds 640 or more entries, or capacity grows 4x or more per doubling of width",
                                             "d64_capacity_at_least": 640, "width_doubling_ratio_at_least": 4},
}


def model(c100, c300, c500, c1000, *, boundary="stable", bits=0.2, lower=False, violations=0):
    return {"capacity_by_budget": {"100": c100, "300": c300, "500": c500, "1000": c1000}, "boundary": boundary,
            "bits_per_param_at_largest_budget": bits, "capacity_is_lower_bound": lower, "control_violations": violations}


@pytest.fixture
def report() -> dict:
    """Every line lands exactly on its prediction."""
    return {"8": model(None, None, None, 14, boundary="moving", bits=0.15),
            "16": model(None, 20, 40, 80, boundary="moving", bits=0.28),
            "32": model(None, 160, 160, 160, boundary="stable", bits=0.17),
            "64": model(20, 320, 320, 320, boundary="stable", bits=0.09)}


def line(rows: list[dict], q: str, what_contains: str = "") -> str:
    return next(r["result"] for r in rows if r["q"] == q and what_contains in r["what"])


def test_a_report_that_lands_on_every_prediction_is_all_hits(report) -> None:
    rows = sp.score(report, PREDICTIONS)
    ok = ("HIT", "HIT (exact)", "STORY SURVIVES")
    assert all(r["result"] in ok for r in rows), [r for r in rows if r["result"] not in ok]
    assert sum(r["result"] == "HIT (exact)" for r in rows) == 13       # Q1-Q3 (12 lines) and Q8; Q4-Q7 have no "exact" grade
    assert len(rows) == 12 + 4 + 1 + 4 + 1 + 1 + 1
    assert sp.render(rows).splitlines()[-1] == "23 hits of 23 scored lines (Q10 is a tripwire report, not scored)"


@pytest.mark.parametrize("actual, expected", [(10, "HIT"), (20, "HIT"), (14, "HIT (exact)"), (None, "MISS"), (28, "MISS"), (8, "MISS")])
def test_a_capacity_range_is_inclusive_and_a_null_capacity_is_zero_entries(report, actual, expected) -> None:
    changed = copy.deepcopy(report)
    changed["8"]["capacity_by_budget"]["1000"] = actual
    assert line(sp.score(changed, PREDICTIONS), "Q1", "d=8") == expected


def test_no_rung_passing_is_zero_entries_and_hits_a_prediction_of_zero(report) -> None:
    assert line(sp.score(report, PREDICTIONS), "Q2", "d=8") == "HIT (exact)"
    changed = copy.deepcopy(report)
    changed["8"]["capacity_by_budget"]["300"] = 10                      # something passed where nothing was predicted
    assert line(sp.score(changed, PREDICTIONS), "Q2", "d=8") == "MISS"


def test_each_budget_is_read_from_its_own_column(report) -> None:
    changed = copy.deepcopy(report)
    changed["64"]["capacity_by_budget"]["100"] = 640
    rows = sp.score(changed, PREDICTIONS)
    assert line(rows, "Q3", "d=64") == "MISS" and line(rows, "Q1", "d=64") == "HIT (exact)"


@pytest.mark.parametrize("status, expected", [("stable", "HIT"), ("moving", "MISS"), ("ceiling", "MISS"), ("none", "MISS")])
def test_the_boundary_status_must_match_exactly(report, status, expected) -> None:
    changed = copy.deepcopy(report)
    changed["32"]["boundary"] = status
    assert line(sp.score(changed, PREDICTIONS), "Q4", "d=32") == expected


@pytest.mark.parametrize("caps, expected", [
    ((14, 80, 160, 320), "HIT"),
    ((14, 80, 80, 320), "MISS"),               # a tie is not strictly increasing
    ((None, 80, 160, 320), "HIT"),             # null counts as zero, which is still below 80
    ((14, 80, 160, 160), "MISS"),
    ((80, 14, 160, 320), "MISS"),
])
def test_q5_needs_strictly_more_entries_at_every_wider_model(report, caps, expected) -> None:
    changed = copy.deepcopy(report)
    for d, value in zip(("8", "16", "32", "64"), caps):
        changed[d]["capacity_by_budget"]["1000"] = value
    assert line(sp.score(changed, PREDICTIONS), "Q5") == expected


@pytest.mark.parametrize("bits, expected", [(0.09, "HIT"), (0.07, "HIT"), (0.2, "HIT"), (0.0699, "MISS"), (0.2001, "MISS"), (None, "MISS")])
def test_bits_per_parameter_is_a_closed_range_and_a_missing_value_is_a_miss(report, bits, expected) -> None:
    changed = copy.deepcopy(report)
    changed["64"]["bits_per_param_at_largest_budget"] = bits
    assert line(sp.score(changed, PREDICTIONS), "Q6", "d=64") == expected


def test_q7_fails_if_any_model_reached_the_top_of_its_ladder(report) -> None:
    assert line(sp.score(report, PREDICTIONS), "Q7") == "HIT"
    changed = copy.deepcopy(report)
    changed["16"]["capacity_is_lower_bound"] = True
    assert line(sp.score(changed, PREDICTIONS), "Q7") == "MISS"


@pytest.mark.parametrize("violations, allowed, expected", [(0, 0, "HIT (exact)"), (1, 0, "MISS"), (1, 1, "HIT (exact)"), (0, 1, "HIT")])
def test_control_book_recall_is_counted_over_all_models(report, violations, allowed, expected) -> None:
    changed = copy.deepcopy(report)
    changed["32"]["control_violations"] = violations
    predictions = copy.deepcopy(PREDICTIONS)
    predictions["q8_control_book"]["max_control_violations"] = allowed
    assert line(sp.score(changed, predictions), "Q8") == expected


def test_q10_flags_a_wide_top_model_or_a_steep_doubling_and_ignores_undefined_ratios(report) -> None:
    assert line(sp.score(report, PREDICTIONS), "Q10") == "STORY SURVIVES"
    wide = copy.deepcopy(report)                                       # exactly at the threshold, with doublings of only 2x
    for d, value in (("16", 160), ("32", 320), ("64", 640)):
        wide[d]["capacity_by_budget"]["1000"] = value
    assert line(sp.score(wide, PREDICTIONS), "Q10") == "STORY CHALLENGED"
    below = copy.deepcopy(wide)
    below["64"]["capacity_by_budget"]["1000"] = 453                    # one rung under the threshold, ratio 1.4: survives
    assert line(sp.score(below, PREDICTIONS), "Q10") == "STORY SURVIVES"
    steep = copy.deepcopy(report)
    steep["32"]["capacity_by_budget"]["1000"] = 320                    # 320 / 80 = 4 and d=64 stays below 640
    steep["64"]["capacity_by_budget"]["1000"] = 453
    assert line(sp.score(steep, PREDICTIONS), "Q10") == "STORY CHALLENGED"
    undefined = copy.deepcopy(report)
    undefined["16"]["capacity_by_budget"]["1000"] = None               # d=16 holds nothing: the ratio to it is undefined, not steep
    assert line(sp.score(undefined, PREDICTIONS), "Q10") == "STORY SURVIVES"
    almost = copy.deepcopy(report)
    almost["32"]["capacity_by_budget"]["1000"] = 226                   # 226 / 80 = 2.8 < 4
    assert line(sp.score(almost, PREDICTIONS), "Q10") == "STORY SURVIVES"


def test_the_command_line_prints_the_table_and_writes_json(report, tmp_path: Path) -> None:
    files = {"capacity": report, "predictions": PREDICTIONS}
    paths = {}
    for name, content in files.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(content), encoding="utf-8")
    out = tmp_path / "scores.json"
    done = subprocess.run([sys.executable, str(ROOT / "phonebook" / "score_e4_predictions.py"), "--capacity", str(paths["capacity"]),
                           "--predictions", str(paths["predictions"]), "--json", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    assert "23 hits of 23 scored lines" in done.stdout
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 24
