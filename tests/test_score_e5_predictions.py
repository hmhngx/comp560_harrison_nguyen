"""The E5 scorer applies rules fixed before any result existed; each rule is pinned here by a hit and a miss."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from phonebook import score_e5_predictions as sp

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "phonebook" / "preregistration" / "e5_predictions.json"
LOCKED_SHA256 = "dc4ac01dd23a1f6f223d74f871a77da29a36e7c0864f096cd24aa9ffd116a8be"


def pair(faster, tied, slower, gain, beats):
    return {"faster": faster, "tied": tied, "slower": slower, "median_gain": gain, "band": 0, "beats": beats}


@pytest.fixture
def predictions() -> dict:
    return json.loads(LOCKED.read_text(encoding="utf-8"))


@pytest.fixture
def primary() -> dict:
    """Numbers shaped like a real analysis at lr 0.001 (k 1 2 3 4 5 9)."""
    return {"noise": {"band": 0, "groups": 6}, "lr": {"0.001": {
        "median_stable_joint": {"1": 40.0, "2": 38.5, "3": 38.5, "4": 40.5, "5": 44.0, "9": 54.0},
        "paired": {"2": pair(5, 0, 1, 1.5, True), "3": pair(3, 0, 3, 0.5, False), "4": pair(1, 1, 4, -1.5, False),
                   "5": pair(1, 1, 4, -6.0, False), "9": pair(1, 0, 5, -15.5, False)},
        "plateau": [2, 3], "cheapest_k_in_work": 1,
        "median_min_a": {"1": 9.0, "2": 6.5, "3": 4.0, "4": 3.0, "5": 2.0, "9": 0.5}}}}


@pytest.fixture
def robust() -> dict:
    return {"lr": {"0.0003": {"median_stable_joint": {"1": 96.5, "2": 93.5, "3": 91.5, "5": 96.5}}}}


def write_cells(path: Path, *, censored: int = 18, main: int = 270, noise: int = 30) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["phonebook", "init", "seed", "lr", "k", "plain", "condition", "stable_joint", "first_joint"])
        for index in range(main):
            writer.writerow([2, 42, 42, 0.001, 1, False, "k1", "" if index < censored else 40, 40])
        for index in range(noise):
            writer.writerow([2, 42, 1 + index % 5, 0.001, 1, False, "k1", "", 40])    # noise cells never count
    return path


@pytest.fixture
def cells(tmp_path: Path) -> Path:
    return write_cells(tmp_path / "cells.csv")


def by_line(rows: list[dict]) -> dict:
    return {(r["q"], r["what"]): r["result"] for r in rows}


def line(rows: list[dict], q: str, what_contains: str = "") -> str:
    return next(r["result"] for r in rows if r["q"] == q and what_contains in r["what"])


def test_the_locked_predictions_file_is_unchanged() -> None:
    assert hashlib.sha256(LOCKED.read_bytes()).hexdigest() == LOCKED_SHA256


def test_a_realistic_analysis_scores_thirteen_of_fifteen_with_the_two_known_misses(predictions, primary, robust, cells) -> None:
    rows = sp.score(primary, robust, cells, predictions)
    verdicts = by_line(rows)
    assert verdicts[("Q1", "median stable_joint at k=1")] == "HIT"
    assert [verdicts[("Q2", f"phonebooks faster than k=1, k={k}")] for k in (2, 3, 4, 5, 9)] == \
        ["HIT (exact)", "HIT", "HIT", "HIT", "MISS"]
    assert verdicts[("Q3", "noise band")] == "HIT (exact)"
    assert verdicts[("Q4", "k that beat k=1 under the rule")] == "HIT"
    assert verdicts[("Q5", "plateau")] == "MISS"
    assert (verdicts[("Q6", "cheapest k in work")], verdicts[("Q6", "fastest k in updates")]) == ("HIT", "HIT")
    assert (verdicts[("Q7", "median dip at k=1")], verdicts[("Q7", "median dip at k=9")]) == ("HIT (exact)", "HIT")
    assert verdicts[("Q8", "main-spec runs with no stable joint recall")] == "HIT (exact)"
    assert verdicts[("Q9", "best k at lr 0.0003 vs 0.001")] == "HIT"
    assert verdicts[("Q10", "tripwires fired")] == "STORY SURVIVES"
    assert sp.render(rows).splitlines()[-1] == "13 hits of 15 scored lines (Q10 is a tripwire report, not scored)"


def test_a_range_is_inclusive_at_both_ends_and_nothing_outside_counts(predictions, primary, robust, cells) -> None:
    for value, expected in ((34, "HIT"), (43, "HIT"), (38, "HIT (exact)"), (33.5, "MISS"), (43.5, "MISS"), (None, "MISS")):
        changed = copy.deepcopy(primary)
        changed["lr"]["0.001"]["median_stable_joint"]["1"] = value
        assert line(sp.score(changed, robust, cells, predictions), "Q1") == expected, value


def test_q2_counts_are_judged_against_each_ks_own_range(predictions, primary, robust, cells) -> None:
    changed = copy.deepcopy(primary)
    changed["lr"]["0.001"]["paired"]["2"]["faster"] = 3          # below the locked 4 to 6
    changed["lr"]["0.001"]["paired"]["9"]["faster"] = 0          # exactly the locked 0 to 0
    rows = sp.score(changed, robust, cells, predictions)
    assert line(rows, "Q2", "k=2") == "MISS" and line(rows, "Q2", "k=9") == "HIT (exact)"


def test_the_noise_band_prediction_covers_zero_and_one_only(predictions, primary, robust, cells) -> None:
    results = {}
    for band in (0, 1, 2):
        changed = copy.deepcopy(primary)
        changed["noise"]["band"] = band
        results[band] = line(sp.score(changed, robust, cells, predictions), "Q3")
    assert results == {0: "HIT (exact)", 1: "HIT", 2: "MISS"}


@pytest.mark.parametrize("beats, expected", [
    ({"2": True}, "HIT"),                          # exactly the locked answer: k=2 alone
    ({"2": True, "3": True}, "MISS"),              # an extra winner is a miss, no partial credit
    ({}, "MISS"),                                  # nobody beats k=1
    ({"3": True}, "MISS"),                         # the wrong winner
])
def test_q4_is_a_hit_only_when_k_equals_two_alone_beats_k_equals_one(predictions, primary, robust, cells, beats, expected) -> None:
    changed = copy.deepcopy(primary)
    for k, entry in changed["lr"]["0.001"]["paired"].items():
        entry["beats"] = beats.get(k, False)
    assert line(sp.score(changed, robust, cells, predictions), "Q4") == expected


@pytest.mark.parametrize("band, plateau, expected", [
    (0, [2], "HIT"),
    (0, [2, 3], "MISS"),                           # a tie at the best median widens the plateau: the real E5 case
    (1, [1, 2, 3], "HIT"),
    (1, [2, 3], "MISS"),
    (2, [2], "MISS"),                              # the prediction covers bands 0 and 1 only
    (0, [2, 4], "MISS"),                           # any other plateau, including one with an excluded k, is a miss
])
def test_q5_plateau_must_equal_the_prediction_for_the_actual_band(predictions, primary, robust, cells, band, plateau, expected) -> None:
    changed = copy.deepcopy(primary)
    changed["noise"]["band"] = band
    changed["lr"]["0.001"]["plateau"] = plateau
    assert line(sp.score(changed, robust, cells, predictions), "Q5") == expected


def test_ties_go_to_the_smaller_k_and_a_null_median_is_infinitely_slow() -> None:
    assert sp.argmin_k({"1": 38.5, "2": 38.5, "3": 40.0}) == 1
    assert sp.argmin_k({"2": 38.5, "3": 38.5}) == 2
    assert sp.argmin_k({"1": None, "2": 50.0}) == 2
    assert sp.argmin_k({"1": 40.0, "2": 38.5, "3": 38.5, "4": 40.5}) == 2


def test_q6_fastest_k_depends_on_the_tie_rule_and_cheapest_on_the_analysis(predictions, primary, robust, cells) -> None:
    changed = copy.deepcopy(primary)
    changed["lr"]["0.001"]["median_stable_joint"]["1"] = 38.5          # k=1 ties the best: the smaller k wins
    changed["lr"]["0.001"]["cheapest_k_in_work"] = 2
    rows = sp.score(changed, robust, cells, predictions)
    assert line(rows, "Q6", "fastest") == "MISS" and line(rows, "Q6", "cheapest") == "MISS"


@pytest.mark.parametrize("censored, main, expected", [(18, 270, "HIT (exact)"), (20, 270, "HIT"), (21, 270, "MISS"),
                                                      (17, 270, "MISS"), (18, 269, "MISS")])
def test_q8_counts_only_main_grid_cells_and_requires_all_270(predictions, primary, robust, tmp_path, censored, main, expected) -> None:
    rows = sp.score(primary, robust, write_cells(tmp_path / "c.csv", censored=censored, main=main), predictions)
    assert line(rows, "Q8") == expected


def test_q8_ignores_censored_noise_cells(predictions, primary, robust, tmp_path) -> None:
    rows = sp.score(primary, robust, write_cells(tmp_path / "c.csv", noise=30), predictions)
    assert next(r["actual"] for r in rows if r["q"] == "Q8") == 18       # the 30 blank noise cells are not counted


@pytest.mark.parametrize("low_medians, expected", [
    ({"1": 96.5, "2": 93.5, "3": 91.5, "5": 96.5}, "HIT"),         # best k 3 at 0.0003 against 2 at 0.001: larger
    ({"1": 96.5, "2": 93.5, "3": 95.0, "5": 99.0}, "MISS"),        # best k 2 equals the primary best: same
    ({"1": 90.0, "2": 93.5, "3": 95.0, "5": 99.0}, "MISS"),        # best k 1: smaller
])
def test_q9_compares_the_best_k_at_the_two_learning_rates(predictions, primary, cells, low_medians, expected) -> None:
    robust = {"lr": {"0.0003": {"median_stable_joint": low_medians}}}
    assert line(sp.score(primary, robust, cells, predictions), "Q9") == expected


def test_q10_reports_the_two_tripwires(predictions, primary, robust, cells) -> None:
    quiet = sp.score(primary, robust, cells, predictions)
    assert line(quiet, "Q10") == "STORY SURVIVES"
    for path, value in (("paired4", 3), ("paired5", 3), ("band", 3)):
        changed = copy.deepcopy(primary)
        if path == "band":
            changed["noise"]["band"] = value
        else:
            changed["lr"]["0.001"]["paired"][path[-1]]["faster"] = value
        assert line(sp.score(changed, robust, cells, predictions), "Q10") == "STORY CHALLENGED", path
    almost = copy.deepcopy(primary)
    almost["lr"]["0.001"]["paired"]["4"]["faster"] = 2
    almost["noise"]["band"] = 2
    assert line(sp.score(almost, robust, cells, predictions), "Q10") == "STORY SURVIVES"


def test_the_command_line_prints_the_table_and_writes_json(predictions, primary, robust, cells, tmp_path) -> None:
    files = {name: tmp_path / f"{name}.json" for name in ("primary", "robust")}
    files["primary"].write_text(json.dumps(primary), encoding="utf-8")
    files["robust"].write_text(json.dumps(robust), encoding="utf-8")
    out = tmp_path / "scores.json"
    done = subprocess.run([sys.executable, str(ROOT / "phonebook" / "score_e5_predictions.py"), "--primary", str(files["primary"]),
                           "--robust", str(files["robust"]), "--cells", str(cells), "--predictions", str(LOCKED), "--json", str(out)],
                          capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    assert "13 hits of 15 scored lines" in done.stdout
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [r["q"] for r in written][:3] == ["Q1", "Q2", "Q2"] and len(written) == 16
