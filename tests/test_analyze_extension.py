"""The E6 analysis reads the trainer's logs of the extended capacity runs; each rule fixed before E6 is pinned here."""
from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import pytest

from phonebook import analyze_extension as ae

ROOT = Path(__file__).resolve().parents[1]


def pts(*triples):
    """Logged curve points as (epoch, A = training entries recalled, B = control entries recalled)."""
    return [{"epoch": e, "A": a, "B": b} for e, a, b in triples]


def run(seed: int, n: int, a_at: dict | None = None, *, default: int = 0, b_at: dict | None = None,
        last: int = 4000, step: int = 500) -> dict:
    """A run logged every `step` epochs: A is `default` except where `a_at` says otherwise."""
    a_at, b_at = a_at or {}, b_at or {}
    return {"seed": seed, "n": n, "points": [{"epoch": e, "A": a_at.get(e, default), "B": b_at.get(e, 0)}
                                             for e in range(step, last + 1, step)]}


BUDGETS = (1000, 2000, 3000, 4000)


# ------------------------------------------------------------------ windows

def test_a_window_excludes_its_lower_epoch_and_includes_its_upper_epoch() -> None:
    points = pts((1000, 1, 0), (1010, 2, 0), (2000, 3, 0), (2010, 4, 0))
    assert [p["epoch"] for p in ae.window_points(points, 1000, 2000)] == [1010, 2000]


def test_the_window_mean_is_the_exact_fraction_of_entries_recalled() -> None:
    points = pts((1500, 5, 0), (2000, 10, 0))
    assert ae.window_mean(points, 10, 1000, 2000) == Fraction(3, 4)        # (5 + 10) / (10 * 2)


def test_a_window_that_the_run_never_reached_is_an_error_not_a_partial_mean() -> None:
    with pytest.raises(ValueError, match="2000"):
        ae.window_mean(pts((1500, 5, 0)), 10, 1000, 2000)                   # the run stopped before the window closed
    with pytest.raises(ValueError, match="no logged epoch"):
        ae.window_mean(pts((500, 5, 0)), 10, 1000, 2000)


def test_a_missing_recall_count_is_an_error() -> None:
    with pytest.raises(ValueError, match="recall"):
        ae.window_mean(pts((1500, None, 0), (2000, 10, 0)), 10, 1000, 2000)


def test_the_pooled_mean_weighs_every_seed_equally() -> None:
    runs = [run(1, 10, {1500: 10, 2000: 10}), run(2, 10, {1500: 0, 2000: 0}), run(3, 10, {1500: 5, 2000: 5})]
    assert ae.pooled_window_mean(runs, 1000, 2000) == Fraction(1, 2)


def test_seeds_that_logged_different_epochs_cannot_be_pooled() -> None:
    runs = [run(1, 10, step=500), run(2, 10, step=250)]
    with pytest.raises(ValueError, match="different epochs"):
        ae.pooled_window_mean(runs, 1000, 2000)


# ------------------------------------------------------------------ trend rule

@pytest.mark.parametrize("gain, expected", [
    (Fraction(5, 1000), "rising"),             # exactly at the threshold counts
    (Fraction(4999, 1000000), "flat"),
    (Fraction(0), "flat"),
    (Fraction(-4999, 1000000), "flat"),
    (Fraction(-5, 1000), "falling"),           # exactly at the negative threshold counts
    (Fraction(1, 10), "rising"),
    (Fraction(-1, 10), "falling"),
])
def test_the_trend_is_rising_flat_or_falling_at_plus_or_minus_half_a_point(gain, expected) -> None:
    assert ae.trend(gain) == expected


def test_the_locked_windows_and_threshold() -> None:
    assert ae.THRESHOLD == Fraction(5, 1000)
    assert ae.EARLY == (1000, 2000) and ae.LATE == (3000, 4000) and ae.PREFIX == 1000


# ------------------------------------------------------------------ the first 1000 epochs against E4

def test_the_first_epochs_must_equal_the_earlier_run_point_for_point() -> None:
    old = pts((10, 1, 0), (500, 4, 0), (1000, 6, 0), (1010, 6, 0))
    assert ae.prefix_equal(old, old, 1000) is True
    new = pts((10, 1, 0), (500, 4, 0), (1000, 6, 0), (1010, 7, 0), (1020, 9, 0))
    assert ae.prefix_equal(new, old, 1000) is True                          # later epochs are free to differ


@pytest.mark.parametrize("new, expected", [
    (pts((10, 1, 0), (500, 5, 0), (1000, 6, 0)), False),                       # a different recall count
    (pts((10, 1, 0), (500, 4, 1), (1000, 6, 0)), False),                       # a different control count
    (pts((10, 1, 0), (1000, 6, 0)), False),                                    # a missing epoch
    (pts((10, 1, 0), (500, 4, 0), (999, 5, 0), (1000, 6, 0)), False),          # an extra epoch at or below the cutoff
    (pts((10, 1, 0), (500, 4, 0), (1000, 6, 0), (1005, 6, 0)), True),          # an extra epoch above the cutoff is free
])
def test_any_difference_up_to_the_cutoff_breaks_the_match(new, expected) -> None:
    old = pts((10, 1, 0), (500, 4, 0), (1000, 6, 0), (1010, 6, 0))
    assert ae.prefix_equal(new, old, 1000) is expected


def test_two_empty_prefixes_prove_nothing() -> None:
    assert ae.prefix_equal([], [], 1000) is False
    assert ae.prefix_equal(pts((2000, 1, 0)), pts((2000, 1, 0)), 1000) is False       # nothing at or below 1000 to compare


def test_a_run_that_stopped_before_the_cutoff_does_not_match() -> None:
    old = pts((500, 4, 0), (1000, 6, 0))
    assert ae.prefix_equal(pts((500, 4, 0)), old, 1000) is False
    assert ae.prefix_equal(old, pts((500, 4, 0)), 1000) is False
    assert ae.prefix_equal(pts((500, 4, 0)), pts((500, 4, 0)), 1000) is False          # neither run reached the cutoff


# ------------------------------------------------------------------ one width

def seeds(first_epochs, n=20, **kwargs):
    """Three seeds; first_epochs[i] is the epoch from which seed i+1 recalls every entry (None: never)."""
    out = []
    for seed, first in enumerate(first_epochs, start=1):
        a_at = {e: n for e in range(500, 4001, 500) if first is not None and e >= first}
        out.append(run(seed, n, a_at, default=n // 2, **kwargs))
    return out


def test_a_rung_passes_when_two_of_three_seeds_are_perfect_within_the_budget() -> None:
    report = ae.width_report(seeds([1500, 3500, None]), BUDGETS)
    assert report["perfect_by"] == {"1000": 0, "2000": 1, "3000": 1, "4000": 2}
    assert report["passes"] == {"1000": False, "2000": False, "3000": False, "4000": True}
    assert report["first_pass_budget"] == 4000
    assert report["first_perfect"] == {"1": 1500, "2": 3500, "3": None}


def test_a_rung_that_never_passes_has_no_first_passing_budget() -> None:
    report = ae.width_report(seeds([1500, None, None]), BUDGETS)
    assert report["passes"]["4000"] is False and report["first_pass_budget"] is None


def test_the_seed_rule_is_a_parameter() -> None:
    assert ae.width_report(seeds([500, 500, None]), BUDGETS)["first_pass_budget"] == 1000
    assert ae.width_report(seeds([500, 500, None]), BUDGETS, min_seeds=3)["first_pass_budget"] is None
    with pytest.raises(ValueError, match="min_seeds"):
        ae.width_report(seeds([500, 500, None]), BUDGETS, min_seeds=4)


def test_reaching_100_percent_and_still_being_there_at_the_end_are_counted_separately() -> None:
    dropped = run(2, 20, {1500: 20, 2000: 20}, default=19)                  # perfect for a while, then 19 of 20 at the end
    report = ae.width_report([seeds([500])[0], dropped, run(3, 20, default=10)], BUDGETS)
    assert report["reached_perfect"] == 2 and report["perfect_at_end"] == 1


def test_the_end_means_the_last_logged_epoch_of_the_run() -> None:
    runs = seeds([500, 500, 500])
    runs[0]["points"][-1]["A"] = 19                                         # one entry lost at epoch 4000
    assert ae.width_report(runs, BUDGETS)["perfect_at_end"] == 2


def test_control_violations_count_runs_with_any_control_entry_recalled() -> None:
    runs = seeds([500, 500, 500])
    runs[0] = run(1, 20, {e: 20 for e in range(500, 4001, 500)}, b_at={3000: 1})
    runs[1] = run(2, 20, {e: 20 for e in range(500, 4001, 500)}, b_at={500: 2, 1000: 2})
    assert ae.width_report(runs, BUDGETS)["control_violations"] == 2


def test_the_gain_is_the_late_window_mean_minus_the_early_window_mean_and_sets_the_trend() -> None:
    def gain_case(early_a: int, late_a: int, n: int = 1000):
        a_at = {1500: early_a, 2000: early_a, 3500: late_a, 4000: late_a}
        return ae.width_report([run(s, n, a_at, default=early_a) for s in (1, 2, 3)], BUDGETS)

    rising = gain_case(970, 975)                                            # +0.005 exactly
    assert rising["mean_accuracy"] == {"1001_2000": 0.97, "3001_4000": 0.975}
    assert rising["gain"] == 0.005 and rising["trend"] == "rising"
    assert gain_case(970, 974)["trend"] == "flat"
    assert gain_case(975, 970)["trend"] == "falling" and gain_case(975, 970)["gain"] == -0.005
    assert gain_case(975, 975)["gain"] == 0.0


def test_a_width_runs_one_rung_only() -> None:
    mixed = seeds([500, 500]) + [run(3, 40, {500: 40}, default=40)]
    with pytest.raises(ValueError, match="one rung"):
        ae.width_report(mixed, BUDGETS)


def test_the_report_carries_the_rung_and_the_number_of_seeds() -> None:
    report = ae.width_report(seeds([500, 500, 500]), BUDGETS)
    assert report["n"] == 20 and report["seeds"] == 3


def test_the_comparison_with_the_earlier_run_is_per_seed_and_all_must_match() -> None:
    runs = seeds([500, 1500, None])
    earlier = {(r["seed"], r["n"]): [p for p in r["points"] if p["epoch"] <= 1000] for r in runs}
    ok = ae.width_report(runs, BUDGETS, earlier=earlier)
    assert ok["first_1000_matches_earlier"] is True and ok["first_1000_match_by_seed"] == {"1": True, "2": True, "3": True}
    earlier[(2, 20)] = [dict(p, A=p["A"] + 1) for p in earlier[(2, 20)]]
    bad = ae.width_report(runs, BUDGETS, earlier=earlier)
    assert bad["first_1000_matches_earlier"] is False and bad["first_1000_match_by_seed"]["2"] is False


def test_without_an_earlier_run_the_comparison_is_unknown_not_true() -> None:
    assert ae.width_report(seeds([500, 500, 500]), BUDGETS)["first_1000_matches_earlier"] is None


def test_an_earlier_run_that_is_missing_is_an_error() -> None:
    runs = seeds([500, 500, 500])
    with pytest.raises(ValueError, match="earlier run"):
        ae.width_report(runs, BUDGETS, earlier={(1, 20): runs[0]["points"]})


# ------------------------------------------------------------------ all widths and the report file

def test_the_summary_is_keyed_by_width_and_serialises_as_strict_json() -> None:
    def refuse(constant):
        raise ValueError(constant)

    runs = {8: seeds([500, 500, 500], n=20), 16: seeds([500, 500, None], n=160)}
    summary = ae.summarize_extension(runs, BUDGETS)
    assert list(summary) == ["8", "16"] and summary["16"]["n"] == 160
    text = json.dumps(summary, allow_nan=False)
    assert json.loads(text, parse_constant=refuse)["8"]["passes"]["4000"] is True


# ------------------------------------------------------------------ evidence on disk

def fake_evidence(root: Path, runs: list[dict]) -> None:
    root.mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps({"runs": [r["protocol"] for r in runs]}), encoding="utf-8")
    for r in runs:
        folder = root / r["protocol"]["scenario"]
        (folder / r["protocol"]["condition"]).mkdir(parents=True)
        books = {"A": {"generation": {"total": r["n"], "correct": r["curve"][-1][1]}},
                 "B": {"generation": {"total": 20, "correct": 0}}}
        (folder / f"{r['protocol']['condition']}.evaluation.json").write_text(json.dumps({"books": books}), encoding="utf-8")
        with (folder / r["protocol"]["condition"] / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Token Acc", "Val Token Acc",
                             "Train Seq Acc", "Val Seq Acc", "Training Time (sec)", "Param Count"])
            for epoch, correct in r["curve"]:
                writer.writerow([epoch, "1.0", "5.0", "0.5", "0.1", f"{correct / r['n']:.4f}", "0.0000", "0.1", r["params"]])


def protocol_run(scenario: str, n: int, d: int, epochs: int) -> dict:
    return {"id": f"{scenario}/c", "scenario": scenario, "condition": f"d{d}", "mode": "a1b0_evalB",
            "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B", "lr": 0.001, "epochs": epochs,
            "batch_size": 64, "accuracy_interval": 500, "l2sp_lambda": 0.0, "adapt": False,
            "n_train": n, "updates_per_epoch": 1, "seq_len": 30,
            "model": {"embedding_dim": d, "n_heads": 2, "n_layers": 2}}


def evidence_runs(d: int, n: int, epochs: int, first: dict) -> list[dict]:
    """first[seed] is the epoch from which that seed recalls all n entries (None: never)."""
    out = []
    for seed in (1, 2, 3):
        curve = [(e, n if first[seed] is not None and e >= first[seed] else n // 2) for e in range(500, epochs + 1, 500)]
        out.append({"protocol": protocol_run(f"s{seed}_n{n}", n, d, epochs), "n": n, "params": 9216, "curve": curve})
    return out


def test_loading_reads_width_seed_rung_and_the_logged_points(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 1500, 2: None, 3: 500}))
    loaded = ae.load_extension_runs(tmp_path / "e6")
    assert sorted((r["d"], r["seed"], r["n"]) for r in loaded) == [(16, 1, 160), (16, 2, 160), (16, 3, 160)]
    first = next(r for r in loaded if r["seed"] == 1)
    assert first["points"][0] == {"epoch": 500, "A": 80, "B": 0} and first["points"][2]["A"] == 160


def test_a_scenario_that_disagrees_with_the_book_size_is_rejected(tmp_path: Path) -> None:
    runs = evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500})
    runs[0]["protocol"]["scenario"] = "s1_n20"
    fake_evidence(tmp_path / "e6", runs)
    with pytest.raises(ValueError, match="book size"):
        ae.load_extension_runs(tmp_path / "e6")


def test_a_scenario_id_outside_the_naming_scheme_is_rejected(tmp_path: Path) -> None:
    runs = evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500})
    runs[0]["protocol"]["scenario"] = "weird"
    fake_evidence(tmp_path / "e6", runs)
    with pytest.raises(ValueError, match="scenario id"):
        ae.load_extension_runs(tmp_path / "e6")


def test_a_run_without_a_model_override_is_rejected(tmp_path: Path) -> None:
    runs = evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500})
    del runs[0]["protocol"]["model"]
    fake_evidence(tmp_path / "e6", runs)
    with pytest.raises(ValueError, match="model"):
        ae.load_extension_runs(tmp_path / "e6")


def test_the_earlier_runs_are_matched_by_width_as_well_as_by_seed_and_rung() -> None:
    narrow, wide = seeds([500, 500, 500]), seeds([500, 500, 500])
    earlier = {(8, r["seed"], 20): [p for p in r["points"] if p["epoch"] <= 1000] for r in narrow}
    earlier.update({(16, r["seed"], 20): [dict(p, A=p["A"] - 1) for p in r["points"] if p["epoch"] <= 1000] for r in wide})
    report = ae.summarize_extension({8: narrow, 16: wide}, BUDGETS, earlier=earlier)
    assert report["8"]["first_1000_matches_earlier"] is True and report["16"]["first_1000_matches_earlier"] is False


def test_the_command_line_seed_threshold_changes_which_rungs_pass(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: None}))     # two of the three seeds memorize

    def first_pass(min_seeds: int):
        out = tmp_path / f"out{min_seeds}"
        ae.main(["--root", str(tmp_path / "e6"), "--budgets", "1000", "4000", "--out", str(out), "--min-seeds", str(min_seeds)])
        return json.loads((out / "e6_extension.json").read_text(encoding="utf-8"))["16"]["first_pass_budget"]

    assert first_pass(2) == 1000
    assert first_pass(3) is None


def test_an_earlier_index_refuses_two_runs_of_the_same_cell(tmp_path: Path) -> None:
    for name in ("a", "b"):
        fake_evidence(tmp_path / name, evidence_runs(16, 160, 1000, {1: 500, 2: 500, 3: 500}))
    with pytest.raises(ValueError, match="twice"):
        ae.index_runs([r for name in ("a", "b") for r in ae.load_extension_runs(tmp_path / name)])


def test_the_command_line_compares_with_the_earlier_evidence_and_writes_the_report(tmp_path: Path) -> None:
    first = {1: 500, 2: 1500, 3: None}
    fake_evidence(tmp_path / "e6_d16", evidence_runs(16, 160, 4000, first))
    fake_evidence(tmp_path / "e6_d8", evidence_runs(8, 20, 4000, {1: 500, 2: 500, 3: 500}))
    fake_evidence(tmp_path / "e4_d16", evidence_runs(16, 160, 1000, first))
    fake_evidence(tmp_path / "e4_d8", evidence_runs(8, 20, 1000, {1: 500, 2: 500, 3: 500}))
    out = tmp_path / "out"
    ae.main(["--root", str(tmp_path / "e6_d16"), "--root", str(tmp_path / "e6_d8"),
             "--earlier-root", str(tmp_path / "e4_d16"), "--earlier-root", str(tmp_path / "e4_d8"),
             "--budgets", "1000", "2000", "3000", "4000", "--out", str(out)])
    report = json.loads((out / "e6_extension.json").read_text(encoding="utf-8"))
    assert list(report) == ["8", "16"]
    assert report["16"]["passes"] == {"1000": False, "2000": True, "3000": True, "4000": True}
    assert report["16"]["first_pass_budget"] == 2000 and report["8"]["first_pass_budget"] == 1000
    assert report["16"]["first_1000_matches_earlier"] is True and report["8"]["first_1000_matches_earlier"] is True


def test_the_command_line_notices_when_the_first_1000_epochs_differ(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500}))
    fake_evidence(tmp_path / "e4", evidence_runs(16, 160, 1000, {1: 500, 2: 500, 3: None}))
    out = tmp_path / "out"
    ae.main(["--root", str(tmp_path / "e6"), "--earlier-root", str(tmp_path / "e4"),
             "--budgets", "1000", "2000", "3000", "4000", "--out", str(out)])
    report = json.loads((out / "e6_extension.json").read_text(encoding="utf-8"))
    assert report["16"]["first_1000_matches_earlier"] is False
    assert report["16"]["first_1000_match_by_seed"] == {"1": True, "2": True, "3": False}


def test_without_earlier_evidence_the_command_line_leaves_the_comparison_unknown(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500}))
    ae.main(["--root", str(tmp_path / "e6"), "--budgets", "1000", "4000", "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "e6_extension.json").read_text(encoding="utf-8"))
    assert report["16"]["first_1000_matches_earlier"] is None


def test_the_command_line_refuses_an_empty_evidence_folder(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "protocol.json").write_text(json.dumps({"runs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no extension runs"):
        ae.main(["--root", str(tmp_path / "empty"), "--budgets", "1000", "4000", "--out", str(tmp_path / "out")])


def test_a_budget_beyond_the_trained_epochs_is_rejected(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500}))
    with pytest.raises(ValueError, match="budget 5000"):
        ae.main(["--root", str(tmp_path / "e6"), "--budgets", "1000", "5000", "--out", str(tmp_path / "out")])


# ------------------------------------------------------------------ E7: the compared prefix is a parameter

def with_earlier(runs: list[dict], up_to: int, change: dict | None = None) -> dict:
    """The earlier runs of the same cells: the first `up_to` epochs of `runs`, optionally with one seed's counts shifted."""
    change = change or {}
    return {(r["seed"], r["n"]): [dict(p, A=p["A"] + change.get(r["seed"], 0)) for p in r["points"] if p["epoch"] <= up_to]
            for r in runs}


def test_the_default_prefix_keeps_the_historic_key_names_in_the_historic_order() -> None:
    runs = seeds([500, 1500, None])
    report = ae.width_report(runs, BUDGETS, earlier=with_earlier(runs, 1000))
    assert list(report)[-2:] == ["first_1000_match_by_seed", "first_1000_matches_earlier"]
    assert report["first_1000_matches_earlier"] is True


def test_a_longer_prefix_renames_the_keys_and_compares_that_many_epochs() -> None:
    runs = seeds([500, 1500, None])
    longer = ae.width_report(runs, BUDGETS, earlier=with_earlier(runs, 2000), prefix=2000)
    assert longer["first_2000_match_by_seed"] == {"1": True, "2": True, "3": True} and longer["first_2000_matches_earlier"] is True
    assert "first_1000_match_by_seed" not in longer and "first_1000_matches_earlier" not in longer
    assert list(longer)[-2:] == ["first_2000_match_by_seed", "first_2000_matches_earlier"]


def test_a_difference_between_epoch_1000_and_epoch_2000_is_seen_only_by_the_longer_prefix() -> None:
    runs = seeds([500, 1500, None])
    earlier = with_earlier(runs, 2000, change={2: 1})                      # the earlier run of seed 2 differs at epochs 1500 and 2000
    earlier[(2, 20)] = [dict(p, A=runs[1]["points"][i]["A"]) if p["epoch"] <= 1000 else p
                        for i, p in enumerate(earlier[(2, 20)])]            # ... but is identical through epoch 1000
    short = ae.width_report(runs, BUDGETS, earlier=earlier, prefix=1000)
    long = ae.width_report(runs, BUDGETS, earlier=earlier, prefix=2000)
    assert short["first_1000_match_by_seed"] == {"1": True, "2": True, "3": True}
    assert long["first_2000_match_by_seed"] == {"1": True, "2": False, "3": True}
    assert long["first_2000_matches_earlier"] is False


def test_the_prefix_is_passed_to_every_width_of_the_summary() -> None:
    narrow, wide = seeds([500, 500, 500]), seeds([500, 500, 500])
    earlier = {(d, r["seed"], 20): [p for p in r["points"] if p["epoch"] <= 2000] for d, rs in ((8, narrow), (16, wide)) for r in rs}
    report = ae.summarize_extension({8: narrow, 16: wide}, BUDGETS, earlier=earlier, prefix=2000)
    assert report["8"]["first_2000_matches_earlier"] is True and report["16"]["first_2000_matches_earlier"] is True
    assert "first_1000_matches_earlier" not in report["8"]


def test_without_an_earlier_run_a_longer_prefix_leaves_the_comparison_unknown() -> None:
    report = ae.width_report(seeds([500, 500, 500]), BUDGETS, prefix=2000)
    assert report["first_2000_match_by_seed"] is None and report["first_2000_matches_earlier"] is None


def test_the_command_line_compares_the_requested_prefix_and_names_the_earlier_runs(tmp_path: Path, capsys) -> None:
    arm = evidence_runs(16, 160, 4000, {1: 500, 2: 1500, 3: None})
    fake_evidence(tmp_path / "arm", arm)
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 1500, 3: None}))
    out = tmp_path / "out"
    ae.main(["--root", str(tmp_path / "arm"), "--earlier-root", str(tmp_path / "e6"), "--budgets", "1000", "2000", "3000", "4000",
             "--prefix-epochs", "2000", "--earlier-label", "E6", "--out", str(out)])
    entry = json.loads((out / "e6_extension.json").read_text(encoding="utf-8"))["16"]
    assert entry["first_2000_matches_earlier"] is True and "first_1000_matches_earlier" not in entry
    assert "first 2000 epochs equal E6: True" in capsys.readouterr().out


def test_a_longer_prefix_catches_a_difference_that_the_default_prefix_misses(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "arm", evidence_runs(16, 160, 4000, {1: 500, 2: 1500, 3: None}))
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 2000, 3: None}))   # seed 2 differs at epoch 1500
    for prefix in (1000, 2000):
        out = tmp_path / f"out{prefix}"
        ae.main(["--root", str(tmp_path / "arm"), "--earlier-root", str(tmp_path / "e6"), "--budgets", "1000", "4000",
                 "--prefix-epochs", str(prefix), "--out", str(out)])
        entry = json.loads((out / "e6_extension.json").read_text(encoding="utf-8"))["16"]
        assert entry[f"first_{prefix}_matches_earlier"] is (prefix == 1000)


def test_by_default_the_command_line_still_compares_1000_epochs_and_calls_them_e4(tmp_path: Path, capsys) -> None:
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 1500, 3: None}))
    fake_evidence(tmp_path / "e4", evidence_runs(16, 160, 1000, {1: 500, 2: 1500, 3: None}))
    ae.main(["--root", str(tmp_path / "e6"), "--earlier-root", str(tmp_path / "e4"), "--budgets", "1000", "4000",
             "--out", str(tmp_path / "out")])
    assert "first 1000 epochs equal E4: True" in capsys.readouterr().out
    entry = json.loads((tmp_path / "out" / "e6_extension.json").read_text(encoding="utf-8"))["16"]
    assert entry["first_1000_matches_earlier"] is True


@pytest.mark.parametrize("value,message", [
    ("0", "^--prefix-epochs must be an integer >= 1$"),
    ("-5", "^--prefix-epochs must be an integer >= 1$"),
    ("4500", "--prefix-epochs 4500 exceeds the 4000 epochs trained$"),
])
def test_a_prefix_outside_the_trained_epochs_is_rejected(tmp_path: Path, value, message) -> None:
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500}))
    with pytest.raises(ValueError, match=message):
        ae.main(["--root", str(tmp_path / "e6"), "--budgets", "1000", "4000", "--prefix-epochs", value, "--out", str(tmp_path / "out")])


# ------------------------------------------------------------------ E7 review note R1: how long a run holds 100 percent

def test_epochs_at_100_counts_the_logged_epochs_at_which_every_entry_is_recalled() -> None:
    points = pts((10, 5, 0), (20, 10, 0), (30, 10, 0), (40, 9, 0), (50, 10, 0))
    assert ae.epochs_at_100(points, 10) == 3
    assert ae.epochs_at_100([], 10) == 0


def test_the_window_of_the_count_excludes_its_lower_epoch_and_includes_its_upper_epoch() -> None:
    points = pts((10, 5, 0), (20, 10, 0), (30, 10, 0), (40, 9, 0), (50, 10, 0))
    assert ae.epochs_at_100(points, 10, (20, 50)) == 2                      # epochs 30 and 50; epoch 20 is the excluded lower edge
    assert ae.epochs_at_100(points, 10, (20, 40)) == 1
    assert ae.epochs_at_100(points, 10, (0, 20)) == 1


def test_a_missing_recall_count_cannot_be_counted_as_not_perfect() -> None:
    with pytest.raises(ValueError, match="^a logged epoch has no recall count$"):
        ae.epochs_at_100(pts((10, None, 0), (20, 10, 0)), 10)


def held_runs() -> list[dict]:
    """n = 20, logged every 500 epochs up to 4000 (8 logged epochs; epochs 3500 and 4000 fall in the late window)."""
    return [run(1, 20, {e: 20 for e in (500, 1000, 3500, 4000)}, default=10),
            run(2, 20, {e: 20 for e in (1500, 2000, 2500, 3000, 3500)}, default=10),
            run(3, 20, default=10)]


HELD = {"logged": {"all": 8, "3001_4000": 2},
        "by_seed": {"1": {"all": 4, "3001_4000": 2}, "2": {"all": 5, "3001_4000": 1}, "3": {"all": 0, "3001_4000": 0}}}


def test_the_held_readout_counts_per_seed_over_the_whole_run_and_over_the_late_window() -> None:
    assert ae.held_report(held_runs(), ae.LATE) == HELD
    assert ae.width_report(held_runs(), BUDGETS, held=True)["epochs_at_100"] == HELD


def test_the_late_window_of_the_held_readout_is_named_like_the_window_means() -> None:
    report = ae.width_report(held_runs(), BUDGETS, early=(500, 1000), late=(1000, 2000), held=True)
    assert set(report["epochs_at_100"]["logged"]) == {"all", "1001_2000"} and set(report["mean_accuracy"]) == {"501_1000", "1001_2000"}
    assert report["epochs_at_100"]["by_seed"]["2"] == {"all": 5, "1001_2000": 2}


def test_without_the_option_the_report_is_exactly_what_it_was_and_with_it_the_report_only_gains_a_key() -> None:
    runs = held_runs()
    plain = ae.width_report(runs, BUDGETS)
    extra = ae.width_report(runs, BUDGETS, held=True)
    assert "epochs_at_100" not in plain
    assert {key: value for key, value in extra.items() if key != "epochs_at_100"} == plain
    assert list(extra)[:-1] == list(plain)


def test_seeds_that_logged_different_numbers_of_epochs_cannot_share_a_held_readout() -> None:
    uneven = [run(1, 20, step=500), run(2, 20, step=250)]
    with pytest.raises(ValueError, match="^the seeds have different numbers of logged epochs, so no single denominator describes them$"):
        ae.held_report(uneven, ae.LATE)


def test_the_summary_passes_the_held_option_to_every_width() -> None:
    summary = ae.summarize_extension({8: held_runs(), 16: held_runs()}, BUDGETS, held=True)
    assert summary["8"]["epochs_at_100"] == HELD and summary["16"]["epochs_at_100"] == HELD
    assert "epochs_at_100" not in ae.summarize_extension({8: held_runs()}, BUDGETS)["8"]


def test_the_command_line_writes_the_held_readout_only_when_asked(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 1500, 3: None}))
    base = ["--root", str(tmp_path / "e6"), "--budgets", "1000", "2000", "3000", "4000"]
    ae.main([*base, "--out", str(tmp_path / "plain")])
    ae.main([*base, "--epochs-at-100", "--out", str(tmp_path / "held")])
    plain = json.loads((tmp_path / "plain" / "e6_extension.json").read_text(encoding="utf-8"))
    held = json.loads((tmp_path / "held" / "e6_extension.json").read_text(encoding="utf-8"))
    assert "epochs_at_100" not in plain["16"]
    assert held["16"]["epochs_at_100"] == {"logged": {"all": 8, "3001_4000": 2},
                                           "by_seed": {"1": {"all": 8, "3001_4000": 2}, "2": {"all": 6, "3001_4000": 2},
                                                       "3": {"all": 0, "3001_4000": 0}}}
    assert {k: v for k, v in held["16"].items() if k != "epochs_at_100"} == plain["16"]


# ---- the committed E4 and E6 evidence: the default output is the committed file, and the new readout is claim 15 ----------

def tracked_roots(fragment: str) -> list[str]:
    listing = subprocess.run(["git", "ls-files", "--", "phonebook/experiments/week_2026-10-31"], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    if listing.returncode != 0:
        pytest.skip("not a git checkout: the tracked evidence cannot be enumerated")
    roots = sorted({str((ROOT / name).parent) for name in listing.stdout.split()
                    if name.endswith("/protocol.json") and fragment in name})
    return roots


def e6_arguments(out: Path, *extra: str) -> list[str]:
    e6, e4 = tracked_roots("/e6_extend_d"), tracked_roots("/e4_capacity_d")
    assert len(e6) == 12 and len(e4) == 24
    arguments = ["--budgets", "1000", "2000", "3000", "4000", "--out", str(out), *extra]
    for root in e6:
        arguments += ["--root", root]
    for root in e4:
        arguments += ["--earlier-root", root]
    return arguments


def test_the_default_analysis_regenerates_the_committed_e6_extension_json_modulo_line_endings(tmp_path: Path) -> None:
    ae.main(e6_arguments(tmp_path / "default"))
    regenerated = (tmp_path / "default" / "e6_extension.json").read_bytes()
    committed = (ROOT / "phonebook" / "results" / "e6_extension.json").read_bytes()    # CRLF in a Windows checkout, LF in git
    assert regenerated.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n")


def test_the_held_readout_reproduces_the_counts_that_the_results_page_states_for_claim_15(tmp_path: Path) -> None:
    ae.main(e6_arguments(tmp_path / "held", "--epochs-at-100", "--prefix", "e6_held"))
    held = json.loads((tmp_path / "held" / "e6_held_extension.json").read_text(encoding="utf-8"))
    by_width = {d: [entry["epochs_at_100"]["by_seed"][seed]["all"] for seed in ("1", "2", "3")] for d, entry in held.items()}
    assert by_width == {"8": [283, 290, 219], "16": [3, 21, 2], "32": [6, 1, 0], "64": [0, 0, 0]}   # defense sheet C19: of 400 logged
    assert {entry["epochs_at_100"]["logged"]["all"] for entry in held.values()} == {400}
    late = {d: [entry["epochs_at_100"]["by_seed"][seed]["3001_4000"] for seed in ("1", "2", "3")] for d, entry in held.items()}
    assert late == {"8": [95, 96, 94], "16": [3, 2, 2], "32": [2, 0, 0], "64": [0, 0, 0]}
    default = json.loads((ROOT / "phonebook" / "results" / "e6_extension.json").read_text(encoding="utf-8"))
    assert {d: {k: v for k, v in entry.items() if k != "epochs_at_100"} for d, entry in held.items()} == default


# ---- review finding: a prefix that no run logged would silently report "no match" for every seed -------------------------

@pytest.mark.parametrize("value", ["1200", "2005", "3999"])
def test_a_prefix_that_is_not_a_logged_epoch_is_rejected(tmp_path: Path, value) -> None:
    fake_evidence(tmp_path / "arm", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500}))     # logged every 500 epochs
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500}))
    with pytest.raises(ValueError, match="--prefix-epochs .* not a logged epoch"):
        ae.main(["--root", str(tmp_path / "arm"), "--earlier-root", str(tmp_path / "e6"), "--budgets", "1000", "4000",
                 "--prefix-epochs", value, "--out", str(tmp_path / "out")])


def test_a_logged_prefix_is_accepted(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "arm", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500}))
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500}))
    ae.main(["--root", str(tmp_path / "arm"), "--earlier-root", str(tmp_path / "e6"), "--budgets", "1000", "4000",
             "--prefix-epochs", "1500", "--out", str(tmp_path / "out")])
    assert json.loads((tmp_path / "out" / "e6_extension.json").read_text(encoding="utf-8"))["16"]["first_1500_matches_earlier"] is True


# ---- added in E7.1a after the planted-bug harness: the edges of the accepted prefix and the order of the held readout --------

def test_the_held_readout_lists_the_seeds_in_order_whatever_order_it_is_given() -> None:
    report = ae.held_report(list(reversed(held_runs())), ae.LATE)
    assert list(report["by_seed"]) == ["1", "2", "3"]
    assert report == HELD


def test_the_smallest_prefix_is_accepted_when_there_is_nothing_to_compare_with(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 500, 3: 500}))
    ae.main(["--root", str(tmp_path / "e6"), "--budgets", "1000", "4000", "--prefix-epochs", "1", "--out", str(tmp_path / "out")])
    entry = json.loads((tmp_path / "out" / "e6_extension.json").read_text(encoding="utf-8"))["16"]
    assert entry["first_1_match_by_seed"] is None and entry["first_1_matches_earlier"] is None


def test_a_prefix_as_long_as_the_runs_is_accepted(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "arm", evidence_runs(16, 160, 4000, {1: 500, 2: 1500, 3: None}))
    fake_evidence(tmp_path / "e6", evidence_runs(16, 160, 4000, {1: 500, 2: 1500, 3: None}))
    ae.main(["--root", str(tmp_path / "arm"), "--earlier-root", str(tmp_path / "e6"), "--budgets", "1000", "4000",
             "--prefix-epochs", "4000", "--out", str(tmp_path / "out")])
    entry = json.loads((tmp_path / "out" / "e6_extension.json").read_text(encoding="utf-8"))["16"]
    assert entry["first_4000_matches_earlier"] is True


# ---- added in E7.1a after the planted-bug harness: --help must work and say what the new options do ----------------------

def test_the_help_describes_the_new_options(monkeypatch, capsys) -> None:
    monkeypatch.setenv("COLUMNS", "1000")                           # no line wrapping
    with pytest.raises(SystemExit) as caught:
        ae.main(["--help"])
    assert caught.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert ("--prefix-epochs PREFIX_EPOCHS epochs of every run that must equal the earlier run point for point "
            "(default: 1000, the length of E4)") in text
    assert "--earlier-label EARLIER_LABEL what the earlier runs are called in the printed summary" in text
    assert ("--epochs-at-100 also count, per seed, the logged epochs with every entry recalled "
            "(whole run and late window)") in text
