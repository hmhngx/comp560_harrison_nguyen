"""The E6 analysis reads the trainer's logs of the extended capacity runs; each rule fixed before E6 is pinned here."""
from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path

import pytest

from phonebook import analyze_extension as ae


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
