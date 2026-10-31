from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from phonebook import analyze_capacity as ac


def pts(*triples):
    """Logged curve points as (epoch, A = training entries recalled, B = held-out control entries recalled)."""
    return [{"epoch": e, "A": a, "B": b} for e, a, b in triples]


# ------------------------------------------------------------------ one run

def test_first_perfect_epoch_is_the_first_logged_epoch_with_every_entry_recalled() -> None:
    assert ac.first_perfect_epoch(pts((10, 3, 0), (20, 10, 0), (30, 10, 0)), 10) == 20


def test_a_run_that_never_recalls_everything_has_no_first_perfect_epoch() -> None:
    assert ac.first_perfect_epoch(pts((10, 3, 0), (20, 9, 0)), 10) is None


def test_a_later_drop_does_not_erase_the_first_perfect_epoch_but_is_flagged() -> None:
    summary = ac.cell_summary(pts((10, 10, 0), (20, 9, 0)), 10, budgets=(10, 20))
    assert summary["first_perfect"] == 10
    assert summary["perfect_at_end"] is False
    assert summary["perfect_by"] == {10: True, 20: True}


def test_perfect_by_a_budget_means_first_perfect_at_or_before_it() -> None:
    summary = ac.cell_summary(pts((100, 7, 0), (200, 10, 0), (300, 10, 0)), 10, budgets=(100, 200, 300))
    assert summary["perfect_by"] == {100: False, 200: True, 300: True}
    assert summary["recall_at"] == {100: 7, 200: 10, 300: 10}


def test_a_budget_that_was_never_logged_has_no_recall_and_never_counts_as_perfect() -> None:
    summary = ac.cell_summary(pts((100, 10, 0)), 10, budgets=(100, 150))
    assert summary["recall_at"][150] is None and summary["perfect_by"][150] is True  # perfect by an earlier epoch


def test_the_control_book_is_expected_to_stay_unrecalled() -> None:
    clean = ac.cell_summary(pts((10, 5, 0), (20, 10, 0)), 10, budgets=(20,))
    leaky = ac.cell_summary(pts((10, 5, 0), (20, 10, 2)), 10, budgets=(20,))
    assert clean["max_control"] == 0 and clean["control_ok"] is True
    assert leaky["max_control"] == 2 and leaky["control_ok"] is False


def test_final_recall_is_that_of_the_last_logged_epoch() -> None:
    assert ac.cell_summary(pts((10, 1, 0), (20, 4, 0)), 10, budgets=(20,))["final"] == 4


# ------------------------------------------------------------------ capacity rule

def cells(first_perfect_by_rung: dict[int, list]) -> list[dict]:
    return [{"n": n, "seed": seed, "first_perfect": value}
            for n, values in first_perfect_by_rung.items() for seed, value in enumerate(values)]


def test_capacity_is_the_largest_rung_where_two_of_three_seeds_are_perfect_within_the_budget() -> None:
    result = ac.capacity(cells({10: [20, 30, 40], 20: [50, 90, None], 40: [100, None, None], 80: [None] * 3}), budget=100)
    assert result["capacity"] == 20
    assert result["first_failing_rung"] == 40
    assert result["rungs"][20] == {"passed": 2, "seeds": 3, "ok": True}
    assert result["rungs"][40] == {"passed": 1, "seeds": 3, "ok": False}


def test_a_smaller_budget_can_only_lower_the_capacity() -> None:
    data = cells({10: [20, 30, 40], 20: [50, 90, None]})
    assert ac.capacity(data, budget=100)["capacity"] == 20
    assert ac.capacity(data, budget=60)["capacity"] == 10     # rung 20 has only one seed by epoch 60
    assert ac.capacity(data, budget=10)["capacity"] is None   # nothing is perfect by epoch 10


def test_capacity_needs_the_stricter_rule_when_asked() -> None:
    data = cells({10: [20, 30, 40], 20: [50, 90, None]})
    assert ac.capacity(data, budget=100, min_seeds=3)["capacity"] == 10


def test_capacity_stops_at_the_first_failing_rung_and_reports_lucky_passes_above_it() -> None:
    result = ac.capacity(cells({10: [10, 10, 10], 20: [None, None, 10], 40: [10, 10, 10]}), budget=100)
    assert result["capacity"] == 10
    assert result["first_failing_rung"] == 20
    assert result["isolated_passes_above"] == [40]


def test_when_even_the_smallest_rung_fails_there_is_no_capacity() -> None:
    result = ac.capacity(cells({10: [None, None, 5], 20: [None] * 3}), budget=100)
    assert result["capacity"] is None and result["first_failing_rung"] == 10


def test_capacity_is_the_top_rung_when_nothing_fails() -> None:
    result = ac.capacity(cells({10: [1, 1, 1], 20: [1, 1, 1]}), budget=100)
    assert result["capacity"] == 20 and result["first_failing_rung"] is None


def test_every_rung_must_have_the_same_seeds() -> None:
    data = cells({10: [1, 1, 1], 20: [1, 1]})
    with pytest.raises(ValueError, match="same seeds"):
        ac.capacity(data, budget=100)


def test_duplicate_runs_for_one_seed_and_rung_are_rejected() -> None:
    data = cells({10: [1, 1, 1]}) + [{"n": 10, "seed": 0, "first_perfect": 1}]
    with pytest.raises(ValueError, match="duplicate"):
        ac.capacity(data, budget=100)


def test_more_seeds_required_than_available_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_seeds"):
        ac.capacity(cells({10: [1, 1]}), budget=100, min_seeds=3)


# ------------------------------------------------------------------ boundary and bits

def test_a_boundary_that_has_not_moved_over_the_last_two_budgets_is_stable() -> None:
    assert ac.boundary_status({100: 20, 300: 40, 1000: 40}) == "stable"


def test_a_boundary_that_still_moves_is_reported_as_moving_not_as_a_capacity() -> None:
    assert ac.boundary_status({100: 20, 300: 20, 1000: 40}) == "moving"
    assert ac.boundary_status({300: None, 1000: 10}) == "moving"


def test_no_capacity_at_the_largest_budget_is_reported_as_none() -> None:
    assert ac.boundary_status({300: None, 1000: None}) == "none"


def test_boundary_status_needs_two_budgets() -> None:
    with pytest.raises(ValueError, match="two budgets"):
        ac.boundary_status({1000: 40})


def test_a_phone_number_carries_log2_of_the_generators_six_point_four_billion_numbers() -> None:
    assert ac.BITS_PER_ENTRY == pytest.approx(math.log2(6.4e9))
    assert round(ac.BITS_PER_ENTRY, 2) == 32.58


def test_bits_per_parameter_is_stored_information_over_parameters() -> None:
    assert ac.bits_per_param(100, 3_258) == pytest.approx(100 * ac.BITS_PER_ENTRY / 3_258)
    assert ac.bits_per_param(None, 3_258) is None


# ------------------------------------------------------------------ loading evidence

def fake_evidence(root: Path, *, runs: list[dict]) -> None:
    """The three file kinds load_capacity_cells reads: protocol.json, each run's metrics.csv and evaluation JSON."""
    (root).mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps({"runs": [r["protocol"] for r in runs]}), encoding="utf-8")
    for r in runs:
        folder = root / r["protocol"]["scenario"]
        (folder / r["protocol"]["condition"]).mkdir(parents=True)
        books = {"A": {"generation": {"total": r["n"], "correct": r["final_a"]}},
                 "B": {"generation": {"total": 20, "correct": 0}}}
        (folder / f"{r['protocol']['condition']}.evaluation.json").write_text(json.dumps({"books": books}), encoding="utf-8")
        with (folder / r["protocol"]["condition"] / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Token Acc", "Val Token Acc",
                             "Train Seq Acc", "Val Seq Acc", "Training Time (sec)", "Param Count"])
            for epoch, correct in r["curve"]:
                writer.writerow([epoch, "1.0", "5.0", "0.5", "0.1", f"{correct / r['n']:.4f}", "0.0000", "0.1", r["params"]])


def protocol_run(scenario: str, n: int, d: int = 16) -> dict:
    return {"id": f"{scenario}/c", "scenario": scenario, "condition": "c", "mode": "a1b0_evalB",
            "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B", "lr": 0.001, "epochs": 30,
            "batch_size": 64, "accuracy_interval": 10, "l2sp_lambda": 0.0, "adapt": False,
            "n_train": n, "updates_per_epoch": 1, "seq_len": 30,
            "model": {"embedding_dim": d, "n_heads": 2, "n_layers": 2}}


def test_loading_reads_the_model_the_seed_the_rung_the_parameters_and_the_first_perfect_epoch(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "root", runs=[
        {"protocol": protocol_run("s7_n20", 20), "n": 20, "final_a": 20, "params": 9216,
         "curve": [(10, 5), (20, 20), (30, 20)]},
        {"protocol": protocol_run("s7_n40", 40), "n": 40, "final_a": 12, "params": 9216,
         "curve": [(10, 4), (20, 8), (30, 12)]},
    ])
    loaded = ac.load_capacity_cells(tmp_path / "root", budgets=(10, 20, 30))
    by_n = {cell["n"]: cell for cell in loaded}
    assert by_n[20]["model_d"] == 16 and by_n[20]["seed"] == 7 and by_n[20]["params"] == 9216
    assert by_n[20]["first_perfect"] == 20 and by_n[40]["first_perfect"] is None
    assert by_n[40]["final"] == 12 and by_n[40]["perfect_at_end"] is False


def test_a_scenario_id_that_disagrees_with_the_book_size_is_rejected(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "root", runs=[
        {"protocol": protocol_run("s7_n20", 20), "n": 40, "final_a": 1, "params": 9216, "curve": [(10, 1), (20, 1), (30, 1)]}])
    with pytest.raises(ValueError, match="book size"):
        ac.load_capacity_cells(tmp_path / "root", budgets=(10,))


def test_a_scenario_id_outside_the_naming_scheme_is_rejected(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "root", runs=[
        {"protocol": protocol_run("weird", 20), "n": 20, "final_a": 1, "params": 9216, "curve": [(10, 1), (20, 1), (30, 1)]}])
    with pytest.raises(ValueError, match="scenario id"):
        ac.load_capacity_cells(tmp_path / "root", budgets=(10,))


def test_a_run_without_a_model_override_is_rejected(tmp_path: Path) -> None:
    run = protocol_run("s7_n20", 20)
    del run["model"]
    fake_evidence(tmp_path / "root", runs=[{"protocol": run, "n": 20, "final_a": 1, "params": 9216, "curve": [(10, 1), (20, 1), (30, 1)]}])
    with pytest.raises(ValueError, match="model"):
        ac.load_capacity_cells(tmp_path / "root", budgets=(10,))


def test_a_parameter_count_that_changes_within_a_run_is_rejected(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "root", runs=[
        {"protocol": protocol_run("s7_n20", 20), "n": 20, "final_a": 1, "params": 9216, "curve": [(10, 1), (20, 1), (30, 1)]}])
    metrics = tmp_path / "root" / "s7_n20" / "c" / "metrics.csv"
    metrics.write_text(metrics.read_text().replace("9216", "9217", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="parameter count"):
        ac.load_capacity_cells(tmp_path / "root", budgets=(10,))


# ------------------------------------------------------------------ review fixes

def test_a_ladder_that_never_fails_is_a_ceiling_not_a_capacity() -> None:
    assert ac.boundary_status({300: 40, 1000: 40}, top_rung_passes=True) == "ceiling"
    assert ac.boundary_status({300: 20, 1000: 40}, top_rung_passes=True) == "ceiling"
    assert ac.boundary_status({300: 40, 1000: 40}) == "stable"
    assert ac.boundary_status({300: None, 1000: None}, top_rung_passes=True) == "none"


def simple_cells(first_perfects_by_rung: dict) -> list[dict]:
    return [{"model_d": 16, "params": 9216, "n": n, "seed": seed, "first_perfect": value, "control_ok": True}
            for n, values in first_perfects_by_rung.items() for seed, value in enumerate(values)]


def test_the_summary_marks_a_model_whose_ladder_never_failed_as_a_lower_bound() -> None:
    never = ac.summarize_models(simple_cells({10: [1, 1, 1], 20: [1, 1, 1]}), (5, 10))[16]
    assert never["boundary"] == "ceiling" and never["capacity_is_lower_bound"] is True
    found = ac.summarize_models(simple_cells({10: [1, 1, 1], 20: [None] * 3}), (5, 10))[16]
    assert found["boundary"] == "stable" and found["capacity_is_lower_bound"] is False


def test_a_stricter_seed_rule_flows_through_the_summary() -> None:
    cells = simple_cells({10: [1, 1, None]})
    assert ac.summarize_models(cells, (5, 10))[16]["capacity_by_budget"] == {5: 10, 10: 10}
    assert ac.summarize_models(cells, (5, 10), min_seeds=3)[16]["capacity_by_budget"] == {5: None, 10: None}


def test_a_budget_beyond_the_trained_epochs_is_rejected(tmp_path: Path) -> None:
    fake_evidence(tmp_path / "root", runs=[
        {"protocol": protocol_run("s7_n20", 20), "n": 20, "final_a": 20, "params": 9216, "curve": [(10, 5), (20, 20), (30, 20)]}])
    with pytest.raises(ValueError, match="budget 60"):
        ac.load_capacity_cells(tmp_path / "root", budgets=(10, 60))


def three_seed_root(tmp_path: Path) -> Path:
    runs = []
    for seed in (1, 2, 3):
        for n, perfect in ((20, True), (40, False)):
            curve = [(10, n // 4), (20, n if perfect else n // 2), (30, n if perfect else n // 2)]
            runs.append({"protocol": protocol_run(f"s{seed}_n{n}", n), "n": n, "final_a": curve[-1][1], "params": 9216, "curve": curve})
    fake_evidence(tmp_path / "root", runs=runs)
    return tmp_path / "root"


def test_the_command_line_writes_cells_and_a_strict_capacity_report(tmp_path: Path) -> None:
    def refuse(constant):
        raise ValueError(f"non-strict JSON constant {constant}")

    root = three_seed_root(tmp_path)
    ac.main(["--root", str(root), "--budgets", "10", "20", "30", "--out", str(tmp_path / "out"), "--prefix", "e4"])
    with (tmp_path / "out" / "e4_cells.csv").open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 6
    report = json.loads((tmp_path / "out" / "e4_capacity.json").read_text(encoding="utf-8"), parse_constant=refuse)
    assert report["16"]["capacity_by_budget"] == {"10": None, "20": 20, "30": 20}
    assert report["16"]["boundary"] == "stable" and report["16"]["capacity_is_lower_bound"] is False


def test_the_command_line_refuses_an_empty_evidence_folder(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "protocol.json").write_text(json.dumps({"runs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no capacity runs"):
        ac.main(["--root", str(tmp_path / "empty"), "--budgets", "10", "20", "--out", str(tmp_path / "out")])


def test_the_command_line_seed_threshold_changes_which_rungs_pass(tmp_path: Path) -> None:
    runs = []
    for seed, perfect in ((1, True), (2, True), (3, False)):         # only two of the three seeds memorize the 20-entry book
        n = 20
        curve = [(10, n // 4), (20, n if perfect else n // 2), (30, n if perfect else n // 2)]
        runs.append({"protocol": protocol_run(f"s{seed}_n{n}", n), "n": n, "final_a": curve[-1][1], "params": 9216, "curve": curve})
    fake_evidence(tmp_path / "root", runs=runs)

    def capacity_at_30(min_seeds: int):
        out = tmp_path / f"out{min_seeds}"
        ac.main(["--root", str(tmp_path / "root"), "--budgets", "20", "30", "--out", str(out), "--min-seeds", str(min_seeds)])
        return json.loads((out / "e4_capacity.json").read_text(encoding="utf-8"))["16"]["capacity_by_budget"]["30"]

    assert capacity_at_30(2) == 20
    assert capacity_at_30(3) is None
