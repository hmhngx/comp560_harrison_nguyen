from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from phonebook import report_experiments as rp
from phonebook.analyze_experiment import summarize_curve


def make_run(scenario: str, condition: str, lam: float, a_b_by_epoch: dict[int, tuple[int, int]],
             a_copies: int = 0, b_copies: int = 1, strict: tuple[int, int] | None = None) -> dict:
    epochs = sorted(a_b_by_epoch)
    points = []
    last = (0, 0)
    for epoch in range(1, max(epochs) + 1):
        last = a_b_by_epoch.get(epoch, last)
        points.append({"epoch": epoch, "A": last[0], "B": last[1]})
    summary = summarize_curve(points, n_a=10, n_b=10, budgets=(100, 300))
    final = points[-1]
    return {"scenario": scenario, "condition": condition, "lr": 0.001, "epochs": max(epochs), "l2sp_lambda": lam,
            "train": {"a_copies": a_copies, "b_copies": b_copies}, "n_a": 10, "n_b": 10, "points": points,
            "summary": summary, "strict": {"A": (strict or (final["A"], final["B"]))[0], "B": (strict or (final["A"], final["B"]))[1]}}


def synthetic_runs() -> list[dict]:
    runs = []
    for scenario in ("d0_a", "d0_b", "d0_c"):
        runs.append(make_run(scenario, "plain", 0.0, {1: (10, 0), 20: (0, 10), 300: (0, 10)}))
        runs.append(make_run(scenario, "replay_a1b1", 0.0, {1: (10, 0), 30: (10, 10), 300: (10, 10)}, a_copies=1))
        for lam, outcome in ((0.1, (1, 10)), (1.0, (10, 10) if scenario != "d0_c" else (9, 10)), (10.0, (10, 0))):
            runs.append(make_run(scenario, f"l2sp_lam{lam:g}", lam, {1: (10, 0), 50: outcome, 300: outcome}))
    return runs


def test_endpoint_row_carries_budgets_final_strict_counts_and_curve_summary() -> None:
    run = make_run("d0_a", "plain", 0.0, {1: (10, 0), 20: (2, 10), 300: (0, 10)}, strict=(0, 10))
    row = rp.endpoint_row(run)
    assert (row["A_at_100"], row["B_at_100"], row["A_at_300"], row["B_at_300"]) == (2, 10, 0, 10)
    assert (row["A_final_strict"], row["B_final_strict"]) == (0, 10)
    assert row["first_b_complete"] == 20 and row["min_a"] == 0 and row["l2sp_lambda"] == 0.0
    assert set(row) == set(rp.ENDPOINT_COLUMNS)


def test_a_budget_that_was_not_logged_leaves_blank_cells() -> None:
    run = make_run("d0_a", "plain", 0.0, {1: (10, 0), 50: (3, 10)})
    row = rp.endpoint_row(run)
    assert row["A_at_100"] == "" and row["B_at_300"] == ""


def test_only_l2sp_runs_become_decision_records() -> None:
    records = rp.l2sp_records(synthetic_runs(), 100)
    assert {record["lambda"] for record in records} == {0.1, 1.0, 10.0}
    assert len(records) == 9


def test_decision_matches_replay_when_one_lambda_succeeds_on_two_of_three_checkpoints() -> None:
    decision = rp.e1_decision(synthetic_runs())
    by_lambda = {row["lambda"]: row for row in decision["budgets"]["100"]["by_lambda"]}
    assert by_lambda[1.0]["full_scenarios"] == 2 and by_lambda[1.0]["near_scenarios"] == 3
    assert decision["budgets"]["100"]["overall"] == "matches replay"
    assert by_lambda[10.0]["verdict"] == "no" and by_lambda[0.1]["verdict"] == "no"


def test_lambda_for_replication_is_chosen_by_the_pre_registered_rule() -> None:
    assert rp.e1_decision(synthetic_runs())["lambda_for_replication"] == 1.0


def test_write_csv_round_trips(tmp_path: Path) -> None:
    rows = [rp.endpoint_row(run) for run in synthetic_runs()]
    rp.write_csv(tmp_path / "x.csv", rows, rp.ENDPOINT_COLUMNS)
    read = list(csv.DictReader((tmp_path / "x.csv").open(newline="")))
    assert len(read) == len(rows) and list(read[0]) == rp.ENDPOINT_COLUMNS


def test_figures_are_written_for_the_standard_conditions(tmp_path: Path) -> None:
    runs = synthetic_runs()
    rp.figure_curves(runs, tmp_path / "curves.png", lambdas=(0.1, 1.0, 10.0))
    rp.figure_dose_response(runs, tmp_path / "dose.png")
    for name in ("curves.png", "dose.png"):
        data = (tmp_path / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000


# ------------------------------------------------------- cross-run consistency


HEADER = ["Epoch", "Train Loss", "Val Loss", "Train Token Acc", "Val Token Acc", "Train Seq Acc", "Val Seq Acc",
          "Training Time (sec)", "Param Count"]


def write_metrics(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


def row(epoch: int, loss: float = 1.0, seq: float = 0.5, seconds: float = 0.1) -> list:
    return [epoch, f"{loss:.4f}", "2.0000", "0.5000", "0.4000", f"{seq:.4f}", "0.3000", f"{seconds:.2f}", 110528]


def test_rows_at_shared_epochs_must_match_ignoring_only_the_timing_column(tmp_path: Path) -> None:
    write_metrics(tmp_path / "short.csv", [row(1, seconds=0.05), row(2, seconds=0.07), row(3, seconds=0.09)])
    write_metrics(tmp_path / "long.csv", [row(1, seconds=0.11), row(2, seconds=0.13), row(3, seconds=0.15), row(4)])
    result = rp.common_epoch_comparison(tmp_path / "short.csv", tmp_path / "long.csv", max_epoch=3)
    assert result == {"compared_epochs": 3, "mismatches": []}


def test_a_changed_value_at_a_shared_epoch_is_reported_with_its_column(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a.csv", [row(1), row(2, loss=1.5)])
    write_metrics(tmp_path / "b.csv", [row(1), row(2, loss=1.6)])
    result = rp.common_epoch_comparison(tmp_path / "a.csv", tmp_path / "b.csv", max_epoch=10)
    assert result["compared_epochs"] == 2
    assert result["mismatches"] == [{"epoch": 2, "column": "Train Loss", "a": "1.5000", "b": "1.6000"}]


def test_only_epochs_present_in_both_files_and_within_the_limit_are_compared(tmp_path: Path) -> None:
    write_metrics(tmp_path / "sparse.csv", [row(5), row(10), row(15)])
    write_metrics(tmp_path / "dense.csv", [row(e) for e in range(1, 13)])
    result = rp.common_epoch_comparison(tmp_path / "sparse.csv", tmp_path / "dense.csv", max_epoch=12)
    assert result["compared_epochs"] == 2  # epochs 5 and 10; 15 is beyond both the limit and the dense file


def test_comparing_files_with_no_shared_epoch_is_an_error_not_a_pass(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a.csv", [row(1)])
    write_metrics(tmp_path / "b.csv", [row(2)])
    with pytest.raises(ValueError, match="share"):
        rp.common_epoch_comparison(tmp_path / "a.csv", tmp_path / "b.csv", max_epoch=5)


def test_pairs_are_compared_across_two_evidence_roots(tmp_path: Path) -> None:
    for root, name in (("short", "plain_ep100"), ("long", "plain")):
        write_metrics(tmp_path / root / "d0_a" / name / "metrics.csv", [row(1), row(2)])
    report = rp.compare_runs(tmp_path / "short", tmp_path / "long", [("d0_a", "plain_ep100", "plain")], max_epoch=2)
    assert report == {"d0_a/plain_ep100 vs d0_a/plain": {"compared_epochs": 2, "mismatches": []}}


def test_a_lambda_that_succeeds_on_only_one_checkpoint_does_not_match_replay() -> None:
    runs = synthetic_runs()
    for run in runs:
        if run["condition"] == "l2sp_lam1" and run["scenario"] != "d0_a":
            run["summary"]["at"] = {100: {"A": 4, "B": 10}, 300: {"A": 4, "B": 10}}
    decision = rp.e1_decision(runs)
    by_lambda = {row["lambda"]: row for row in decision["budgets"]["100"]["by_lambda"]}
    assert by_lambda[1.0]["full_scenarios"] == 1 and by_lambda[1.0]["verdict"] == "no"
    assert decision["budgets"]["100"]["overall"] == "no"


def test_the_replication_lambda_is_chosen_at_100_updates_not_at_300() -> None:
    runs = []
    for scenario in ("d0_a", "d0_b", "d0_c"):
        runs.append(make_run(scenario, "plain", 0.0, {1: (10, 0), 300: (0, 10)}))
        runs.append(make_run(scenario, "l2sp_lam0.3", 0.3, {1: (10, 0), 100: (10, 10), 101: (0, 0), 300: (0, 0)}))
        runs.append(make_run(scenario, "l2sp_lam3", 3.0, {1: (10, 0), 100: (0, 0), 101: (10, 10), 300: (10, 10)}))
    assert rp.e1_decision(runs)["lambda_for_replication"] == 0.3


def run_report_cli(*args: str):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "phonebook" / "report_experiments.py"
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def test_compare_command_writes_a_json_report_and_succeeds_when_rows_agree(tmp_path: Path) -> None:
    for root, name in (("a", "plain_ep100"), ("b", "plain")):
        write_metrics(tmp_path / root / "d0_x" / name / "metrics.csv", [row(1), row(2), row(3)])
    out = tmp_path / "report.json"
    result = run_report_cli("compare", "--a", str(tmp_path / "a"), "--b", str(tmp_path / "b"),
                            "--pair", "d0_x:plain_ep100:plain", "--max-epoch", "3", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text()) == {"d0_x/plain_ep100 vs d0_x/plain": {"compared_epochs": 3, "mismatches": []}}


def test_compare_command_fails_loudly_on_any_mismatch(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a" / "d0_x" / "c" / "metrics.csv", [row(1), row(2, loss=1.5)])
    write_metrics(tmp_path / "b" / "d0_x" / "c" / "metrics.csv", [row(1), row(2, loss=1.9)])
    out = tmp_path / "report.json"
    result = run_report_cli("compare", "--a", str(tmp_path / "a"), "--b", str(tmp_path / "b"),
                            "--pair", "d0_x:c:c", "--max-epoch", "5", "--out", str(out))
    assert result.returncode == 1
    assert "MISMATCH" in result.stdout and out.exists()
    assert json.loads(out.read_text())["d0_x/c vs d0_x/c"]["mismatches"][0]["column"] == "Train Loss"


# ------------------------------------------------- E2 replication verdict (pre-registered)


def grid_runs(scenario: str, *, replay=(10, 10), plain=(0, 10), lam_outcomes=None) -> list[dict]:
    """A synthetic E1-style grid for one phonebook: plain, replay and a few L2-SP lambdas at 100 and 300 updates."""
    runs = [make_run(scenario, "plain", 0.0, {1: (10, 0), 20: plain, 300: plain}),
            make_run(scenario, "replay_a1b1", 0.0, {1: (10, 0), 30: replay, 300: replay}, a_copies=1)]
    for lam, outcome in (lam_outcomes or {0.1: (1, 10), 1.0: (9, 0), 10.0: (10, 0)}).items():
        runs.append(make_run(scenario, f"l2sp_lam{lam:g}", lam, {1: (10, 0), 50: outcome, 300: outcome}))
    return runs


def reference_decision() -> dict:
    return rp.e1_decision(grid_runs("d0_a") + grid_runs("d0_b") + grid_runs("d0_c"))


def test_replication_holds_when_the_three_pre_registered_criteria_hold_on_two_of_three_phonebooks() -> None:
    runs = grid_runs("ds2") + grid_runs("ds4") + grid_runs("ds7")
    verdict = rp.e2_replication(runs, reference_decision())
    assert verdict["replay_full_recall"]["holds"] and verdict["plain_forgets_a_and_learns_b"]["holds"]
    assert verdict["l2sp_verdict"]["holds"] and verdict["replicates"] is True


def test_one_failing_phonebook_is_tolerated_but_two_are_not() -> None:
    bad_replay = lambda s: grid_runs(s, replay=(10, 4))
    one_bad = grid_runs("ds2") + grid_runs("ds4") + bad_replay("ds7")
    assert rp.e2_replication(one_bad, reference_decision())["replay_full_recall"]["holds"] is True
    two_bad = bad_replay("ds2") + bad_replay("ds4") + grid_runs("ds7")
    verdict = rp.e2_replication(two_bad, reference_decision())
    assert verdict["replay_full_recall"] == {"scenarios": ["ds7"], "holds": False}
    assert verdict["replicates"] is False


def test_plain_fine_tuning_must_forget_a_by_at_most_two_entries_retained() -> None:
    keeps_three = lambda s: grid_runs(s, plain=(3, 10))
    verdict = rp.e2_replication(keeps_three("ds2") + keeps_three("ds4") + keeps_three("ds7"), reference_decision())
    assert verdict["plain_forgets_a_and_learns_b"] == {"scenarios": [], "holds": False}


def test_l2sp_verdict_must_equal_the_reference_verdict_at_both_budgets() -> None:
    matching = {0.1: (1, 10), 1.0: (10, 10), 10.0: (10, 0)}  # a lambda that reaches joint recall on every phonebook
    runs = grid_runs("ds2", lam_outcomes=matching) + grid_runs("ds4", lam_outcomes=matching) + grid_runs("ds7", lam_outcomes=matching)
    verdict = rp.e2_replication(runs, reference_decision())  # reference verdict is "no"
    assert verdict["l2sp_verdict"]["e2"] == {"100": "matches replay", "300": "matches replay"}
    assert verdict["l2sp_verdict"]["e1"] == {"100": "no", "300": "no"} and verdict["l2sp_verdict"]["holds"] is False


# --------------------------------------- E1b: union of the E1 and E1b grids, long horizon


def make_long_run(scenario: str, lam: float, at: dict[int, tuple[int, int]]) -> dict:
    points = []
    last = (0, 0)
    for epoch in range(5, 1001, 5):
        last = at.get(epoch, last)
        points.append({"epoch": epoch, "A": last[0], "B": last[1]})
    run = make_run(scenario, f"l2sp_lam{lam:g}", lam, {1: (0, 0)})
    run["points"] = points
    run["epochs"] = 1000
    run["summary"] = summarize_curve(points, n_a=10, n_b=10, budgets=(100, 300, 1000))
    return run


def test_union_records_keep_one_record_per_scenario_and_lambda() -> None:
    e1 = [make_run("d0_a", "l2sp_lam0.3", 0.3, {1: (3, 4), 300: (3, 8)}), make_run("d0_a", "l2sp_lam3", 3.0, {1: (10, 0), 300: (10, 0)})]
    e1b = [make_long_run("d0_a", 0.3, {5: (3, 4), 300: (3, 8)}), make_long_run("d0_a", 0.5, {5: (5, 2), 300: (5, 6), 1000: (5, 8)})]
    records = rp.union_l2sp_records(e1, e1b, 300)
    assert sorted((r["scenario"], r["lambda"]) for r in records) == [("d0_a", 0.3), ("d0_a", 0.5), ("d0_a", 3.0)]
    assert [r for r in rp.union_l2sp_records(e1, e1b, 1000)] == [
        {"scenario": "d0_a", "lambda": 0.3, "A": 3, "B": 8}, {"scenario": "d0_a", "lambda": 0.5, "A": 5, "B": 8}]


def test_a_union_lambda_that_matches_replay_at_1000_updates_is_reported() -> None:
    e1 = [make_run(s, "l2sp_lam3", 3.0, {1: (10, 0), 300: (10, 0)}) for s in ("d0_a", "d0_b", "d0_c")]
    e1b = [make_long_run(s, 0.7, {5: (10, 0), 600: (10, 10)}) for s in ("d0_a", "d0_b", "d0_c")]
    decision = rp.e1b_decision(e1, e1b)
    assert decision["budgets"]["1000"]["overall"] == "matches replay"
    assert decision["budgets"]["300"]["overall"] == "no"


# ------------------------------------------------------ E3: replay ratio cells


def e3_run(scenario: str, lr: float, k: int, stable_at: int | None, dip: int = 8) -> dict:
    steps = {1: (10, 0), 2: (dip, 0)}
    if stable_at is not None:
        steps[stable_at] = (10, 10)
    run = make_run(scenario, f"k{k}_lr{lr:g}", 0.0, steps, a_copies=1, b_copies=k)
    run["lr"] = lr
    run["epochs"] = max(steps)
    if stable_at is None:
        points = [{"epoch": e, "A": 10 if e == 1 else dip, "B": 0} for e in range(1, 501)]
        run["points"], run["summary"] = points, summarize_curve(points, n_a=10, n_b=10, budgets=(100,))
    else:
        points = [{"epoch": e, "A": (10, 0)[0] if e < 2 else (dip if e < stable_at else 10), "B": 0 if e < stable_at else 10} for e in range(1, 501)]
        run["points"], run["summary"] = points, summarize_curve(points, n_a=10, n_b=10, budgets=(100,))
    return run


def test_cells_carry_k_lr_and_the_curve_statistics() -> None:
    cell = rp.e3_cells([e3_run("d0_a", 0.001, 3, 40, dip=6)])[0]
    assert (cell["scenario"], cell["lr"], cell["k"]) == ("d0_a", 0.001, 3)
    assert cell["first_joint"] == 40 and cell["stable_joint"] == 40 and cell["min_a"] == 6 and cell["epoch_of_min_a"] == 2


def test_median_is_taken_over_checkpoints_and_censoring_counts_as_infinitely_slow() -> None:
    assert rp.median_or_none([50, 60, 70]) == 60
    assert rp.median_or_none([50, 60, None]) == 60
    assert rp.median_or_none([50, None, None]) is None
    assert rp.median_or_none([None, None, None]) is None


def test_best_k_is_the_smallest_median_and_ties_go_to_the_smaller_k() -> None:
    runs = []
    for scenario, (k1, k3, k5) in (("d0_a", (60, 40, 40)), ("d0_b", (70, 45, 45)), ("d0_c", (80, 50, 50))):
        runs += [e3_run(scenario, 0.001, 1, k1), e3_run(scenario, 0.001, 3, k3), e3_run(scenario, 0.001, 5, k5)]
    best = rp.best_k_per_lr(rp.e3_cells(runs))
    assert best == {0.001: 3}


def test_no_best_k_when_every_k_is_censored_for_that_learning_rate() -> None:
    runs = [e3_run(s, 0.0001, k, None) for s in ("d0_a", "d0_b", "d0_c") for k in (1, 3)]
    assert rp.best_k_per_lr(rp.e3_cells(runs)) == {0.0001: None}


def test_the_e3_table_has_one_row_per_lr_and_k_with_medians_and_censored_counts() -> None:
    runs = [e3_run("d0_a", 0.001, 1, 50), e3_run("d0_b", 0.001, 1, 60), e3_run("d0_c", 0.001, 1, None)]
    row = rp.e3_table(rp.e3_cells(runs))[0]
    assert (row["lr"], row["k"], row["median_stable_joint"], row["censored"], row["checkpoints"]) == (0.001, 1, 60, 1, 3)


# ------------------------------------------------- drivers on synthetic evidence folders


def write_evidence(root: Path, runs: list[dict]) -> None:
    """Write protocol.json, strict evaluation JSON and metrics.csv in the layout run_experiment.py produces."""
    root.mkdir(parents=True, exist_ok=True)
    protocol_runs = []
    for run in runs:
        protocol_runs.append({"id": f"{run['scenario']}/{run['condition']}", "scenario": run["scenario"],
                              "condition": run["condition"], "lr": run["lr"], "epochs": run["epochs"],
                              "l2sp_lambda": run["l2sp_lambda"], "train": run["train"], "eval_book": "A"})
        copies = run["train"]
        rows = []
        for point in run["points"]:
            train = (copies["a_copies"] * point["A"] + copies["b_copies"] * point["B"]) / (10 * (copies["a_copies"] + copies["b_copies"]))
            rows.append([point["epoch"], "1.0000", "1.0000", "0.5000", "0.5000", f"{train:.4f}", f"{point['A'] / 10:.4f}", "0.10", 110528])
        write_metrics(root / run["scenario"] / run["condition"] / "metrics.csv", rows)
        final = run["points"][-1]
        evaluation = {"books": {"A": {"generation": {"total": 10, "correct": final["A"]}},
                                "B": {"generation": {"total": 10, "correct": final["B"]}}}}
        (root / run["scenario"] / f"{run['condition']}.evaluation.json").write_text(json.dumps(evaluation))
    (root / "protocol.json").write_text(json.dumps({"runs": protocol_runs}))


def lam_outcome(lam: float, epoch: int) -> tuple[int, int]:
    a = 10 if epoch < 3 else min(10, round(lam * 4))
    b = 0 if epoch < 20 else max(0, 10 - round(lam * 8))
    return a, b


def evidence_run(scenario: str, condition: str, lam: float, *, epochs: int, step: int = 1, replay: bool = False,
                 b_copies: int = 1, lr: float = 0.001, curve=None) -> dict:
    points = []
    for epoch in range(step, epochs + 1, step):
        if curve:
            a, b = curve(epoch)
        elif replay:
            a, b = (10, 10) if epoch >= 30 else (10, 0)
        else:
            a, b = lam_outcome(lam, epoch)
        points.append({"epoch": epoch, "A": a, "B": b})
    return {"scenario": scenario, "condition": condition, "lr": lr, "epochs": epochs, "l2sp_lambda": lam,
            "train": {"a_copies": 1 if replay else 0, "b_copies": b_copies}, "points": points}


def e1_evidence(scenarios: tuple[str, ...]) -> list[dict]:
    runs = []
    for scenario in scenarios:
        runs.append(evidence_run(scenario, "plain", 0.0, epochs=300))
        runs.append(evidence_run(scenario, "replay_a1b1", 0.0, epochs=300, replay=True))
        runs += [evidence_run(scenario, f"l2sp_lam{lam:g}", lam, epochs=300) for lam in (0.3, 1.0, 3.0)]
    return runs


def png_ok(path: Path) -> bool:
    data = path.read_bytes()
    return data[:4] == b"\x89PNG" and len(data) > 5000


def test_report_grid_writes_the_endpoint_table_decision_and_figures(tmp_path: Path) -> None:
    write_evidence(tmp_path / "e1", e1_evidence(("d0_a", "d0_b", "d0_c")))
    decision, runs = rp.report_grid(tmp_path / "e1", tmp_path / "out", "t")
    rows = list(csv.DictReader((tmp_path / "out" / "t_endpoints.csv").open(newline="")))
    assert len(rows) == 15 and len(runs) == 15
    plain = next(row for row in rows if row["condition"] == "plain")
    assert (plain["A_at_100"], plain["B_at_100"]) == ("0", "10")
    assert json.loads((tmp_path / "out" / "t_decision.json").read_text()) == decision
    assert png_ok(tmp_path / "out" / "t_curves.png") and png_ok(tmp_path / "out" / "t_dose_response.png")


def test_report_e2_writes_the_replication_verdict(tmp_path: Path) -> None:
    write_evidence(tmp_path / "e1", e1_evidence(("d0_a", "d0_b", "d0_c")))
    reference = tmp_path / "e1_decision.json"
    reference.write_text(json.dumps(rp.e1_decision(rp.load_runs(tmp_path / "e1"))))
    write_evidence(tmp_path / "e2", e1_evidence(("ds2", "ds4", "ds7")))
    verdict = rp.report_e2(tmp_path / "e2", tmp_path / "out", reference)
    assert verdict["replicates"] is True
    assert json.loads((tmp_path / "out" / "e2_replication.json").read_text()) == verdict
    assert png_ok(tmp_path / "out" / "e2_curves.png")


def test_report_e1b_uses_the_union_grid_and_checks_the_overlap_with_e1(tmp_path: Path) -> None:
    scenarios = ("d0_a", "d0_b", "d0_c")
    write_evidence(tmp_path / "e1", e1_evidence(scenarios))
    write_evidence(tmp_path / "e1b", [evidence_run(s, f"l2sp_lam{lam:g}", lam, epochs=1000, step=5)
                                      for s in scenarios for lam in (0.3, 0.5, 1.0)])
    decision = rp.report_e1b(tmp_path / "e1", tmp_path / "e1b", tmp_path / "out")
    assert set(decision["budgets"]) == {"100", "300", "1000"}
    cross = json.loads((tmp_path / "out" / "e1b_vs_e1.json").read_text())
    assert len(cross) == 6 and all(not result["mismatches"] and result["compared_epochs"] > 0 for result in cross.values())
    header = next(csv.reader((tmp_path / "out" / "e1b_endpoints.csv").open(newline="")))
    assert "A_at_1000" in header and png_ok(tmp_path / "out" / "e1b_fine_grid.png")


def e3_evidence(scenarios: tuple[str, ...], lrs=(0.001, 0.0001), ks=(1, 3)) -> list[dict]:
    runs = []
    for scenario in scenarios:
        for lr in lrs:
            for k in ks:
                stable = max(10, round(40 / k * (0.001 / lr)))
                curve = lambda epoch, stable=stable: (10, 10) if epoch >= stable else (8, 0) if epoch >= 2 else (10, 0)
                runs.append(evidence_run(scenario, f"k{k}_lr{lr}", 0.0, epochs=500, replay=True, b_copies=k, lr=lr, curve=curve))
    return runs


def e1_evidence_matching_e3(scenarios: tuple[str, ...]) -> list[dict]:
    """E1 evidence whose replay curve equals E3's k=1, lr=0.001 curve, as the real experiments must."""
    runs = e1_evidence(scenarios)
    e3 = e3_evidence(scenarios)
    for run in runs:
        if run["condition"] == "replay_a1b1":
            twin = next(r for r in e3 if r["scenario"] == run["scenario"] and r["condition"] == "k1_lr0.001")
            run["points"] = [dict(point) for point in twin["points"][:300]]
    return runs


def fake_last_week(experiments: Path, b_counts: dict[str, int]) -> None:
    base = experiments / "week_2026-09-30"
    for name, folder in (("d0_seed42", "run_2026-10-04"), ("d0_seed123", "replication_phaseA_seed123"),
                         ("d0_seed999", "replication_phaseA_seed999")):
        (base / folder).mkdir(parents=True, exist_ok=True)
        (base / folder / "summary.csv").write_text(
            "condition,B_generation_correct\nreplay_lr0.0001_ep100," + str(b_counts[name]) + "\nreplay_lr0.001_ep100,10\n")


SEEDS = ("d0_seed42", "d0_seed123", "d0_seed999")


def test_report_e3_writes_cells_best_k_heatmaps_and_both_cross_checks(tmp_path: Path) -> None:
    write_evidence(tmp_path / "e3", e3_evidence(SEEDS))
    write_evidence(tmp_path / "e1", e1_evidence_matching_e3(SEEDS))
    fake_last_week(tmp_path / "experiments", {"d0_seed42": 0, "d0_seed123": 0, "d0_seed999": 0})
    result = rp.report_e3(tmp_path / "e3", tmp_path / "e1", tmp_path / "out", tmp_path / "experiments")
    assert result["best_k_per_lr"] == {"0.001": 3, "0.0001": 3}
    checks = result["cross_checks"]
    assert all(not r["mismatches"] for r in checks["k1_lr0.001_vs_e1_replay"].values())
    assert checks["k1_lr0.0001_b_at_100_updates"]["match"] is True
    table = list(csv.DictReader((tmp_path / "out" / "e3_table.csv").open(newline="")))
    assert [(row["lr"], row["k"]) for row in table] == [("0.001", "1"), ("0.001", "3"), ("0.0001", "1"), ("0.0001", "3")]
    assert png_ok(tmp_path / "out" / "e3_heatmaps.png") and (tmp_path / "out" / "e3_cells.csv").exists()


def test_report_e3_flags_a_mismatch_with_last_weeks_results(tmp_path: Path) -> None:
    write_evidence(tmp_path / "e3", e3_evidence(SEEDS))
    write_evidence(tmp_path / "e1", e1_evidence_matching_e3(SEEDS))
    fake_last_week(tmp_path / "experiments", {"d0_seed42": 0, "d0_seed123": 1, "d0_seed999": 0})
    checks = rp.report_e3(tmp_path / "e3", tmp_path / "e1", tmp_path / "out", tmp_path / "experiments")["cross_checks"]
    assert checks["k1_lr0.0001_b_at_100_updates"]["match"] is False


def test_report_e3_flags_a_replay_curve_that_differs_from_e1(tmp_path: Path) -> None:
    write_evidence(tmp_path / "e3", e3_evidence(SEEDS))
    write_evidence(tmp_path / "e1", e1_evidence(SEEDS))  # its replay reaches 10/10 at update 30, E3's at update 40
    fake_last_week(tmp_path / "experiments", {"d0_seed42": 0, "d0_seed123": 0, "d0_seed999": 0})
    checks = rp.report_e3(tmp_path / "e3", tmp_path / "e1", tmp_path / "out", tmp_path / "experiments")["cross_checks"]
    assert any(r["mismatches"] for r in checks["k1_lr0.001_vs_e1_replay"].values())


def test_the_grid_command_runs_end_to_end_from_the_command_line(tmp_path: Path) -> None:
    write_evidence(tmp_path / "e1", e1_evidence(SEEDS))
    result = run_report_cli("grid", "--root", str(tmp_path / "e1"), "--out", str(tmp_path / "out"), "--prefix", "x")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "x_endpoints.csv").exists() and "budgets" in json.loads(result.stdout)
    assert run_report_cli("nonsense").returncode != 0


# ------------------------------------------------ tests added after mutation checking found gaps


def test_replay_that_only_reaches_full_recall_after_100_updates_fails_criterion_one() -> None:
    late = lambda s: grid_runs(s, replay=(10, 10))
    runs = []
    for scenario in ("ds2", "ds4", "ds7"):
        base = [r for r in late(scenario) if r["condition"] != "replay_a1b1"]
        runs += base + [make_run(scenario, "replay_a1b1", 0.0, {1: (10, 0), 150: (10, 10), 300: (10, 10)}, a_copies=1)]
    verdict = rp.e2_replication(runs, reference_decision())
    assert verdict["replay_full_recall"] == {"scenarios": [], "holds": False}


def test_union_records_never_contain_plain_or_replay_rows() -> None:
    e1 = [make_run("d0_a", "plain", 0.0, {1: (10, 0), 300: (0, 10)}),
          make_run("d0_a", "replay_a1b1", 0.0, {1: (10, 0), 300: (10, 10)}, a_copies=1),
          make_run("d0_a", "l2sp_lam3", 3.0, {1: (10, 0), 300: (10, 0)})]
    e1b = [make_long_run("d0_a", 0.5, {5: (5, 2), 1000: (5, 8)})]
    assert {r["lambda"] for r in rp.union_l2sp_records(e1, e1b, 300)} == {0.5, 3.0}
    assert {r["lambda"] for r in rp.union_l2sp_records(e1, e1b, 1000)} == {0.5}


def test_a_run_that_reached_joint_recall_and_lost_it_is_censored_for_stable_but_not_for_first() -> None:
    run = e3_run("d0_a", 0.001, 1, 20)
    run["points"] = [{"epoch": e, "A": 10, "B": 10} if 20 <= e <= 30 else {"epoch": e, "A": 8, "B": 4} for e in range(1, 501)]
    run["summary"] = summarize_curve(run["points"], n_a=10, n_b=10, budgets=(100,))
    row = rp.e3_table(rp.e3_cells([run]))[0]
    assert run["summary"]["first_joint"] == 20 and run["summary"]["stable_joint"] is None
    assert row["censored"] == 1 and row["median_first_joint"] == 20 and row["median_stable_joint"] is None


def test_report_e3_catches_a_replay_curve_that_departs_from_e1_late_in_training(tmp_path: Path) -> None:
    write_evidence(tmp_path / "e3", e3_evidence(SEEDS))
    e1_runs = e1_evidence_matching_e3(SEEDS)
    for run in e1_runs:
        if run["condition"] == "replay_a1b1":
            run["points"][199]["A"] = 9  # identical for 199 updates, then one entry differs
    write_evidence(tmp_path / "e1", e1_runs)
    fake_last_week(tmp_path / "experiments", {"d0_seed42": 0, "d0_seed123": 0, "d0_seed999": 0})
    checks = rp.report_e3(tmp_path / "e3", tmp_path / "e1", tmp_path / "out", tmp_path / "experiments")["cross_checks"]
    assert all(r["mismatches"] and r["mismatches"][0]["epoch"] == 200 for r in checks["k1_lr0.001_vs_e1_replay"].values())
