from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from phonebook import analyze_experiment as ax


def row(train_seq: float, val_seq: float, epoch: int = 1) -> dict:
    return {"epoch": epoch, "Train Seq Acc": train_seq, "Val Seq Acc": val_seq}


# ------------------------------------------------------------ counts_from_row


def test_plain_fine_tuning_reads_a_from_validation_and_b_from_training() -> None:
    counts = ax.counts_from_row(row(0.7, 0.2), a_copies=0, b_copies=1, eval_book="A", n_a=10, n_b=10)
    assert counts == (2, 7)


def test_balanced_replay_separates_the_two_books_from_the_mixture() -> None:
    # 20 training sequences: A once, B once. 10 correct in total, A alone has 6 correct -> B has 4.
    counts = ax.counts_from_row(row(0.5, 0.6), a_copies=1, b_copies=1, eval_book="A", n_a=10, n_b=10)
    assert counts == (6, 4)


def test_repeated_b_is_divided_back_out_of_the_mixture() -> None:
    # A once (10 seqs) + B three times (30 seqs) = 40. A=7 correct, B=5 correct -> 7 + 3*5 = 22 correct.
    counts = ax.counts_from_row(row(22 / 40, 0.7), a_copies=1, b_copies=3, eval_book="A", n_a=10, n_b=10)
    assert counts == (7, 5)


def test_phase_a_style_run_reads_b_from_validation_and_a_from_training() -> None:
    counts = ax.counts_from_row(row(1.0, 0.0), a_copies=1, b_copies=0, eval_book="B", n_a=10, n_b=10)
    assert counts == (10, 0)


def test_the_book_that_is_not_trained_on_is_unknown() -> None:
    assert ax.counts_from_row(row(0.3, 0.4), a_copies=0, b_copies=1, eval_book="A", n_a=10, n_b=10) == (4, 3)
    assert ax.counts_from_row(row(0.4, 0.4), a_copies=1, b_copies=0, eval_book="A", n_a=10, n_b=10) == (4, None)


def test_the_same_book_must_have_the_same_count_in_training_and_validation() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        ax.counts_from_row(row(0.3, 0.4), a_copies=1, b_copies=0, eval_book="A", n_a=10, n_b=10)
    with pytest.raises(ValueError, match="inconsistent"):
        ax.counts_from_row(row(0.3, 0.4), a_copies=0, b_copies=2, eval_book="B", n_a=10, n_b=10)


def test_four_decimal_rounding_in_the_log_is_tolerated() -> None:
    # 30 training sequences, 20 correct = 0.6666... which the log prints as 0.6667; A has 6 of 10, so B has 7 of 10.
    assert ax.counts_from_row(row(0.6667, 0.6), a_copies=1, b_copies=2, eval_book="A", n_a=10, n_b=10) == (6, 7)


@pytest.mark.parametrize("train,val", [(0.52, 0.6), (0.5, 0.65)])  # 10.4 of 20 and 6.5 of 10
def test_non_integer_counts_are_reported_not_rounded_away(train, val) -> None:
    with pytest.raises(ValueError, match="integer"):
        ax.counts_from_row(row(train, val), a_copies=1, b_copies=1, eval_book="A", n_a=10, n_b=10)


def test_counts_that_cannot_come_from_identical_copies_are_rejected() -> None:
    # A=6 correct, 2 copies of B means the B share (train_correct - 6) must be even.
    with pytest.raises(ValueError, match="copies"):
        ax.counts_from_row(row(7 / 30, 0.6), a_copies=1, b_copies=2, eval_book="A", n_a=10, n_b=10)


def test_impossible_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="range"):
        ax.counts_from_row(row(0.1, 0.6), a_copies=1, b_copies=1, eval_book="A", n_a=10, n_b=10)


# ------------------------------------------------------------- curve summaries


def points(a: list[int], b: list[int], start: int = 1) -> list[dict]:
    return [{"epoch": start + i, "A": x, "B": y} for i, (x, y) in enumerate(zip(a, b))]


def test_summary_reads_budget_endpoints_and_the_final_point() -> None:
    pts = points([10, 6, 2, 0, 0], [0, 3, 8, 10, 10])
    summary = ax.summarize_curve(pts, n_a=10, n_b=10, budgets=(2, 4))
    assert summary["at"] == {2: {"A": 6, "B": 3}, 4: {"A": 0, "B": 10}}
    assert summary["final"] == {"epoch": 5, "A": 0, "B": 10}


def test_summary_finds_first_complete_b_and_a_at_that_moment() -> None:
    pts = points([10, 6, 2, 1, 1], [0, 3, 8, 10, 10])
    summary = ax.summarize_curve(pts, n_a=10, n_b=10, budgets=())
    assert summary["first_b_complete"] == 4 and summary["a_at_first_b_complete"] == 1


def test_first_joint_and_stable_joint_differ_when_perfection_is_lost_and_regained() -> None:
    pts = points([10, 10, 9, 10, 10], [10, 10, 10, 10, 10])
    summary = ax.summarize_curve(pts, n_a=10, n_b=10, budgets=())
    assert summary["first_joint"] == 1 and summary["stable_joint"] == 4


def test_never_joint_means_neither_first_nor_stable() -> None:
    summary = ax.summarize_curve(points([10, 9, 10], [9, 10, 9]), n_a=10, n_b=10, budgets=())
    assert summary["first_joint"] is None and summary["stable_joint"] is None


def test_perfection_reached_but_not_held_to_the_end_has_a_first_but_no_stable_epoch() -> None:
    summary = ax.summarize_curve(points([10, 10, 9], [10, 10, 10]), n_a=10, n_b=10, budgets=())
    assert summary["first_joint"] == 1 and summary["stable_joint"] is None


def test_minimum_a_and_its_earliest_epoch() -> None:
    summary = ax.summarize_curve(points([10, 4, 2, 2, 7], [0, 5, 9, 10, 10]), n_a=10, n_b=10, budgets=())
    assert summary["min_a"] == 2 and summary["epoch_of_min_a"] == 3


def test_best_joint_sum_is_reported_with_its_earliest_epoch_as_exploratory() -> None:
    summary = ax.summarize_curve(points([10, 8, 7, 8], [0, 9, 10, 9]), n_a=10, n_b=10, budgets=())
    assert summary["best_joint_sum"] == 17 and summary["epoch_of_best_joint_sum"] == 2


def test_budget_without_a_logged_point_is_none() -> None:
    summary = ax.summarize_curve(points([10, 10], [10, 10]), n_a=10, n_b=10, budgets=(5,))
    assert summary["at"] == {5: None}


# ------------------------------------------------------------ lambda selection


def record(scenario: str, lam: float, a: int, b: int) -> dict:
    return {"scenario": scenario, "lambda": lam, "A": a, "B": b}


def test_lambda_with_the_largest_summed_recall_wins() -> None:
    records = [record("s1", 0.1, 2, 10), record("s2", 0.1, 3, 10),
               record("s1", 1.0, 8, 9), record("s2", 1.0, 7, 9)]
    assert ax.select_lambda(records) == 1.0


def test_ties_go_to_larger_summed_a_then_smaller_lambda() -> None:
    # Both lambdas sum to 19 per scenario, 1.0 has the larger A.
    records = [record("s1", 0.3, 4, 15 - 0), record("s1", 1.0, 9, 10)]
    assert ax.select_lambda(records) == 1.0
    equal = [record("s1", 3.0, 5, 5), record("s1", 0.3, 5, 5)]
    assert ax.select_lambda(equal) == 0.3


def test_selection_ignores_the_plain_and_replay_rows() -> None:
    records = [record("s1", 0.0, 10, 10), record("s1", 0.1, 1, 1)]
    assert ax.select_lambda(records) == 0.1


def test_selection_requires_every_lambda_to_cover_the_same_scenarios() -> None:
    with pytest.raises(ValueError, match="scenarios"):
        ax.select_lambda([record("s1", 0.1, 1, 1), record("s2", 0.1, 1, 1), record("s1", 1.0, 1, 1)])


# ------------------------------------------------------------ decision tables


def test_decision_requires_the_same_lambda_to_succeed_on_enough_scenarios() -> None:
    records = [record("s1", 1.0, 10, 10), record("s2", 1.0, 10, 10), record("s3", 1.0, 4, 10),
               record("s1", 3.0, 10, 10), record("s2", 3.0, 9, 10), record("s3", 3.0, 8, 10)]
    table = {row["lambda"]: row for row in ax.decision_table(records, n_a=10, n_b=10, min_scenarios=2)}
    assert table[1.0]["full_scenarios"] == 2 and table[1.0]["verdict"] == "matches replay"
    assert table[3.0]["full_scenarios"] == 1 and table[3.0]["near_scenarios"] == 2 and table[3.0]["verdict"] == "nearly matches"


def test_decision_says_no_when_no_lambda_gets_b_fully_learned_with_a_nearly_kept() -> None:
    records = [record("s1", 1.0, 10, 5), record("s2", 1.0, 3, 10)]
    assert ax.decision_table(records, n_a=10, n_b=10, min_scenarios=1)[0]["verdict"] == "no"


def test_overall_verdict_is_the_strongest_one_reached() -> None:
    rows = [{"verdict": "no"}, {"verdict": "nearly matches"}, {"verdict": "no"}]
    assert ax.overall_verdict(rows) == "nearly matches"
    assert ax.overall_verdict([{"verdict": "no"}]) == "no"
    assert ax.overall_verdict([{"verdict": "nearly matches"}, {"verdict": "matches replay"}]) == "matches replay"


# ------------------------------------------------------------------ file layer


def test_load_metrics_parses_numbers_and_missing_accuracy_markers(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Token Acc", "Val Token Acc", "Train Seq Acc",
                         "Val Seq Acc", "Training Time (sec)", "Param Count"])
        writer.writerow([10, "1.5000", "2.0000", "N/A", "N/A", "N/A", "N/A", "0.10", 110528])
        writer.writerow([20, "0.5000", "1.0000", "0.9000", "0.8000", "0.7000", "0.6000", "0.10", 110528])
    rows = ax.load_metrics(path)
    assert rows[0]["epoch"] == 10 and rows[0]["Train Seq Acc"] is None and rows[0]["Train Loss"] == 1.5
    assert rows[1]["Val Seq Acc"] == 0.6 and isinstance(rows[1]["epoch"], int)


def test_curve_skips_epochs_without_accuracies() -> None:
    rows = [{"epoch": 10, "Train Seq Acc": None, "Val Seq Acc": None},
            {"epoch": 20, "Train Seq Acc": 0.7, "Val Seq Acc": 0.2}]
    assert ax.curve(rows, a_copies=0, b_copies=1, eval_book="A", n_a=10, n_b=10) == [{"epoch": 20, "A": 2, "B": 7}]
