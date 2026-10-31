from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from phonebook import make_capacity_books as mcb
from phonebook import make_capacity_spec as mcs
from phonebook import run_experiment as rx

ROOT = Path(__file__).resolve().parents[1]
RUNGS = [10, 20, 40]
KWARGS = dict(epochs=100, accuracy_interval=10, budgets=[30, 100], timeout_seconds=600)


def spec(**overrides) -> dict:
    return mcs.capacity_spec(16, RUNGS, [1, 2, 3], **{**KWARGS, **overrides})


def test_the_generated_spec_passes_the_runners_validation() -> None:
    rx.validate_spec(spec())


def test_one_scenario_per_seed_and_rung_pointing_at_the_books_the_generator_writes() -> None:
    result = spec()
    assert [s["id"] for s in result["scenarios"]] == [f"s{seed}_n{n}" for seed in (1, 2, 3) for n in RUNGS]
    first = result["scenarios"][0]
    assert first["book_a"] == mcb.capacity_path(Path("phonebook/inputs"), 1, 10).as_posix()
    assert first["book_b"] == mcb.control_path(Path("phonebook/inputs"), 1).as_posix()


def test_the_single_condition_trains_from_scratch_on_book_a_and_validates_on_the_control() -> None:
    (condition,) = spec()["conditions"]
    assert condition["id"] == "d16"
    assert condition["train"] == {"a_copies": 1, "b_copies": 0} and condition["eval_book"] == "B"
    assert condition["adapt"] is False
    assert condition["model"] == {"embedding_dim": 16, "n_heads": 2, "n_layers": 2}
    assert (condition["lr"], condition["batch_size"], condition["epochs"], condition["accuracy_interval"]) == (0.001, 64, 100, 10)


def test_the_spec_allows_many_updates_per_epoch_and_evaluates_both_books() -> None:
    result = spec()
    assert result["require_one_update_per_epoch"] is False
    assert result["evaluate_books"] == ["A", "B"] and result["timeout_seconds"] == 600


def test_every_budget_is_a_logged_epoch_and_the_largest_budget_is_the_run_length() -> None:
    logged = rx.expected_logged_epochs(100, 10)
    assert all(budget in logged for budget in KWARGS["budgets"])


@pytest.mark.parametrize("overrides,message", [
    ({"budgets": [30, 95]}, "logged"),
    ({"budgets": [30, 90]}, "run length"),
    ({"budgets": []}, "budgets"),
    ({"accuracy_interval": 7}, "multiple"),
])
def test_budgets_that_cannot_be_read_back_from_the_log_are_rejected(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        spec(**overrides)


def test_the_rule_text_states_the_pre_registered_endpoints() -> None:
    rules = spec()["decision_rules"]
    for key in ("question", "success", "capacity", "boundary", "controls", "budgets"):
        assert key in rules and rules[key]
    assert "predictions" not in rules


def test_predictions_are_embedded_verbatim_when_given() -> None:
    result = spec(predictions={"d16": "about 40 entries"})
    assert result["decision_rules"]["predictions"] == {"d16": "about 40 entries"}


@pytest.mark.parametrize("rungs,seeds", [([20, 10], [1]), ([10, 10], [1]), ([], [1]), ([10], []), ([10], [1, 1])])
def test_rungs_must_increase_and_seeds_must_be_distinct(rungs, seeds) -> None:
    with pytest.raises(ValueError, match="rungs|seeds"):
        mcs.capacity_spec(16, rungs, seeds, **KWARGS)


def test_the_planner_accepts_the_spec_on_real_generated_books(tmp_path: Path) -> None:
    (tmp_path / "phonebook" / "config").mkdir(parents=True)
    shutil.copy(ROOT / "phonebook" / "config" / "phonebook.py", tmp_path / "phonebook" / "config" / "phonebook.py")
    for seed in (1, 2, 3):
        mcb.write_capacity_inputs(tmp_path / "phonebook" / "inputs", seed, RUNGS)
    protocol = rx.plan(spec(), repo_root=tmp_path, provenance={"commit": "x", "dirty": False})
    assert len(protocol["runs"]) == 9
    by_id = {run["id"]: run for run in protocol["runs"]}
    assert by_id["s2_n40/d16"]["n_train"] == 40 and by_id["s2_n40/d16"]["updates_per_epoch"] == 1
    assert all(run["model"] == {"embedding_dim": 16, "n_heads": 2, "n_layers": 2} and not run["adapt"] for run in protocol["runs"])


def test_a_rung_that_needs_many_updates_per_epoch_is_planned_as_such(tmp_path: Path) -> None:
    (tmp_path / "phonebook" / "config").mkdir(parents=True)
    shutil.copy(ROOT / "phonebook" / "config" / "phonebook.py", tmp_path / "phonebook" / "config" / "phonebook.py")
    mcb.write_capacity_inputs(tmp_path / "phonebook" / "inputs", 1, [10, 200])
    protocol = rx.plan(mcs.capacity_spec(16, [10, 200], [1], min_seeds=1, **KWARGS), repo_root=tmp_path,
                       provenance={"commit": "x", "dirty": False})
    assert {run["id"]: run["updates_per_epoch"] for run in protocol["runs"]} == {"s1_n10/d16": 1, "s1_n200/d16": 4}


def test_the_command_line_writes_the_spec_and_refuses_to_overwrite_it(tmp_path: Path) -> None:
    target = tmp_path / "e4_capacity_d16.json"
    command = [sys.executable, str(ROOT / "phonebook" / "make_capacity_spec.py"), "--d", "16", "--rungs", "10,20",
               "--seeds", "1,2", "--epochs", "100", "--budgets", "30,100", "--timeout", "600", "--out", str(target)]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["name"] == "e4_capacity_d16" and len(written["scenarios"]) == 4
    rx.validate_spec(written)
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode != 0 and "already exists" in again.stderr


# ------------------------------------------------------------------ review fixes

def test_the_rule_text_states_the_real_seed_count_and_threshold() -> None:
    assert "2 of its 3 book seeds" in spec()["decision_rules"]["capacity"]
    five = mcs.capacity_spec(16, RUNGS, [1, 2, 3, 4, 5], min_seeds=3, **KWARGS)
    assert "3 of its 5 book seeds" in five["decision_rules"]["capacity"] and "5 book seeds" in five["description"]


def test_a_per_seed_spec_describes_the_whole_experiments_rule_not_its_own_single_seed() -> None:
    one = mcs.capacity_spec(16, RUNGS, [7], experiment_seeds=3, train_seed=7, name="e4_capacity_d16_s7", **KWARGS)
    assert one["seed"] == 7 and one["name"] == "e4_capacity_d16_s7"
    assert "2 of its 3 book seeds" in one["decision_rules"]["capacity"]
    rx.validate_spec(one)


@pytest.mark.parametrize("overrides", [{"min_seeds": 0}, {"min_seeds": 4}, {"experiment_seeds": 2}])
def test_the_seed_threshold_must_fit_the_experiment(overrides) -> None:
    with pytest.raises(ValueError, match="min_seeds|experiment_seeds"):
        spec(**overrides)


def test_the_spec_says_what_the_replicates_vary() -> None:
    text = spec()["decision_rules"]["replicates"]
    assert "book" in text and "initialisation" in text


def test_the_command_line_passes_the_per_seed_options_through(tmp_path: Path) -> None:
    target = tmp_path / "s7.json"
    command = [sys.executable, str(ROOT / "phonebook" / "make_capacity_spec.py"), "--d", "16", "--rungs", "10,20", "--seeds", "7",
               "--experiment-seeds", "3", "--train-seed", "7", "--name", "e4_capacity_d16_s7", "--epochs", "100",
               "--budgets", "30,100", "--timeout", "600", "--out", str(target)]
    assert subprocess.run(command, capture_output=True, text=True, check=False).returncode == 0
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["seed"] == 7 and written["name"] == "e4_capacity_d16_s7"
