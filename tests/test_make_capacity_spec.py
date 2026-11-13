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


# ------------------------------------------------------------------ E7: weight-decay and learning-rate-schedule arms

CONDITION_KEYS = {"id", "train", "eval_book", "lr", "epochs", "batch_size", "accuracy_interval", "adapt", "model"}


def test_without_the_options_the_spec_is_exactly_what_it_was() -> None:
    plain = spec()
    assert plain == spec(weight_decay=None, lr_schedule=None)
    assert set(plain["conditions"][0]) == CONDITION_KEYS and "arm" not in plain["decision_rules"]


def test_a_weight_decay_option_writes_the_key_and_a_plain_statement_of_the_difference_from_e6() -> None:
    result = spec(weight_decay=0.0)
    (condition,) = result["conditions"]
    assert condition["weight_decay"] == 0.0 and "lr_schedule" not in condition
    arm = result["decision_rules"]["arm"]
    assert "in one setting: AdamW weight decay is 0.0 (E6: 0.1)" in arm
    assert "E6 run of the same cell" in arm and arm.endswith("The E6 run is the control.")
    assert "schedule" not in arm and "step size" not in arm


def test_a_schedule_option_writes_the_key_and_says_it_lowers_the_step_size_and_the_per_step_decay_together() -> None:
    result = spec(lr_schedule="step:50:0.1")
    (condition,) = result["conditions"]
    assert condition["lr_schedule"] == "step:50:0.1" and "weight_decay" not in condition
    arm = result["decision_rules"]["arm"]
    assert ("in one setting: the learning rate follows the step schedule step:50:0.1, "
            "which multiplies it by 0.1 after epoch 50 (E6: constant)") in arm
    assert "learning rate times weight decay" in arm and "the step size and the per-step decay together" in arm
    assert "not an effect of the step size alone" in arm
    assert "weight decay is" not in arm                 # it does not claim to change the weight decay


def test_several_milestones_are_all_named() -> None:
    arm = spec(lr_schedule="step:20,40:0.5")["decision_rules"]["arm"]
    assert "multiplies it by 0.5 after each of epochs 20, 40" in arm


def test_both_options_make_a_two_setting_arm_and_the_decay_warning_depends_on_the_decay_left() -> None:
    off = spec(weight_decay=0.0, lr_schedule="step:50:0.1")
    assert set(off["conditions"][0]) == CONDITION_KEYS | {"weight_decay", "lr_schedule"}
    arm = off["decision_rules"]["arm"]
    assert ("in two settings: AdamW weight decay is 0.0 (E6: 0.1), and the learning rate follows "
            "the step schedule step:50:0.1") in arm
    assert "per-step decay" not in arm                  # with the decay off, the schedule cannot change it
    some = spec(weight_decay=0.05, lr_schedule="step:50:0.1")["decision_rules"]["arm"]
    assert "weight decay is 0.05 (E6: 0.1)" in some and "the step size and the per-step decay together" in some


def test_the_new_keys_come_after_the_model_and_the_arm_before_the_predictions() -> None:
    result = spec(weight_decay=0.0, lr_schedule="step:50:0.1", predictions={"d16": "about 40 entries"})
    assert list(result["conditions"][0])[-3:] == ["model", "weight_decay", "lr_schedule"]
    assert list(result["decision_rules"])[-3:] == ["budgets", "arm", "predictions"]


@pytest.mark.parametrize("overrides,message", [
    ({"weight_decay": -0.1}, "weight_decay must be a finite number"),
    ({"weight_decay": float("nan")}, "weight_decay must be a finite number"),
    ({"weight_decay": True}, "weight_decay must be a finite number"),
    ({"lr_schedule": "cosine:50:0.1"}, "lr_schedule: "),
    ({"lr_schedule": "step:50:0.10"}, "lr_schedule: "),
    ({"lr_schedule": "step:100:0.1"}, "before the last epoch"),         # the spec has 100 epochs
])
def test_a_bad_override_is_refused_before_a_spec_exists(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        spec(**overrides)


def test_a_spec_with_both_overrides_passes_the_runners_validation() -> None:
    rx.validate_spec(spec(weight_decay=0.0, lr_schedule="step:50:0.1"))


def test_the_planner_carries_the_overrides_of_a_generated_spec_into_every_run(tmp_path: Path) -> None:
    (tmp_path / "phonebook" / "config").mkdir(parents=True)
    shutil.copy(ROOT / "phonebook" / "config" / "phonebook.py", tmp_path / "phonebook" / "config" / "phonebook.py")
    for seed in (1, 2, 3):
        mcb.write_capacity_inputs(tmp_path / "phonebook" / "inputs", seed, RUNGS)
    protocol = rx.plan(spec(weight_decay=0.0, lr_schedule="step:50:0.1"), repo_root=tmp_path,
                       provenance={"commit": "x", "dirty": False})
    assert len(protocol["runs"]) == 9
    assert all(run["weight_decay"] == 0.0 and run["lr_schedule"] == "step:50:0.1" for run in protocol["runs"])
    assert protocol["optimizer"]["per_run_keys"] == ["lr_schedule", "weight_decay"]


def test_the_command_line_passes_the_override_options_through_and_refuses_bad_ones(tmp_path: Path) -> None:
    base = [sys.executable, str(ROOT / "phonebook" / "make_capacity_spec.py"), "--d", "16", "--rungs", "10,20", "--seeds", "1,2,3",
            "--epochs", "100", "--budgets", "30,100", "--timeout", "600"]
    target = tmp_path / "arm.json"
    done = subprocess.run([*base, "--weight-decay", "0.0", "--lr-schedule", "step:50:0.1", "--out", str(target)],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    written = json.loads(target.read_text(encoding="utf-8"))
    (condition,) = written["conditions"]
    assert (condition["weight_decay"], condition["lr_schedule"]) == (0.0, "step:50:0.1") and "arm" in written["decision_rules"]
    rx.validate_spec(written)
    for args, message in ((["--weight-decay=-1"], "weight_decay"), (["--lr-schedule=step:100:0.1"], "before the last epoch"),
                          (["--lr-schedule=step:50:0.10"], "lr_schedule")):
        bad = subprocess.run([*base, *args, "--out", str(tmp_path / "bad.json")], capture_output=True, text=True, check=False)
        assert bad.returncode != 0 and "error:" in bad.stderr and message in bad.stderr
        assert not (tmp_path / "bad.json").exists()


# ---- the 12 committed E6 specs are the output of this tool, byte for byte ------------------------------------------------
# The E6 README documents how the runs were started and analysed, not how the specs were generated, so the options below are
# the ones the committed files themselves imply (their name, seed, rung, epochs, budgets, timeout and "2 of its 3 book seeds").

E6_RUNG = {8: 20, 16: 160, 32: 320, 64: 640}


@pytest.mark.parametrize("d,seed", [(d, seed) for d in E6_RUNG for seed in (1, 2, 3)])
def test_the_committed_e6_specs_regenerate_byte_for_byte_from_their_command_lines(monkeypatch, capsys, tmp_path, d, seed) -> None:
    name = f"e6_extend_d{d}_seed{seed}"
    target = tmp_path / f"{name}.json"
    monkeypatch.setattr(sys, "argv", [
        "make_capacity_spec.py", "--d", str(d), "--rungs", str(E6_RUNG[d]), "--seeds", str(seed), "--experiment-seeds", "3",
        "--train-seed", str(seed), "--name", name, "--epochs", "4000", "--accuracy-interval", "10",
        "--budgets", "1000,2000,3000,4000", "--timeout", "28800",
        "--predictions", str(ROOT / "phonebook" / "preregistration" / "e6_predictions.json"), "--out", str(target)])
    mcs.main()
    assert capsys.readouterr().out.startswith(f"wrote {target}")
    assert target.read_bytes() == (ROOT / "phonebook" / "specs" / f"{name}.json").read_bytes()


# ---- added in E7.1a after the planted-bug harness: the sentence is locked into specs, so its exact text is pinned -----------

OPENING = ("This spec differs from the E6 run of the same cell (same book, seeds, model, base learning rate, batch size and "
           "epochs) in ")
COUPLING = (" AdamW decays every weight by learning rate times weight decay at each step, so the schedule lowers the step size "
            "and the per-step decay together; a difference from E6 is not an effect of the step size alone.")


@pytest.mark.parametrize("weight_decay,schedule,expected", [
    (0.0, None, OPENING + "one setting: AdamW weight decay is 0.0 (E6: 0.1). The E6 run is the control."),
    (None, "step:2000:0.1", OPENING + "one setting: the learning rate follows the step schedule step:2000:0.1, which multiplies it "
     "by 0.1 after epoch 2000 (E6: constant)." + COUPLING + " The E6 run is the control."),
    (0.0, "step:2000:0.1", OPENING + "two settings: AdamW weight decay is 0.0 (E6: 0.1), and the learning rate follows the step "
     "schedule step:2000:0.1, which multiplies it by 0.1 after epoch 2000 (E6: constant). The E6 run is the control."),
    (0.05, "step:20,40:0.5", OPENING + "two settings: AdamW weight decay is 0.05 (E6: 0.1), and the learning rate follows the step "
     "schedule step:20,40:0.5, which multiplies it by 0.5 after each of epochs 20, 40 (E6: constant)." + COUPLING
     + " The E6 run is the control."),
])
def test_the_arm_sentence_is_exactly_this_text(weight_decay, schedule, expected) -> None:
    assert mcs.arm_statement(weight_decay, schedule) == expected


# ---- added in E7.1a after the planted-bug harness: --help must work and say what the two options do ------------------------

def test_the_help_describes_the_two_options(monkeypatch, capsys) -> None:
    monkeypatch.setenv("COLUMNS", "1000")                           # no line wrapping
    monkeypatch.setattr(sys, "argv", ["make_capacity_spec.py", "--help"])
    with pytest.raises(SystemExit) as caught:
        mcs.main()
    assert caught.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "--weight-decay WEIGHT_DECAY AdamW weight decay of the condition (default: not written, the trainer's 0.1)" in text
    assert ("--lr-schedule LR_SCHEDULE learning-rate schedule of the condition, e.g. step:2000:0.1 "
            "(default: not written, constant)") in text
