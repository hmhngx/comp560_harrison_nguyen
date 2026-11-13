"""E7 Tier 1: the trainer's two optimizer overrides, --weight-decay and --lr-schedule.

Every default run (neither flag) must stay exactly what it was: no new output line, the same recorded training_config, and a
learning rate that the trainer never assigns. With a flag the setting reaches the optimizer, is printed once and is recorded.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest
import torch

from completion_core.prep import build_char_vocab, encode_chars, save_outputs
import train

TINY_TEXT = "a=12\nb=21\nc=11\nd=22\ne=12\nf=21\ng=11\nh=22\n"
DEFAULT_CONFIG = {"seed": 123, "loader_rng": "isolated"}   # what every historical checkpoint records (test_logging_invariance)


@pytest.fixture
def tiny_data(tmp_path: Path) -> Path:
    stoi, itos = build_char_vocab(TINY_TEXT, "_")
    data = tmp_path / "data"
    save_outputs(data, encode_chars(TINY_TEXT, stoi), encode_chars(TINY_TEXT, stoi), stoi, itos)
    return data


def parse(monkeypatch, *args: str) -> train.TrainConfig:
    monkeypatch.setattr(sys, "argv", ["train.py", *args])
    return train.parse_args()


def run_training(monkeypatch, capsys, data: Path, out: Path, *extra: str, epochs: int = 10) -> dict:
    """Run train.main() in process on tiny data and keep, for every training epoch, the optimizer's rate and decay
    (as set at the start of the epoch) and the weights after it; also the printed text and the saved checkpoint."""
    epochs_seen = []
    original = train.run_epoch

    def record(**kwargs):
        optimizer = kwargs["optimizer"]
        entry = None
        if optimizer is not None:
            group = optimizer.param_groups[0]
            entry = {"lr": group["lr"], "weight_decay": group["weight_decay"]}
            epochs_seen.append(entry)
        loss = original(**kwargs)
        if entry is not None:
            entry["state"] = {name: value.detach().clone() for name, value in kwargs["model"].state_dict().items()}
        return loss

    with monkeypatch.context() as patch:
        patch.setattr(train, "run_epoch", record)
        patch.setattr(sys, "argv", [
            "train.py", "--device", "cpu", "--data-dir", str(data), "--out-dir", str(out),
            "--embedding-dim", "8", "--n-heads", "2", "--n-layers", "1", "--batch-size", "3",
            "--epochs", str(epochs), "--accuracy-interval", "5", "--seed", "123", "--lr", "0.01", *extra,
        ])
        train.main()
    stdout = capsys.readouterr().out
    payload = torch.load(out / "model.pth", map_location="cpu", weights_only=True)
    with (out / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = [{key: value for key, value in row.items() if key != "Training Time (sec)"}
                for row in csv.DictReader(file)]
    return {"epochs": epochs_seen, "stdout": stdout, "payload": payload, "metrics": rows}


def overrides_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("Optimizer overrides:")]


def same_weights(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


# ---- the premises of the E7 interpretation (E7 plan, section 1): they pin PyTorch, so they hold before any change ----------

def decayed(lr: float, weight_decay: float) -> float:
    """One AdamW step on a parameter of 1.0 whose gradient is exactly zero: only the decoupled decay acts."""
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.zeros_like(parameter)
    torch.optim.AdamW([parameter], lr=lr, weight_decay=weight_decay).step()
    return parameter.item()


def test_adamw_decays_a_parameter_by_lr_times_weight_decay_per_step() -> None:
    assert decayed(0.1, 0.5) == pytest.approx(0.95, abs=1e-7)       # 1 - 0.1 * 0.5
    assert decayed(0.01, 0.5) == pytest.approx(0.995, abs=1e-7)     # 1 - 0.01 * 0.5
    assert decayed(0.1, 0.0) == 1.0


def test_changing_the_learning_rate_between_steps_takes_effect_on_the_decay_too() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.5)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.param_groups[0]["lr"] = 0.01
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert parameter.item() == pytest.approx(0.94525, abs=1e-6)     # 0.95 * (1 - 0.01 * 0.5)


# ---- the flags reach the config, in both argparse spellings --------------------------------------------------------------

def test_the_defaults_are_the_historical_ones(monkeypatch) -> None:
    cfg = parse(monkeypatch)
    assert cfg.weight_decay == 0.1 and cfg.lr_schedule is None
    assert train.TrainConfig().weight_decay == 0.1 and train.TrainConfig().lr_schedule is None


@pytest.mark.parametrize("args", [["--weight-decay", "0.0"], ["--weight-decay=0.0"]])
def test_weight_decay_flag_reaches_the_config(monkeypatch, args) -> None:
    cfg = parse(monkeypatch, *args)
    assert cfg.weight_decay == 0.0 and isinstance(cfg.weight_decay, float)


def test_weight_decay_flag_takes_any_valid_value(monkeypatch) -> None:
    assert parse(monkeypatch, "--weight-decay", "0.05").weight_decay == 0.05
    assert parse(monkeypatch, "--weight-decay=1e-3").weight_decay == 0.001


@pytest.mark.parametrize("args", [
    ["--epochs", "4000", "--lr-schedule", "step:2000:0.1"],
    ["--epochs=4000", "--lr-schedule=step:2000:0.1"],
])
def test_lr_schedule_flag_reaches_the_config(monkeypatch, args) -> None:
    assert parse(monkeypatch, *args).lr_schedule == "step:2000:0.1"


def test_a_flag_beats_the_same_setting_in_a_config_file(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    config.write_text("weight_decay = 0.5\nepochs = 4000\nlr_schedule = 'step:1000:0.5'\n", encoding="utf-8")
    kept = parse(monkeypatch, str(config))
    assert (kept.weight_decay, kept.lr_schedule) == (0.5, "step:1000:0.5")
    flagged = parse(monkeypatch, str(config), "--weight-decay=0.0", "--lr-schedule=step:2000:0.1")
    assert (flagged.weight_decay, flagged.lr_schedule) == (0.0, "step:2000:0.1")


# ---- invalid values are refused before anything is written ---------------------------------------------------------------

@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e999", "-0.1", "-1e-9"])
def test_a_bad_weight_decay_is_rejected_naming_the_flag(monkeypatch, value) -> None:
    with pytest.raises(ValueError, match="--weight-decay"):
        parse(monkeypatch, f"--weight-decay={value}")


@pytest.mark.parametrize("value", ["nan", "inf", "1e999", "-0.1"])
def test_a_bad_weight_decay_is_rejected_when_separated_by_a_space_too(monkeypatch, value) -> None:
    with pytest.raises(ValueError, match="--weight-decay"):
        parse(monkeypatch, "--weight-decay", value)


@pytest.mark.parametrize("value", ["-inf", "-1e-9"])
def test_argparse_itself_refuses_a_space_separated_negative_that_looks_like_an_option(monkeypatch, value) -> None:
    with pytest.raises(SystemExit) as error:          # argparse reads "-inf" and "-1e-9" as options, so it never gets here
        parse(monkeypatch, "--weight-decay", value)
    assert error.value.code == 2


@pytest.mark.parametrize("schedule", [
    "cosine:2000:0.1",        # not a step schedule
    "step:2000",              # missing gamma
    "step:2000:0.10",         # a non-canonical spelling of a valid schedule
    "step:02000:0.1",
    "step:2000:0",            # gamma outside (0, 1]
    "step:2000:1.5",
    "step:2000:nan",
    "",                       # empty
])
def test_a_malformed_or_non_canonical_schedule_is_rejected_naming_the_flag(monkeypatch, schedule) -> None:
    with pytest.raises(ValueError, match="--lr-schedule"):
        parse(monkeypatch, "--epochs", "4000", f"--lr-schedule={schedule}")


@pytest.mark.parametrize("epochs,schedule", [
    (4000, "step:4000:0.1"),          # the milestone is the last epoch: it could never take effect
    (4000, "step:5000:0.1"),
    (4000, "step:1000,4000:0.1"),     # the last of several milestones is too late
    (100, "step:2000:0.1"),           # the default run length is 100 epochs
])
def test_a_schedule_whose_milestone_is_not_before_the_last_epoch_is_rejected(monkeypatch, epochs, schedule) -> None:
    with pytest.raises(ValueError, match="^--lr-schedule: every milestone must be before the last epoch$"):
        parse(monkeypatch, "--epochs", str(epochs), "--lr-schedule", schedule)


def test_the_last_milestone_may_be_the_second_to_last_epoch(monkeypatch) -> None:
    assert parse(monkeypatch, "--epochs", "4000", "--lr-schedule", "step:3999:0.1").lr_schedule == "step:3999:0.1"
    assert parse(monkeypatch, "--epochs", "101", "--lr-schedule", "step:100:0.1").lr_schedule == "step:100:0.1"


def test_a_non_string_schedule_from_a_config_file_is_rejected(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    config.write_text("epochs = 4000\nlr_schedule = 2000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="--lr-schedule.*string"):
        parse(monkeypatch, str(config))


@pytest.mark.parametrize("args", [
    ["--weight-decay=nan"], ["--weight-decay=-0.1"], ["--lr-schedule=cosine:2000:0.1"],
    ["--lr-schedule=step:2000:0.10", "--epochs=4000"], ["--lr-schedule=step:100:0.1"],
])
def test_an_invalid_override_fails_before_touching_the_output_folder(monkeypatch, tmp_path: Path, args) -> None:
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "metrics.csv"
    sentinel.write_bytes(b"preserve existing evidence\n")
    monkeypatch.setattr(sys, "argv", ["train.py", *args, "--out-dir", str(out)])
    with pytest.raises(ValueError, match="--weight-decay|--lr-schedule"):
        train.main()
    assert list(out.iterdir()) == [sentinel] and sentinel.read_bytes() == b"preserve existing evidence\n"


# ---- the default path is untouched ---------------------------------------------------------------------------------------

@pytest.mark.integration
def test_a_default_run_prints_no_override_line_and_records_the_historical_training_config(
    monkeypatch, capsys, tiny_data, tmp_path,
) -> None:
    run = run_training(monkeypatch, capsys, tiny_data, tmp_path / "default")
    assert overrides_lines(run["stdout"]) == []
    assert "Optimizer overrides" not in run["stdout"]
    assert run["payload"]["training_config"] == DEFAULT_CONFIG
    assert {entry["lr"] for entry in run["epochs"]} == {0.01} and {entry["weight_decay"] for entry in run["epochs"]} == {0.1}


class RateLock(dict):
    """A parameter group that fails the moment anybody assigns its learning rate."""

    def __setitem__(self, key, value):
        if key == "lr":
            raise AssertionError("the trainer assigned the learning rate")
        super().__setitem__(key, value)


@pytest.mark.integration
def test_without_a_schedule_the_trainer_never_assigns_the_learning_rate(monkeypatch, capsys, tiny_data, tmp_path) -> None:
    original = torch.optim.AdamW

    def locked(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        optimizer.param_groups[:] = [RateLock(group) for group in optimizer.param_groups]
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", locked)
    run_training(monkeypatch, capsys, tiny_data, tmp_path / "default")                      # must not raise
    run_training(monkeypatch, capsys, tiny_data, tmp_path / "decay", "--weight-decay=0.0")   # decay alone does not touch it
    with pytest.raises(AssertionError, match="assigned the learning rate"):                  # and the lock does catch a schedule
        run_training(monkeypatch, capsys, tiny_data, tmp_path / "step", "--lr-schedule=step:5:0.1")


@pytest.mark.integration
def test_an_explicit_weight_decay_equal_to_the_default_reproduces_the_default_run(
    monkeypatch, capsys, tiny_data, tmp_path,
) -> None:
    default = run_training(monkeypatch, capsys, tiny_data, tmp_path / "default")
    explicit = run_training(monkeypatch, capsys, tiny_data, tmp_path / "explicit", "--weight-decay", "0.1")
    assert explicit["metrics"] == default["metrics"]
    for epoch, (expected, actual) in enumerate(zip(default["epochs"], explicit["epochs"]), start=1):
        assert same_weights(expected["state"], actual["state"]), f"epoch {epoch}"
    assert same_weights(default["payload"]["state_dict"], explicit["payload"]["state_dict"])
    assert explicit["stdout"].replace("\r", "").count("\n") == default["stdout"].replace("\r", "").count("\n")
    assert explicit["payload"]["training_config"] == default["payload"]["training_config"] == DEFAULT_CONFIG


# ---- the overrides do what they say ------------------------------------------------------------------------------------

@pytest.mark.integration
def test_weight_decay_zero_reaches_the_optimizer_and_changes_the_trajectory(monkeypatch, capsys, tiny_data, tmp_path) -> None:
    default = run_training(monkeypatch, capsys, tiny_data, tmp_path / "default")
    off = run_training(monkeypatch, capsys, tiny_data, tmp_path / "off", "--weight-decay", "0.0")
    assert {entry["weight_decay"] for entry in off["epochs"]} == {0.0}
    assert {entry["weight_decay"] for entry in default["epochs"]} == {0.1}
    assert not same_weights(default["epochs"][0]["state"], off["epochs"][0]["state"])      # different from the first epoch on
    assert not same_weights(default["payload"]["state_dict"], off["payload"]["state_dict"])


@pytest.mark.integration
@pytest.mark.parametrize("schedule,epochs,expected", [
    ("step:3:0.1", 6, [0.01, 0.01, 0.01, 0.01 * 0.1, 0.01 * 0.1, 0.01 * 0.1]),
    ("step:2,4:0.5", 6, [0.01, 0.01, 0.01 * 0.5, 0.01 * 0.5, 0.01 * 0.5 ** 2, 0.01 * 0.5 ** 2]),
])
def test_the_schedule_sets_the_optimizers_learning_rate_for_each_epoch(
    monkeypatch, capsys, tiny_data, tmp_path, schedule, epochs, expected,
) -> None:
    run = run_training(monkeypatch, capsys, tiny_data, tmp_path / "scheduled", f"--lr-schedule={schedule}", epochs=epochs)
    assert [entry["lr"] for entry in run["epochs"]] == expected                           # the milestone epoch itself is not decayed
    assert {entry["weight_decay"] for entry in run["epochs"]} == {0.1}                    # a schedule leaves the decay setting alone


@pytest.mark.integration
def test_a_step_schedule_equals_the_unscheduled_run_through_its_milestone_and_differs_after_it(
    monkeypatch, capsys, tiny_data, tmp_path,
) -> None:
    plain = run_training(monkeypatch, capsys, tiny_data, tmp_path / "plain")
    stepped = run_training(monkeypatch, capsys, tiny_data, tmp_path / "step", "--lr-schedule", "step:5:0.1")
    for epoch in range(1, 6):
        assert same_weights(plain["epochs"][epoch - 1]["state"], stepped["epochs"][epoch - 1]["state"]), f"epoch {epoch}"
    assert not same_weights(plain["epochs"][5]["state"], stepped["epochs"][5]["state"])    # epoch 6 is the first decayed epoch
    assert not same_weights(plain["payload"]["state_dict"], stepped["payload"]["state_dict"])


@pytest.mark.integration
def test_the_logged_metrics_follow_the_same_prefix_rule(monkeypatch, capsys, tiny_data, tmp_path) -> None:
    plain = run_training(monkeypatch, capsys, tiny_data, tmp_path / "plain")
    stepped = run_training(monkeypatch, capsys, tiny_data, tmp_path / "step", "--lr-schedule=step:5:0.1")
    assert [row["Epoch"] for row in plain["metrics"]] == ["5", "10"]
    assert stepped["metrics"][0] == plain["metrics"][0]          # epoch 5 is the milestone epoch: still the full rate
    assert stepped["metrics"][1] != plain["metrics"][1]          # epoch 10 follows five decayed epochs


# ---- what is printed and recorded --------------------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("extra,recorded,printed", [
    (["--weight-decay", "0.0"], {"weight_decay": 0.0}, "Optimizer overrides: weight_decay=0.0"),
    (["--lr-schedule", "step:5:0.1"], {"lr_schedule": "step:5:0.1"}, "Optimizer overrides: lr_schedule=step:5:0.1"),
    (["--weight-decay=0.0", "--lr-schedule=step:5:0.1"], {"weight_decay": 0.0, "lr_schedule": "step:5:0.1"},
     "Optimizer overrides: weight_decay=0.0, lr_schedule=step:5:0.1"),
])
def test_an_override_is_printed_once_and_recorded_in_the_checkpoint(
    monkeypatch, capsys, tiny_data, tmp_path, extra, recorded, printed,
) -> None:
    run = run_training(monkeypatch, capsys, tiny_data, tmp_path / "overridden", *extra)
    assert overrides_lines(run["stdout"]) == [printed]
    assert run["payload"]["training_config"] == {**DEFAULT_CONFIG, **recorded}


@pytest.mark.integration
def test_an_override_from_a_config_file_is_recorded_like_one_from_a_flag(monkeypatch, capsys, tiny_data, tmp_path) -> None:
    config = tmp_path / "config.py"
    config.write_text("weight_decay = 0.0\nlr_schedule = 'step:5:0.1'\n", encoding="utf-8")
    run = run_training(monkeypatch, capsys, tiny_data, tmp_path / "from_config", str(config))
    assert run["payload"]["training_config"] == {**DEFAULT_CONFIG, "weight_decay": 0.0, "lr_schedule": "step:5:0.1"}
    assert {entry["weight_decay"] for entry in run["epochs"]} == {0.0}
    assert run["epochs"][5]["lr"] == 0.01 * 0.1


# ---- added in E7.1a after the planted-bug harness: --help must work and say what the two options do ------------------------

def test_the_help_describes_the_two_options(monkeypatch, capsys) -> None:
    monkeypatch.setenv("COLUMNS", "1000")                           # no line wrapping
    with pytest.raises(SystemExit) as caught:
        parse(monkeypatch, "--help")
    assert caught.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "--weight-decay WEIGHT_DECAY AdamW weight decay (default: the TrainConfig value, 0.1)" in text
    assert ("--lr-schedule LR_SCHEDULE step learning-rate schedule such as step:2000:0.1, which multiplies the rate by 0.1 "
            "after epoch 2000 (default: constant)") in text
