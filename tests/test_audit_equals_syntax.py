"""The audit's "flag must be absent" rules match whole tokens, so --flag=value (which train.py accepts) evades them.

The runner always writes `--flag value`. A recorded command that contains `--flag=value` therefore did not come from the
runner, and it can change a setting (or repeat one: argparse keeps the last) without any rule noticing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phonebook.audit_experiment import _audit_command

BASE = ["python", "train.py", "cfg.py", "--device", "cpu", "--seed", "1", "--lr", "0.001", "--epochs", "10",
        "--batch-size", "64", "--accuracy-interval", "1", "--l2sp-lambda", "0.0",
        "--data-dir", "x/s/data_a0b1_evalA", "--out-dir", "x/s/c"]
RUN = {"id": "s/c", "lr": 0.001, "l2sp_lambda": 0.0, "epochs": 10, "batch_size": 64, "accuracy_interval": 1,
       "adapt": False, "mode": "a0b1_evalA", "scenario": "s", "condition": "c"}  # declares no override of any kind
PROTOCOL = {"spec": {"model_config": "cfg.py", "seed": 1}}


def audit(extra: list[str]) -> None:
    _audit_command(RUN, {"command": BASE + extra}, PROTOCOL, Path("."))


def test_a_clean_command_is_accepted() -> None:
    audit([])


@pytest.mark.parametrize("extra", [["--weight-decay", "0.0"], ["--lr-schedule", "step:2000:0.1"],
                                   ["--embedding-dim", "8"], ["--loader-rng", "legacy"]])
def test_an_undeclared_flag_in_space_syntax_is_refused(extra) -> None:
    with pytest.raises(ValueError, match="although the protocol declares no"):
        audit(extra)


@pytest.mark.parametrize("extra", [["--weight-decay=0.0"], ["--lr-schedule=step:2000:0.1"], ["--embedding-dim=8"],
                                   ["--n-heads=2"], ["--loader-rng=legacy"], ["--lr=0.5"]])
def test_equals_syntax_is_refused_because_the_runner_never_writes_it(extra) -> None:
    with pytest.raises(ValueError, match="--flag=value"):
        audit(extra)
