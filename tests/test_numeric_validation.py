from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import train
from phonebook import run_experiment as rx
from phonebook import run_lr_sweep as sweep

ROOT = Path(__file__).resolve().parents[1]


def spec() -> dict:
    return rx.load_spec(ROOT / "phonebook/specs/e1_prefix_check.json")


@pytest.mark.parametrize("flag", ["--lr", "--l2sp-lambda"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e999"])
def test_nonfinite_training_cli_fails_before_touching_outputs(monkeypatch, tmp_path, flag, value):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "metrics.csv"
    sentinel.write_bytes(b"preserve existing evidence\n")
    monkeypatch.setattr(sys, "argv", ["train.py", f"{flag}={value}", "--out-dir", str(out)])
    with pytest.raises(ValueError, match=flag):
        train.main()
    assert list(out.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"preserve existing evidence\n"


@pytest.mark.parametrize("field", ["lr", "l2sp_lambda", "weight_decay", "grad_clip"])
@pytest.mark.parametrize("value", ["float('nan')", "float('inf')", "True", "'0.01'"])
def test_invalid_numeric_config_fails_before_creating_output(monkeypatch, tmp_path, field, value):
    config = tmp_path / "bad.py"
    config.write_text(f"{field} = {value}\n", encoding="utf-8")
    out = tmp_path / "new-output"
    monkeypatch.setattr(sys, "argv", ["train.py", str(config), "--out-dir", str(out)])
    with pytest.raises(ValueError, match=field.replace("_", "-")):
        train.main()
    assert not out.exists()


@pytest.mark.parametrize("field,value", [
    ("epochs", 0), ("batch_size", -1), ("log_interval", 0), ("accuracy_interval", True),
    ("embedding_dim", 3.5), ("n_heads", 0), ("n_layers", False),
    ("seed", True), ("seed", 1.5), ("seed", -(2**63) - 1), ("seed", 2**64),
    ("lr", 0), ("weight_decay", -0.1), ("grad_clip", -1),
])
def test_invalid_sizes_budgets_and_seed_are_rejected(monkeypatch, tmp_path, field, value):
    config = tmp_path / "bad.py"
    config.write_text(f"{field} = {value!r}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config)])
    with pytest.raises(ValueError, match=field.replace("_", "-")):
        train.parse_args()


def test_incompatible_head_dimensions_fail_during_config_validation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--embedding-dim=7", "--n-heads=2"])
    with pytest.raises(ValueError, match="divisible"):
        train.parse_args()


@pytest.mark.parametrize("seed", [-(2**63), -1, 0, 2**64 - 1])
def test_supported_seed_boundaries_and_zero_optional_penalties_remain_valid(monkeypatch, tmp_path, seed):
    config = tmp_path / "valid.py"
    config.write_text("weight_decay = 0\ngrad_clip = 0\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config), f"--seed={seed}"])
    cfg = train.parse_args()
    assert cfg.seed == seed and cfg.weight_decay == cfg.grad_clip == cfg.l2sp_lambda == 0


@pytest.mark.parametrize("field", ["lr", "l2sp_lambda"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.01", 10**400])
def test_spec_numeric_fields_reject_nonfinite_and_non_numeric_values(field, value):
    invalid = spec()
    invalid["conditions"][0][field] = value
    with pytest.raises(ValueError, match=field):
        rx.validate_spec(invalid)


@pytest.mark.parametrize("seed", [-(2**63) - 1, 2**64, True])
def test_spec_seed_must_be_supported_by_torch(seed):
    invalid = spec()
    invalid["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        rx.validate_spec(invalid)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_spec_json_rejects_nonstandard_constants_and_overflow_even_when_nested(tmp_path, literal):
    path = tmp_path / "bad.json"
    path.write_text('{"decision_rules": {"nested": [' + literal + ']}}', encoding="utf-8")
    with pytest.raises(ValueError):
        rx.load_spec(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_json_hashing_and_writing_reject_nonfinite_without_overwriting(tmp_path, value):
    obj = {"nested": [{"value": value}]}
    with pytest.raises(ValueError):
        rx.canonical_sha256(obj)
    path = tmp_path / "evidence.json"
    path.write_bytes(b"preserve me\n")
    with pytest.raises(ValueError):
        rx.write_json(path, obj)
    assert path.read_bytes() == b"preserve me\n"


def test_nonfinite_programmatic_metadata_fails_before_experiment_output_creation(tmp_path):
    invalid = spec()
    invalid["decision_rules"] = {"thresholds": [float("nan")]}
    out = tmp_path / "experiment"
    with pytest.raises(ValueError):
        rx.run_spec(invalid, out, repo_root=ROOT, provenance_fn=lambda: {"commit": "fixture", "dirty": False})
    assert not out.exists()


@pytest.mark.parametrize("args", [
    ["--lrs=nan"], ["--lrs=inf"], ["--lrs=-inf"], ["--lrs=0"],
    ["--l2sp-lambdas=nan"], ["--l2sp-lambdas=inf"], ["--l2sp-lambdas=-inf"],
    ["--epochs=0"], ["--timeout=0"], [f"--seeds={2**64}"],
])
def test_legacy_sweep_rejects_bad_inputs_before_creating_output(monkeypatch, tmp_path, args):
    out = tmp_path / "sweep"
    monkeypatch.setattr(sys, "argv", ["sweep", "--lrs=0.001", "--dry-run", "--out-root", str(out), *args])
    with pytest.raises(SystemExit) as exc:
        sweep.main()
    assert exc.value.code == 2
    assert not out.exists()


def test_finite_json_encoding_and_hashes_remain_compatible(tmp_path):
    obj = {"b": [0, 0.001, -1.25], "a": "entry"}
    path = tmp_path / "valid.json"
    rx.write_json(path, obj)
    assert path.read_bytes() == (json.dumps(obj, indent=2) + "\n").encode()
    assert rx.load_spec(path) == obj
