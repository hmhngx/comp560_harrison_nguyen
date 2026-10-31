"""The runner's two additive spec keys: a per-condition `model` override and a spec-level `loader_rng`."""
from __future__ import annotations

import copy
import csv
import json
import shutil
from pathlib import Path

import pytest

from completion_core.vocabulary import Vocabulary
from phonebook import audit_experiment as audit_mod
from phonebook import run_experiment as rx
from phonebook.param_count import expected_param_count

ROOT = Path(__file__).resolve().parents[1]


def clean() -> dict:
    return {"commit": "testcommit", "dirty": False}


def base_spec() -> dict:
    return {
        "name": "unit", "seed": 42, "model_config": "phonebook/config/phonebook.py",
        "scenarios": [{"id": "s1", "book_a": "a.txt", "book_b": "b.txt"}],
        "conditions": [{"id": "c", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B", "lr": 0.001,
                        "epochs": 5, "batch_size": 64, "accuracy_interval": 1, "adapt": False,
                        "model": {"embedding_dim": 16, "n_heads": 2, "n_layers": 2}}],
    }


# ------------------------------------------------------------------ spec validation

def test_a_spec_with_a_model_override_and_a_loader_policy_is_valid() -> None:
    spec = base_spec()
    spec["loader_rng"] = "legacy"
    rx.validate_spec(spec)


@pytest.mark.parametrize("model,message", [
    ({"embedding_dim": 16, "n_heads": 2}, "n_layers"),
    ({"embedding_dim": 16, "n_heads": 2, "n_layers": 2, "dropout": 0.1}, "dropout"),
    ({"embedding_dim": 0, "n_heads": 2, "n_layers": 2}, "embedding_dim"),
    ({"embedding_dim": 16.0, "n_heads": 2, "n_layers": 2}, "embedding_dim"),
    ({"embedding_dim": 16, "n_heads": True, "n_layers": 2}, "n_heads"),
    ({"embedding_dim": 16, "n_heads": 3, "n_layers": 2}, "divisible"),
    ("big", "model"),
])
def test_a_malformed_model_override_is_rejected(model, message) -> None:
    spec = base_spec()
    spec["conditions"][0]["model"] = model
    with pytest.raises(ValueError, match=message):
        rx.validate_spec(spec)


@pytest.mark.parametrize("policy", ["shared", "", None, 1])
def test_the_loader_policy_must_be_isolated_or_legacy(policy) -> None:
    spec = base_spec()
    spec["loader_rng"] = policy
    with pytest.raises(ValueError, match="loader_rng"):
        rx.validate_spec(spec)


def test_a_spec_without_either_key_expands_exactly_as_before() -> None:
    spec = base_spec()
    del spec["conditions"][0]["model"]
    (run,) = rx.expand_runs(spec)
    assert "model" not in run
    assert set(run) == {"id", "scenario", "condition", "mode", "train", "eval_book", "lr", "epochs",
                        "batch_size", "accuracy_interval", "l2sp_lambda", "adapt"}


def test_a_declared_model_travels_with_the_expanded_run() -> None:
    (run,) = rx.expand_runs(base_spec())
    assert run["model"] == {"embedding_dim": 16, "n_heads": 2, "n_layers": 2}


# ------------------------------------------------------------------ command construction

def command_for(spec: dict, *, loader_rng=None) -> list[str]:
    (run,) = rx.expand_runs(spec)
    return rx.training_command(run, config=spec["model_config"], data_dir=Path("d"), out_dir=Path("o"),
                               seed=spec["seed"], init_checkpoint=None, loader_rng=loader_rng)


def test_the_model_override_becomes_three_trainer_flags() -> None:
    command = command_for(base_spec())
    for flag, value in (("--embedding-dim", "16"), ("--n-heads", "2"), ("--n-layers", "2")):
        assert command.count(flag) == 1 and command[command.index(flag) + 1] == value


def test_no_model_override_and_no_loader_policy_add_no_flags() -> None:
    spec = base_spec()
    del spec["conditions"][0]["model"]
    command = command_for(spec)
    assert not {"--embedding-dim", "--n-heads", "--n-layers", "--loader-rng"} & set(command)


def test_the_loader_policy_becomes_a_trainer_flag_only_when_declared() -> None:
    command = command_for(base_spec(), loader_rng="legacy")
    assert command[command.index("--loader-rng") + 1] == "legacy"
    assert "--loader-rng" not in command_for(base_spec())


# ------------------------------------------------------------------ end to end on tiny data

@pytest.fixture(scope="module")
def evidence(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("capacity_keys")
    a, b = base / "a.txt", base / "b.txt"
    a.write_text("".join(f"Alba Cole{i}=20{i}-202-000{i}\n" for i in range(4)), encoding="utf-8")
    b.write_text("".join(f"Boro Dunn{i}=30{i}-303-111{i}\n" for i in range(4)), encoding="utf-8")
    config = base / "tiny_config.py"
    config.write_text("batch_size = 8\nepochs = 2\nembedding_dim = 16\nn_heads = 2\nn_layers = 1\n", encoding="utf-8")
    spec = {
        "name": "tiny_keys", "seed": 42, "model_config": str(config), "loader_rng": "legacy",
        "require_one_update_per_epoch": False,
        "scenarios": [{"id": "s1_n4", "book_a": str(a), "book_b": str(b)}],
        "conditions": [
            {"id": "with_model", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B", "lr": 0.01,
             "epochs": 3, "batch_size": 2, "accuracy_interval": 1, "adapt": False,
             "model": {"embedding_dim": 8, "n_heads": 2, "n_layers": 2}},
            {"id": "without_model", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B", "lr": 0.01,
             "epochs": 3, "batch_size": 2, "accuracy_interval": 1, "adapt": False},
        ],
    }
    root = base / "evidence"
    rx.run_spec(spec, root, repo_root=ROOT, provenance_fn=clean)
    return {"root": root, "spec": spec}


@pytest.fixture
def copy_of_evidence(evidence, tmp_path) -> Path:
    destination = tmp_path / "evidence"
    shutil.copytree(evidence["root"], destination)
    return destination


def audit(root: Path, **kwargs) -> dict:
    return audit_mod.audit(root, repo_root=ROOT, check_git=False, **kwargs)


def manifest(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


def rewrite_manifest(root: Path, change) -> None:
    records = manifest(root)
    change(records)
    (root / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def evaluation_path(root: Path, condition: str) -> Path:
    return root / "s1_n4" / f"{condition}.evaluation.json"


def test_evidence_with_overrides_passes_the_audit_with_and_without_local_artifacts(copy_of_evidence) -> None:
    assert audit(copy_of_evidence)["status"] == "passed"
    assert audit(copy_of_evidence, require_local_artifacts=True)["status"] == "passed"


def test_the_trained_models_have_the_declared_sizes_and_the_closed_form_parameter_count(evidence) -> None:
    root = evidence["root"]
    for condition, (d, layers) in {"with_model": (8, 2), "without_model": (16, 1)}.items():
        config = json.loads(evaluation_path(root, condition).read_text(encoding="utf-8"))["model_config"]
        assert (config["d_model"], config["n_layers"], config["n_heads"]) == (d, layers, 2)
        vocab = Vocabulary.from_pickle(root / "s1_n4" / "data_a1b0_evalB" / "meta.pkl")
        with (root / "s1_n4" / condition / "metrics.csv").open(newline="", encoding="utf-8") as file:
            logged = {int(row["Param Count"]) for row in csv.DictReader(file)}
        assert logged == {expected_param_count(vocab.vocab_size, config["seq_len"], d, layers)}


def test_the_recorded_commands_carry_exactly_the_declared_flags(evidence) -> None:
    by_condition = {record["run"]["condition"]: record["command"] for record in manifest(evidence["root"])}
    assert by_condition["with_model"].count("--embedding-dim") == 1
    assert "--embedding-dim" not in by_condition["without_model"]
    assert all(command[command.index("--loader-rng") + 1] == "legacy" for command in by_condition.values())


@pytest.mark.parametrize("flag,new_value", [("--embedding-dim", "16"), ("--n-heads", "4"), ("--n-layers", "1"), ("--loader-rng", "isolated")])
def test_a_changed_model_or_loader_flag_is_detected(copy_of_evidence, flag, new_value) -> None:
    def change(records):
        command = records[0]["command"]
        command[command.index(flag) + 1] = new_value
    rewrite_manifest(copy_of_evidence, change)
    with pytest.raises(ValueError, match="command"):
        audit(copy_of_evidence)


@pytest.mark.parametrize("flag", ["--embedding-dim", "--n-heads", "--n-layers", "--loader-rng"])
def test_a_dropped_model_or_loader_flag_is_detected(copy_of_evidence, flag) -> None:
    def change(records):
        command = records[0]["command"]
        index = command.index(flag)
        del command[index:index + 2]
    rewrite_manifest(copy_of_evidence, change)
    with pytest.raises(ValueError, match="command"):
        audit(copy_of_evidence)


def test_a_model_flag_on_a_run_that_declares_no_model_is_detected(copy_of_evidence) -> None:
    rewrite_manifest(copy_of_evidence, lambda records: records[1]["command"].extend(["--embedding-dim", "8"]))
    with pytest.raises(ValueError, match="command"):
        audit(copy_of_evidence)


def test_a_loader_flag_that_the_spec_never_declared_is_detected(evidence, tmp_path) -> None:
    spec = copy.deepcopy(evidence["spec"])
    del spec["loader_rng"]
    root = tmp_path / "other"
    shutil.copytree(evidence["root"], root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    protocol["spec"] = spec
    protocol["spec_sha256"] = rx.canonical_sha256(spec)
    (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="command"):
        audit(root)


def test_an_evaluated_checkpoint_of_the_wrong_size_is_detected(copy_of_evidence) -> None:
    path = evaluation_path(copy_of_evidence, "with_model")
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    evaluation["model_config"]["d_model"] = 16
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluated checkpoint's model differs"):
        audit(copy_of_evidence)


def test_a_logged_parameter_count_that_no_vocabulary_explains_is_detected(copy_of_evidence) -> None:
    path = copy_of_evidence / "s1_n4" / "with_model" / "metrics.csv"
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    for row in rows[1:]:
        row[-1] = str(int(row[-1]) + 1)
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file, lineterminator="\n").writerows(rows)
    with pytest.raises(ValueError, match="parameter count"):
        audit(copy_of_evidence)


def test_a_parameter_count_that_changes_during_a_run_is_detected(copy_of_evidence) -> None:
    path = copy_of_evidence / "s1_n4" / "with_model" / "metrics.csv"
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    rows[1][-1] = str(int(rows[1][-1]) + 2 * 8)    # still explained by a different vocabulary size, but not constant
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file, lineterminator="\n").writerows(rows)
    with pytest.raises(ValueError, match="parameter count"):
        audit(copy_of_evidence)
