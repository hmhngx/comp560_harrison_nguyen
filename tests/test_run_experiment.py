from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from phonebook import run_experiment as rx

ROOT = Path(__file__).resolve().parents[1]


def make_spec() -> dict:
    return {
        "name": "unit",
        "seed": 42,
        "model_config": "phonebook/config/phonebook.py",
        "scenarios": [
            {"id": "s1", "book_a": "a.txt", "book_b": "b.txt",
             "init": {"checkpoint": "init.pth", "meta": "meta.pkl"}},
        ],
        "conditions": [
            {"id": "plain", "train": {"a_copies": 0, "b_copies": 1}, "eval_book": "A",
             "lr": 0.001, "epochs": 5, "batch_size": 64, "accuracy_interval": 1, "adapt": True},
        ],
    }


def test_valid_spec_passes() -> None:
    rx.validate_spec(make_spec())


def test_unknown_keys_are_rejected_so_typos_cannot_silently_change_a_protocol() -> None:
    spec = make_spec()
    spec["conditions"][0]["l2sp_lamda"] = 1.0
    with pytest.raises(ValueError, match="l2sp_lamda"):
        rx.validate_spec(spec)


def test_l2sp_requires_adaptation() -> None:
    spec = make_spec()
    spec["conditions"][0].update({"adapt": False, "l2sp_lambda": 0.5})
    with pytest.raises(ValueError, match="l2sp"):
        rx.validate_spec(spec)


def test_adaptation_requires_an_init_for_every_scenario() -> None:
    spec = make_spec()
    del spec["scenarios"][0]["init"]
    with pytest.raises(ValueError, match="init"):
        rx.validate_spec(spec)


def test_fresh_training_needs_no_init() -> None:
    spec = make_spec()
    del spec["scenarios"][0]["init"]
    spec["conditions"][0]["adapt"] = False
    rx.validate_spec(spec)


def test_duplicate_scenario_and_condition_ids_are_rejected() -> None:
    spec = make_spec()
    spec["scenarios"].append(copy.deepcopy(spec["scenarios"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        rx.validate_spec(spec)
    spec = make_spec()
    spec["conditions"].append(copy.deepcopy(spec["conditions"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        rx.validate_spec(spec)


def test_ids_must_be_filesystem_safe() -> None:
    spec = make_spec()
    spec["conditions"][0]["id"] = "bad/id"
    with pytest.raises(ValueError, match="id"):
        rx.validate_spec(spec)


def test_book_b_is_required_whenever_it_is_used() -> None:
    spec = make_spec()
    del spec["scenarios"][0]["book_b"]
    with pytest.raises(ValueError, match="book_b"):
        rx.validate_spec(spec)
    spec["conditions"][0].update({"train": {"a_copies": 1, "b_copies": 0}, "eval_book": "A"})
    spec["evaluate_books"] = ["A"]
    rx.validate_spec(spec)


@pytest.mark.parametrize("field,value", [
    ("lr", 0), ("lr", -0.001), ("epochs", 0), ("batch_size", 0), ("accuracy_interval", 0),
    ("eval_book", "C"), ("adapt", "yes"),
])
def test_bad_condition_values_are_rejected(field, value) -> None:
    spec = make_spec()
    spec["conditions"][0][field] = value
    with pytest.raises(ValueError):
        rx.validate_spec(spec)


@pytest.mark.parametrize("train", [{"a_copies": -1, "b_copies": 1}, {"a_copies": 0, "b_copies": 0},
                                   {"a_copies": 1}, {"a_copies": 1, "b_copies": 1, "c": 2}])
def test_bad_train_mixtures_are_rejected(train) -> None:
    spec = make_spec()
    spec["conditions"][0]["train"] = train
    with pytest.raises(ValueError):
        rx.validate_spec(spec)


def test_expand_runs_is_scenario_major_with_stable_ids() -> None:
    spec = make_spec()
    spec["scenarios"].append({**copy.deepcopy(spec["scenarios"][0]), "id": "s2"})
    spec["conditions"].append({**copy.deepcopy(spec["conditions"][0]), "id": "replay",
                               "train": {"a_copies": 1, "b_copies": 1}, "l2sp_lambda": 0.1})
    runs = rx.expand_runs(spec)
    assert [run["id"] for run in runs] == ["s1/plain", "s1/replay", "s2/plain", "s2/replay"]
    assert runs[1]["l2sp_lambda"] == 0.1 and runs[0]["l2sp_lambda"] == 0.0
    assert runs[1]["mode"] == "a1b1_evalA" and runs[0]["mode"] == "a0b1_evalA"


def test_training_command_adapts_only_when_asked_and_carries_every_setting() -> None:
    run = {**make_spec()["conditions"][0], "l2sp_lambda": 0.3}
    command = rx.training_command(run, config="cfg.py", data_dir=Path("d"), out_dir=Path("o"),
                                  seed=7, init_checkpoint=Path("init.pth"))
    flags = {flag: command[command.index(flag) + 1] for flag in
             ["--device", "--data-dir", "--out-dir", "--seed", "--lr", "--epochs", "--batch-size",
              "--accuracy-interval", "--l2sp-lambda", "--adapt-from"]}
    assert flags == {"--device": "cpu", "--data-dir": "d", "--out-dir": "o", "--seed": "7", "--lr": "0.001",
                     "--epochs": "5", "--batch-size": "64", "--accuracy-interval": "1",
                     "--l2sp-lambda": "0.3", "--adapt-from": "init.pth"}
    fresh = rx.training_command({**run, "adapt": False, "l2sp_lambda": 0.0}, config="cfg.py", data_dir=Path("d"),
                                out_dir=Path("o"), seed=7, init_checkpoint=None)
    assert "--adapt-from" not in fresh


@pytest.mark.parametrize("n_train,batch,expected", [(20, 64, 1), (64, 64, 1), (65, 64, 2), (100, 128, 1), (1, 1, 1)])
def test_updates_per_epoch_is_ceiling_division(n_train, batch, expected) -> None:
    assert rx.updates_per_epoch(n_train, batch) == expected


@pytest.mark.parametrize("epochs,interval,expected", [
    (5, 1, [1, 2, 3, 4, 5]),
    (25, 10, [10, 20, 25]),
    (25, 7, [7, 10, 14, 20, 21, 25]),
    (100, 100, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
])
def test_expected_logged_epochs_matches_what_train_py_writes(epochs, interval, expected) -> None:
    assert rx.expected_logged_epochs(epochs, interval) == expected


def test_sequence_length_guard_catches_a_longer_evaluation_book() -> None:
    train, evaluation = ["abc=1\n", "abcd=1\n"], ["abcde=1\n"]
    with pytest.raises(ValueError, match="longer"):
        rx.check_sequence_lengths("s/c", train, evaluation, None)


def test_sequence_length_guard_catches_a_checkpoint_with_a_different_context() -> None:
    lines = ["abc=1\n", "abcd=1\n"]
    assert rx.check_sequence_lengths("s/c", lines, lines, 7) == 7
    with pytest.raises(ValueError, match="context"):
        rx.check_sequence_lengths("s/c", lines, lines, 6)
    assert rx.check_sequence_lengths("s/c", lines, lines, None) == 7


def test_committed_specs_are_valid_and_expand() -> None:
    specs = sorted((ROOT / "phonebook" / "specs").glob("*.json"))
    for path in specs:
        spec = rx.load_spec(path)
        rx.validate_spec(spec)
        assert rx.expand_runs(spec), path


def tiny_books(tmp_path: Path) -> tuple[Path, Path]:
    # Same-length names in both books: the trainer infers the context length from the training book only.
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("".join(f"Alba Cole{i}=20{i}-202-000{i}\n" for i in range(4)), encoding="utf-8")
    b.write_text("".join(f"Boro Dunn{i}=30{i}-303-111{i}\n" for i in range(4)), encoding="utf-8")
    return a, b


def tiny_config(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_config.py"
    path.write_text("batch_size = 8\nepochs = 2\nembedding_dim = 8\nn_heads = 2\nn_layers = 1\n", encoding="utf-8")
    return path


def clean(commit: str = "testcommit") -> dict:
    return {"commit": commit, "dirty": False}


@pytest.mark.integration
def test_end_to_end_tiny_phase_a_then_adaptation_leaves_complete_evidence(tmp_path: Path) -> None:
    a, b = tiny_books(tmp_path)
    config = tiny_config(tmp_path)
    phase_a_spec = {
        "name": "tiny_phase_a", "seed": 42, "model_config": str(config),
        "scenarios": [{"id": "s1", "book_a": str(a), "book_b": str(b)}],
        "conditions": [{"id": "phase_a", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B",
                        "lr": 0.01, "epochs": 2, "batch_size": 8, "accuracy_interval": 1, "adapt": False}],
    }
    root_a = tmp_path / "evidence_a"
    rx.run_spec(phase_a_spec, root_a, repo_root=ROOT, provenance_fn=lambda: clean())

    for name in ["protocol.json", "prepared_data.json", "manifest.jsonl", "summary.csv",
                 "s1/phase_a.evaluation.json", "s1/phase_a.log", "s1/phase_a/metrics.csv", "s1/phase_a/model.pth"]:
        assert (root_a / name).exists(), name
    protocol = json.loads((root_a / "protocol.json").read_text())
    assert protocol["git"] == clean() and protocol["spec"] == phase_a_spec
    assert protocol["runs"][0]["updates_per_epoch"] == 1 and protocol["runs"][0]["seq_len"] > 0

    with pytest.raises(ValueError, match="exists"):
        rx.run_spec(phase_a_spec, root_a, repo_root=ROOT, provenance_fn=lambda: clean())

    adapt_spec = {
        "name": "tiny_adapt", "seed": 42, "model_config": str(config),
        "scenarios": [{"id": "s1", "book_a": str(a), "book_b": str(b),
                       "init": {"checkpoint": str(root_a / "s1/phase_a/model.pth"),
                                "meta": str(root_a / "s1/data_a1b0_evalB/meta.pkl")}}],
        "conditions": [
            {"id": "plain", "train": {"a_copies": 0, "b_copies": 1}, "eval_book": "A",
             "lr": 0.01, "epochs": 2, "batch_size": 8, "accuracy_interval": 1, "adapt": True},
            {"id": "replay_l2sp", "train": {"a_copies": 1, "b_copies": 2}, "eval_book": "A",
             "lr": 0.01, "epochs": 2, "batch_size": 16, "accuracy_interval": 1, "adapt": True, "l2sp_lambda": 0.1},
        ],
    }
    root_b = tmp_path / "evidence_b"
    rx.run_spec(adapt_spec, root_b, repo_root=ROOT, provenance_fn=lambda: clean())

    manifest = [json.loads(line) for line in (root_b / "manifest.jsonl").read_text().splitlines()]
    assert [record["run"]["id"] for record in manifest] == ["s1/plain", "s1/replay_l2sp"]
    assert all(record["returncode"] == 0 and not record["timed_out"] for record in manifest)
    assert all("--adapt-from" in record["command"] for record in manifest)

    rows = list(csv.DictReader((root_b / "summary.csv").open(newline="")))
    assert [row["run_id"] for row in rows] == ["s1/plain", "s1/replay_l2sp"]
    assert all(int(row["A_total"]) == 4 and int(row["B_total"]) == 4 for row in rows)
    assert rows[1]["a_copies"] == "1" and rows[1]["b_copies"] == "2" and rows[1]["l2sp_lambda"] == "0.1"

    metrics = list(csv.DictReader((root_b / "s1/replay_l2sp/metrics.csv").open(newline="")))
    assert [int(row["Epoch"]) for row in metrics] == [1, 2]


@pytest.mark.integration
def test_dirty_tree_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    a, b = tiny_books(tmp_path)
    spec = {
        "name": "tiny", "seed": 42, "model_config": str(tiny_config(tmp_path)),
        "scenarios": [{"id": "s1", "book_a": str(a), "book_b": str(b)}],
        "conditions": [{"id": "phase_a", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B",
                        "lr": 0.01, "epochs": 1, "batch_size": 8, "accuracy_interval": 1, "adapt": False}],
    }
    root = tmp_path / "evidence"
    with pytest.raises(ValueError, match="clean"):
        rx.run_spec(spec, root, repo_root=ROOT, provenance_fn=lambda: {"commit": "x", "dirty": True})
    assert not root.exists()


def test_planning_rejects_an_evaluation_book_longer_than_the_training_context(tmp_path: Path) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("Al Co=201-202-0001\n", encoding="utf-8")
    b.write_text("Alexander Cooper=301-303-1111\n", encoding="utf-8")
    spec = {
        "name": "mismatch", "seed": 42, "model_config": "phonebook/config/phonebook.py",
        "scenarios": [{"id": "s1", "book_a": str(a), "book_b": str(b)}],
        "conditions": [{"id": "phase_a", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B",
                        "lr": 0.01, "epochs": 1, "batch_size": 8, "accuracy_interval": 1, "adapt": False}],
    }
    with pytest.raises(ValueError, match="longer"):
        rx.plan(spec, repo_root=ROOT, provenance=clean())


def fake_initialization(tmp_path: Path, context: int) -> tuple[Path, Path]:
    import torch

    checkpoint, meta = tmp_path / "init.pth", tmp_path / "meta.pkl"
    torch.save({"state_dict": {"position_embedding.weight": torch.zeros(context, 4)}}, checkpoint)
    meta.write_bytes(b"meta")
    return checkpoint, meta


def test_pinned_initialization_hashes_are_enforced_at_planning_time(tmp_path: Path) -> None:
    from phonebook.evaluate_checkpoint import sha256

    a, b = tiny_books(tmp_path)
    checkpoint, meta = fake_initialization(tmp_path, context=24)
    init = {"checkpoint": str(checkpoint), "meta": str(meta),
            "checkpoint_sha256": sha256(checkpoint), "meta_sha256": sha256(meta)}
    spec = {
        "name": "pinned", "seed": 42, "model_config": str(tiny_config(tmp_path)),
        "scenarios": [{"id": "s1", "book_a": str(a), "book_b": str(b), "init": init}],
        "conditions": [{"id": "plain", "train": {"a_copies": 0, "b_copies": 1}, "eval_book": "A",
                        "lr": 0.01, "epochs": 1, "batch_size": 8, "accuracy_interval": 1, "adapt": True}],
    }
    protocol = rx.plan(spec, repo_root=ROOT, provenance=clean())
    assert protocol["initializations"]["s1"]["checkpoint_sha256"] == init["checkpoint_sha256"]
    assert protocol["initializations"]["s1"]["context_length"] == 24

    for key in ("checkpoint_sha256", "meta_sha256"):
        wrong = copy.deepcopy(spec)
        wrong["scenarios"][0]["init"][key] = "0" * 64
        with pytest.raises(ValueError, match="pinned"):
            rx.plan(wrong, repo_root=ROOT, provenance=clean())


def test_pinned_hashes_must_be_well_formed_to_pass_validation() -> None:
    spec = make_spec()
    spec["scenarios"][0]["init"]["checkpoint_sha256"] = "a" * 64
    spec["scenarios"][0]["init"]["meta_sha256"] = "0123456789abcdef" * 4
    rx.validate_spec(spec)
    for bad in ("not-a-hash", "a" * 63, "A" * 64, 12345):
        spec["scenarios"][0]["init"]["checkpoint_sha256"] = bad
        with pytest.raises(ValueError, match="64 lowercase hex"):
            rx.validate_spec(spec)


def test_planning_enforces_one_update_per_epoch_when_the_spec_requires_it(tmp_path: Path) -> None:
    a, b = tiny_books(tmp_path)
    spec = {
        "name": "batching", "seed": 42, "model_config": str(tiny_config(tmp_path)),
        "scenarios": [{"id": "s1", "book_a": str(a), "book_b": str(b)}],
        "conditions": [{"id": "phase_a", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B",
                        "lr": 0.01, "epochs": 1, "batch_size": 2, "accuracy_interval": 1, "adapt": False}],
    }
    with pytest.raises(ValueError, match="update"):
        rx.plan(spec, repo_root=ROOT, provenance=clean())
    spec["require_one_update_per_epoch"] = False
    assert rx.plan(spec, repo_root=ROOT, provenance=clean())["runs"][0]["updates_per_epoch"] == 2
