from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from phonebook import make_e5_specs as e5
from phonebook import make_heldout_dataset as mh
from phonebook import run_experiment as rx

ROOT = Path(__file__).resolve().parents[1]
PHONEBOOKS = [2, 4, 7, 8, 9, 12]


def fake_pins(inits=(42, 123, 999)) -> dict:
    return {init: {"checkpoint": f"ev/s{init}/pb2/phase_a/model.pth", "checkpoint_sha256": f"{init:064x}",
                   "meta": f"ev/s{init}/pb2/data_a1b0_evalB/meta.pkl", "meta_sha256": f"{init + 1:064x}"}
            for init in inits}


# ------------------------------------------------------------------ phase A

def test_phase_a_spec_validates_and_uses_the_init_seed_as_the_spec_seed() -> None:
    spec = e5.phase_a_spec(123, PHONEBOOKS)
    rx.validate_spec(spec)
    assert spec["seed"] == 123 and spec["name"] == "e5_phase_a_s123"
    assert [s["id"] for s in spec["scenarios"]] == [f"pb{p}" for p in PHONEBOOKS]


def test_phase_a_spec_points_at_the_heldout_books_the_generator_writes() -> None:
    first = e5.phase_a_spec(42, PHONEBOOKS)["scenarios"][0]
    a_path, b_path = mh.book_paths(Path("phonebook/inputs"), 2)
    assert first["book_a"] == a_path.as_posix() and first["book_b"] == b_path.as_posix()


def test_phase_a_training_recipe_is_exactly_the_one_e2_used() -> None:
    e2 = json.loads((ROOT / "phonebook" / "specs" / "e2_phase_a.json").read_text(encoding="utf-8"))
    assert e5.phase_a_spec(42, PHONEBOOKS)["conditions"] == e2["conditions"]


# ------------------------------------------------------------------ pins and the gate

def write_phase_a(repo: Path, init: int, phonebook: int, *, correct: int = 10, total: int = 10) -> Path:
    root = repo / "phonebook" / "experiments" / "week_x" / f"e5_phase_a_s{init}"
    folder = root / f"pb{phonebook}"
    (folder / "phase_a").mkdir(parents=True)
    (folder / "data_a1b0_evalB").mkdir(parents=True)
    (folder / "phase_a" / "model.pth").write_bytes(f"weights-{init}-{phonebook}".encode())
    (folder / "data_a1b0_evalB" / "meta.pkl").write_bytes(f"meta-{init}".encode())
    books = {"A": {"generation": {"correct": correct, "total": total}}, "B": {"generation": {"correct": 0, "total": total}}}
    (folder / "phase_a.evaluation.json").write_text(json.dumps({"books": books}), encoding="utf-8")
    return root


def test_pins_are_repository_relative_paths_with_the_hashes_of_the_files(tmp_path: Path) -> None:
    root = write_phase_a(tmp_path, 42, 2)
    pins = e5.read_pins(root, 2, repo_root=tmp_path)
    assert pins["checkpoint"] == "phonebook/experiments/week_x/e5_phase_a_s42/pb2/phase_a/model.pth"
    assert pins["meta"] == "phonebook/experiments/week_x/e5_phase_a_s42/pb2/data_a1b0_evalB/meta.pkl"
    assert pins["checkpoint_sha256"] == hashlib.sha256(b"weights-42-2").hexdigest()
    assert pins["meta_sha256"] == hashlib.sha256(b"meta-42").hexdigest()


def test_a_phase_a_model_that_did_not_memorize_its_own_book_is_refused(tmp_path: Path) -> None:
    root = write_phase_a(tmp_path, 42, 2, correct=9)
    with pytest.raises(ValueError, match="gate.*9/10"):
        e5.read_pins(root, 2, repo_root=tmp_path)


def test_missing_phase_a_evidence_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        e5.read_pins(tmp_path / "nowhere", 2, repo_root=tmp_path)


def test_evidence_outside_the_repository_cannot_be_pinned(tmp_path: Path) -> None:
    root = write_phase_a(tmp_path / "elsewhere", 42, 2)
    with pytest.raises(ValueError, match="repository"):
        e5.read_pins(root, 2, repo_root=tmp_path / "repo")


# ------------------------------------------------------------------ main grid

def test_the_main_spec_validates_and_has_three_inits_of_one_phonebook() -> None:
    spec = e5.main_spec(2, fake_pins())
    rx.validate_spec(spec)
    assert spec["name"] == "e5_main_pb2" and spec["seed"] == 42 and "loader_rng" not in spec
    assert [s["id"] for s in spec["scenarios"]] == ["pb2_s42", "pb2_s123", "pb2_s999"]
    assert spec["scenarios"][1]["init"]["checkpoint_sha256"] == f"{123:064x}"


def test_the_main_grid_is_the_pre_registered_one() -> None:
    conditions = e5.main_spec(2, fake_pins())["conditions"]
    by_id = {c["id"]: c for c in conditions}
    assert len(conditions) == len(by_id) == 15
    primary = {f"k{k}_lr0.001" for k in (1, 2, 3, 4, 5, 9)} | {"plain_lr0.001"}
    robust = {f"k{k}_lr{lr}" for lr in ("0.003", "0.0003") for k in (1, 2, 3, 5)}
    assert set(by_id) == primary | robust
    assert by_id["k5_lr0.003"]["train"] == {"a_copies": 1, "b_copies": 5} and by_id["k5_lr0.003"]["lr"] == 0.003
    assert by_id["plain_lr0.001"]["train"] == {"a_copies": 0, "b_copies": 1}


def test_every_condition_is_one_batch_per_update_with_per_update_logging() -> None:
    for c in e5.main_spec(2, fake_pins())["conditions"]:
        sequences = 10 * c["train"]["a_copies"] + 10 * c["train"]["b_copies"]
        assert sequences <= c["batch_size"] == 128
        assert (c["epochs"], c["accuracy_interval"], c["adapt"], c["eval_book"]) == (500, 1, True, "A")


def test_the_rule_text_names_the_unit_the_noise_band_and_the_verdict() -> None:
    rules = e5.main_spec(2, fake_pins())["decision_rules"]
    for key in ("question", "endpoint", "unit", "noise", "verdict", "plateau", "work", "scope"):
        assert key in rules and rules[key]


def test_the_main_spec_needs_pins_for_every_init() -> None:
    with pytest.raises(ValueError, match="init"):
        e5.main_spec(2, {})


def test_main_spec_from_evidence_pins_each_init_from_its_own_phase_a_folder(tmp_path: Path) -> None:
    roots = {init: write_phase_a(tmp_path, init, 4) for init in (42, 123, 999)}
    spec = e5.main_spec_from_evidence(4, roots, repo_root=tmp_path)
    assert [s["id"] for s in spec["scenarios"]] == ["pb4_s42", "pb4_s123", "pb4_s999"]
    digest = hashlib.sha256(b"weights-123-4").hexdigest()
    assert spec["scenarios"][1]["init"]["checkpoint_sha256"] == digest


# ------------------------------------------------------------------ noise arm and anchor

def test_the_noise_spec_repeats_two_configurations_of_three_units_under_another_shuffle_seed() -> None:
    spec = e5.noise_spec(3, {pb: fake_pins((42,))[42] for pb in (2, 4, 7)})
    rx.validate_spec(spec)
    assert spec["name"] == "e5_noise_seed3" and spec["seed"] == 3
    assert [s["id"] for s in spec["scenarios"]] == ["pb2_s42", "pb4_s42", "pb7_s42"]
    assert [c["id"] for c in spec["conditions"]] == ["k1_lr0.001", "k3_lr0.001"]


def test_the_noise_arm_must_not_reuse_the_main_shuffle_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        e5.noise_spec(42, {2: fake_pins((42,))[42]})


def test_the_anchor_keeps_only_k1_and_k3_at_the_default_rate_and_uses_the_historical_loader() -> None:
    e3 = json.loads((ROOT / "phonebook" / "specs" / "e3_replay_ratio.json").read_text(encoding="utf-8"))
    spec = e5.anchor_spec(e3)
    rx.validate_spec(spec)
    assert spec["name"] == "e5_anchor_e3" and spec["loader_rng"] == "legacy"
    assert [c["id"] for c in spec["conditions"]] == ["k1_lr0.001", "k3_lr0.001"]
    assert spec["scenarios"] == e3["scenarios"] and spec["seed"] == e3["seed"]
    originals = {c["id"]: c for c in e3["conditions"]}
    assert all(c == originals[c["id"]] for c in spec["conditions"])


def test_the_anchor_refuses_a_spec_that_lacks_a_wanted_condition() -> None:
    with pytest.raises(ValueError, match="k3_lr0.001"):
        e5.anchor_spec({"name": "x", "seed": 1, "scenarios": [], "conditions": [{"id": "k1_lr0.001"}]})


# ------------------------------------------------------------------ command line

def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "phonebook" / "make_e5_specs.py"), *args],
                          capture_output=True, text=True, check=False)


def test_the_command_line_writes_phase_a_and_anchor_specs_and_refuses_to_overwrite(tmp_path: Path) -> None:
    phase_a, anchor = tmp_path / "pa.json", tmp_path / "anchor.json"
    assert run_cli("phase-a", "--init", "42", "--phonebooks", "2,4", "--out", str(phase_a)).returncode == 0
    assert run_cli("anchor", "--e3-spec", str(ROOT / "phonebook/specs/e3_replay_ratio.json"), "--out", str(anchor)).returncode == 0
    rx.validate_spec(json.loads(phase_a.read_text(encoding="utf-8")))
    rx.validate_spec(json.loads(anchor.read_text(encoding="utf-8")))
    again = run_cli("phase-a", "--init", "42", "--phonebooks", "2,4", "--out", str(phase_a))
    assert again.returncode != 0 and "already exists" in again.stderr


# ------------------------------------------------------------------ review fixes

def test_building_a_main_spec_audits_every_phase_a_folder_first(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    roots = {init: write_phase_a(tmp_path, init, 4) for init in (42, 123, 999)}
    audited = []
    monkeypatch.setattr(e5, "audit_phase_a", lambda root, repo_root: audited.append(Path(root)))
    args = SimpleNamespace(command="main", phonebook=4, phase_a_root=[f"{init}={root}" for init, root in roots.items()])
    spec = e5.build_spec(args, repo_root=tmp_path)
    assert spec["name"] == "e5_main_pb4" and sorted(audited) == sorted(roots.values())


def test_building_a_noise_spec_audits_the_init_42_folder_and_pins_each_phonebook(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    root = write_phase_a(tmp_path, 42, 2)
    write_phase_a(tmp_path, 42, 4)
    audited = []
    monkeypatch.setattr(e5, "audit_phase_a", lambda root, repo_root: audited.append(Path(root)))
    args = SimpleNamespace(command="noise", shuffle_seed=3, phonebooks="2,4", phase_a_root=f"42={root}")
    spec = e5.build_spec(args, repo_root=tmp_path)
    assert [s["id"] for s in spec["scenarios"]] == ["pb2_s42", "pb4_s42"] and audited == [root]


def test_the_noise_arm_needs_its_phase_a_root_keyed_by_the_main_init(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    monkeypatch.setattr(e5, "audit_phase_a", lambda root, repo_root: None)
    args = SimpleNamespace(command="noise", shuffle_seed=3, phonebooks="2", phase_a_root=f"123={write_phase_a(tmp_path, 123, 2)}")
    with pytest.raises(ValueError, match="42=PATH"):
        e5.build_spec(args, repo_root=tmp_path)


def test_an_audit_failure_becomes_a_refusal_to_pin(monkeypatch) -> None:
    def failing(*args, **kwargs):
        raise ValueError("Checkpoint hash differs")

    monkeypatch.setattr(e5.audit_mod, "audit", failing)
    with pytest.raises(ValueError, match="Phase-A evidence failed its audit"):
        e5.audit_phase_a(Path("nowhere"), repo_root=Path("."))
