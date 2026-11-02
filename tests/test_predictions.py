from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from phonebook import make_capacity_spec as mcs
from phonebook import make_e5_specs as e5
from phonebook import predictions as pr
from phonebook import run_experiment as rx

ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path: Path, text: str, name: str = "predictions.json") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_json_object_is_returned_unchanged_in_order_and_with_unicode(tmp_path: Path) -> None:
    data = {"q2_k3_faster_in": "5 of 6 — B is tilted up", "q1": [40, 50], "a": {"nested": True}}
    loaded = pr.load_predictions(write(tmp_path, json.dumps(data, ensure_ascii=False)))
    assert loaded == data and list(loaded) == list(data)


@pytest.mark.parametrize("text,message", [
    ("[1, 2]", "non-empty JSON object"),
    ("{}", "non-empty JSON object"),
    ('"just words"', "non-empty JSON object"),
    ("{not json", "not valid JSON"),
    ('{"q1": NaN}', "strict JSON"),
    ('{"q1": Infinity}', "strict JSON"),
])
def test_anything_but_a_strict_non_empty_object_is_refused(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pr.load_predictions(write(tmp_path, text))


def test_a_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read predictions file"):
        pr.load_predictions(tmp_path / "nowhere.json")


# ------------------------------------------------------------------ the E5 main spec

def write_phase_a(repo: Path, init: int, phonebook: int) -> Path:
    root = repo / "phonebook" / "experiments" / "week_x" / f"e5_phase_a_s{init}"
    folder = root / f"pb{phonebook}"
    (folder / "phase_a").mkdir(parents=True)
    (folder / "data_a1b0_evalB").mkdir(parents=True)
    (folder / "phase_a" / "model.pth").write_bytes(f"w{init}{phonebook}".encode())
    (folder / "data_a1b0_evalB" / "meta.pkl").write_bytes(f"m{init}".encode())
    books = {"A": {"generation": {"correct": 10, "total": 10}}, "B": {"generation": {"correct": 0, "total": 10}}}
    (folder / "phase_a.evaluation.json").write_text(json.dumps({"books": books}), encoding="utf-8")
    return root


def test_the_main_spec_embeds_the_predictions_file_verbatim(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    roots = {init: write_phase_a(tmp_path, init, 4) for init in (42, 123, 999)}
    monkeypatch.setattr(e5, "audit_phase_a", lambda root, repo_root: None)
    predictions = {"q1_k1_median_updates": "40 to 50", "q4_any_k_beats_1": "yes, k=3"}
    args = SimpleNamespace(command="main", phonebook=4, phase_a_root=[f"{i}={r}" for i, r in roots.items()],
                           predictions=write(tmp_path, json.dumps(predictions)))
    spec = e5.build_spec(args, repo_root=tmp_path)
    assert spec["decision_rules"]["predictions"] == predictions
    rx.validate_spec(spec)


def test_a_main_spec_built_without_predictions_has_no_predictions_block(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    roots = {init: write_phase_a(tmp_path, init, 4) for init in (42, 123, 999)}
    monkeypatch.setattr(e5, "audit_phase_a", lambda root, repo_root: None)
    args = SimpleNamespace(command="main", phonebook=4, phase_a_root=[f"{i}={r}" for i, r in roots.items()])
    assert "predictions" not in e5.build_spec(args, repo_root=tmp_path)["decision_rules"]


def test_the_command_line_will_not_write_a_main_spec_without_predictions(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, str(ROOT / "phonebook" / "make_e5_specs.py"), "main", "--phonebook", "2",
                             "--phase-a-root", "42=somewhere", "--out", str(tmp_path / "spec.json")],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "--predictions" in result.stderr and not (tmp_path / "spec.json").exists()


# ------------------------------------------------------------------ the capacity spec

def test_the_capacity_command_line_embeds_predictions_when_given(tmp_path: Path) -> None:
    target = tmp_path / "e4.json"
    predictions = write(tmp_path, json.dumps({"d16": "about 40 entries by 1000 epochs"}))
    command = [sys.executable, str(ROOT / "phonebook" / "make_capacity_spec.py"), "--d", "16", "--rungs", "10,20", "--seeds", "1,2,3",
               "--epochs", "100", "--budgets", "30,100", "--timeout", "600", "--out", str(target), "--predictions", str(predictions)]
    assert subprocess.run(command, capture_output=True, text=True, check=False).returncode == 0
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["decision_rules"]["predictions"] == {"d16": "about 40 entries by 1000 epochs"}
    assert mcs.capacity_spec(16, [10, 20], [1, 2, 3], epochs=100, accuracy_interval=10, budgets=[30, 100],
                             timeout_seconds=600, predictions={"d16": "about 40 entries by 1000 epochs"}) == written


def test_the_capacity_command_line_refuses_an_unreadable_predictions_file(tmp_path: Path) -> None:
    command = [sys.executable, str(ROOT / "phonebook" / "make_capacity_spec.py"), "--d", "16", "--rungs", "10,20", "--seeds", "1,2,3",
               "--epochs", "100", "--budgets", "30,100", "--timeout", "600", "--out", str(tmp_path / "e4.json"),
               "--predictions", str(tmp_path / "missing.json")]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "cannot read predictions file" in result.stderr and not (tmp_path / "e4.json").exists()
