from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import pytest

from phonebook import audit_experiment as audit_mod
from phonebook import run_experiment as rx

ROOT = Path(__file__).resolve().parents[1]


def clean() -> dict:
    return {"commit": "testcommit", "dirty": False}


def tiny_books(base: Path) -> tuple[Path, Path]:
    a, b = base / "a.txt", base / "b.txt"
    a.write_text("".join(f"Alba Cole{i}=20{i}-202-000{i}\n" for i in range(4)), encoding="utf-8")
    b.write_text("".join(f"Boro Dunn{i}=30{i}-303-111{i}\n" for i in range(4)), encoding="utf-8")
    return a, b


@pytest.fixture(scope="module")
def evidence(tmp_path_factory) -> dict:
    # AUDIT_EVIDENCE_CACHE lets a developer reuse one tiny-evidence build across repeated runs (mutation checks).
    cache = os.environ.get("AUDIT_EVIDENCE_CACHE")
    if cache and (Path(cache) / "evidence_b" / "summary.csv").exists():
        return {"base": Path(cache), "root": Path(cache) / "evidence_b", "books": (Path(cache) / "a.txt", Path(cache) / "b.txt")}
    base = Path(cache) if cache else tmp_path_factory.mktemp("audit_base")
    base.mkdir(parents=True, exist_ok=True)
    a, b = tiny_books(base)
    config = base / "tiny_config.py"
    config.write_text("batch_size = 8\nepochs = 2\nembedding_dim = 8\nn_heads = 2\nn_layers = 1\n", encoding="utf-8")
    phase_a = {
        "name": "tiny_phase_a", "seed": 42, "model_config": str(config),
        "scenarios": [{"id": "s1", "book_a": str(a), "book_b": str(b)}],
        "conditions": [{"id": "phase_a", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B",
                        "lr": 0.01, "epochs": 2, "batch_size": 8, "accuracy_interval": 1, "adapt": False}],
    }
    root_a = base / "evidence_a"
    rx.run_spec(phase_a, root_a, repo_root=ROOT, provenance_fn=clean)
    adapt = {
        "name": "tiny_adapt", "seed": 42, "model_config": str(config),
        "scenarios": [{"id": "s1", "book_a": str(a), "book_b": str(b),
                       "init": {"checkpoint": str(root_a / "s1/phase_a/model.pth"),
                                "meta": str(root_a / "s1/data_a1b0_evalB/meta.pkl")}}],
        "conditions": [
            {"id": "plain", "train": {"a_copies": 0, "b_copies": 1}, "eval_book": "A",
             "lr": 0.01, "epochs": 3, "batch_size": 8, "accuracy_interval": 1, "adapt": True},
            {"id": "replay_l2sp", "train": {"a_copies": 1, "b_copies": 2}, "eval_book": "A",
             "lr": 0.01, "epochs": 3, "batch_size": 16, "accuracy_interval": 1, "adapt": True, "l2sp_lambda": 0.1},
        ],
    }
    root_b = base / "evidence_b"
    rx.run_spec(adapt, root_b, repo_root=ROOT, provenance_fn=clean)
    return {"base": base, "root": root_b, "books": (a, b)}


@pytest.fixture
def copy_of_evidence(evidence, tmp_path) -> Path:
    destination = tmp_path / "evidence"
    shutil.copytree(evidence["root"], destination)
    return destination


def audit(root: Path, **kwargs) -> dict:
    return audit_mod.audit(root, repo_root=ROOT, check_git=False, **kwargs)


def edit_json(path: Path, change) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_untouched_evidence_passes_and_reports_what_it_checked(copy_of_evidence) -> None:
    report = audit(copy_of_evidence)
    assert report["status"] == "passed" and report["runs"] == 2
    assert report["completions"] == 2 * 2 * 4  # two runs, two books, four entries each


def test_a_raw_prediction_edited_without_its_score_is_detected(copy_of_evidence) -> None:
    def change(data):
        first = data["books"]["A"]["generation"]["predictions"][0]
        # Make the raw string contradict the recorded score whichever way the original was scored.
        first["prediction"], first["stop_reason"] = ("tampered", "newline") if first["correct"] else (first["expected"], "newline")
    edit_json(copy_of_evidence / "s1/plain.evaluation.json", change)
    with pytest.raises(ValueError, match="disagrees"):
        audit(copy_of_evidence)


def test_re_evaluation_catches_tampering_that_leaves_the_score_unchanged(copy_of_evidence) -> None:
    def change(data):
        predictions = data["books"]["A"]["generation"]["predictions"]
        victim = next(p for p in predictions if not p["correct"])  # a wrong answer replaced by a different wrong answer
        victim["prediction"] = victim["prediction"] + "?"
    edit_json(copy_of_evidence / "s1/plain.evaluation.json", change)
    assert audit(copy_of_evidence)["status"] == "passed"  # internally consistent, so the readable audit cannot see it
    with pytest.raises(ValueError, match="fresh"):
        audit(copy_of_evidence, require_local_artifacts=True)


def test_a_flipped_correct_flag_is_detected(copy_of_evidence) -> None:
    def change(data):
        first = data["books"]["B"]["generation"]["predictions"][0]
        first["correct"] = not first["correct"]
    edit_json(copy_of_evidence / "s1/plain.evaluation.json", change)
    with pytest.raises(ValueError, match="disagrees"):
        audit(copy_of_evidence)


def test_a_missing_final_logged_epoch_is_detected(copy_of_evidence) -> None:
    path = copy_of_evidence / "s1/replay_l2sp/metrics.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="epochs"):
        audit(copy_of_evidence)


def test_a_non_finite_metric_is_detected(copy_of_evidence) -> None:
    path = copy_of_evidence / "s1/plain/metrics.csv"
    rows = list(csv.reader(path.open(newline="")))
    rows[1][1] = "nan"
    with path.open("w", newline="") as file:
        csv.writer(file, lineterminator="\n").writerows(rows)
    with pytest.raises(ValueError, match="finite"):
        audit(copy_of_evidence)


def test_a_logged_final_recall_that_contradicts_the_strict_evaluation_is_detected(copy_of_evidence) -> None:
    path = copy_of_evidence / "s1/plain/metrics.csv"
    rows = list(csv.reader(path.open(newline="")))
    header = rows[0]
    column = header.index("Val Seq Acc")
    rows[-1][column] = "0.2500" if rows[-1][column] != "0.2500" else "0.5000"
    with path.open("w", newline="") as file:
        csv.writer(file, lineterminator="\n").writerows(rows)
    with pytest.raises(ValueError, match="agree"):
        audit(copy_of_evidence)


def test_a_command_that_differs_from_the_protocol_is_detected(copy_of_evidence) -> None:
    def change(data):
        command = data["command"]
        command[command.index("--lr") + 1] = "0.5"
    path = copy_of_evidence / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    change(records[0])
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="command"):
        audit(copy_of_evidence)


def test_a_dropped_manifest_record_is_detected(copy_of_evidence) -> None:
    path = copy_of_evidence / "manifest.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        audit(copy_of_evidence)


def test_a_failed_training_run_is_detected(copy_of_evidence) -> None:
    path = copy_of_evidence / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["returncode"] = 1
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="failed"):
        audit(copy_of_evidence)


def test_an_edited_summary_count_is_detected(copy_of_evidence) -> None:
    path = copy_of_evidence / "summary.csv"
    rows = list(csv.reader(path.open(newline="")))
    column = rows[0].index("A_correct")
    rows[1][column] = str(int(rows[1][column]) + 1)
    with path.open("w", newline="") as file:
        csv.writer(file, lineterminator="\n").writerows(rows)
    with pytest.raises(ValueError, match="summary"):
        audit(copy_of_evidence)


def test_a_replay_mixture_that_differs_from_the_declared_composition_is_detected(copy_of_evidence) -> None:
    edit_json(copy_of_evidence / "prepared_data.json", lambda data: data["s1/a1b2_evalA"].update({"train_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="mixture"):
        audit(copy_of_evidence)


def test_a_spec_edited_after_the_fact_is_detected(copy_of_evidence) -> None:
    edit_json(copy_of_evidence / "protocol.json", lambda data: data["spec"]["conditions"][0].update({"lr": 0.5}))
    with pytest.raises(ValueError, match="spec"):
        audit(copy_of_evidence)


def test_training_on_a_dirty_tree_is_rejected(copy_of_evidence) -> None:
    edit_json(copy_of_evidence / "protocol.json", lambda data: data["git"].update({"dirty": True}))
    with pytest.raises(ValueError, match="dirty"):
        audit(copy_of_evidence)


def test_an_input_book_changed_after_the_run_is_detected(evidence, copy_of_evidence) -> None:
    a, _ = evidence["books"]
    original = a.read_bytes()
    try:
        a.write_bytes(original.replace(b"Alba Cole0", b"Alba Cole9"))
        with pytest.raises(ValueError, match="Input"):
            audit(copy_of_evidence)
    finally:
        a.write_bytes(original)
    assert audit(copy_of_evidence)["status"] == "passed"


def test_local_artifact_check_catches_a_replaced_checkpoint(evidence, copy_of_evidence) -> None:
    assert audit(copy_of_evidence, require_local_artifacts=True)["local_artifacts_checked"] is True
    checkpoint = evidence["root"] / "s1/plain/model.pth"
    original = checkpoint.read_bytes()
    try:
        checkpoint.write_bytes(original + b"x")
        with pytest.raises(ValueError, match="Checkpoint"):
            audit(copy_of_evidence, require_local_artifacts=True)
    finally:
        checkpoint.write_bytes(original)


def test_without_local_artifacts_the_committed_readable_evidence_still_audits(copy_of_evidence) -> None:
    for name in ["s1/plain/model.pth", "s1/replay_l2sp/model.pth"]:
        (copy_of_evidence / name).unlink()
    assert audit(copy_of_evidence)["local_artifacts_checked"] is False


def rewrite_manifest(root: Path, change) -> None:
    path = root / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    change(records)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.mark.parametrize("flag,new_value", [
    ("--device", "cuda"), ("--seed", "7"), ("--lr", "0.5"), ("--l2sp-lambda", "9"), ("--epochs", "99"),
    ("--batch-size", "3"), ("--accuracy-interval", "2"),
])
def test_every_recorded_training_flag_is_checked_against_the_protocol(copy_of_evidence, flag, new_value) -> None:
    def change(records):
        command = records[1]["command"]
        command[command.index(flag) + 1] = new_value
    rewrite_manifest(copy_of_evidence, change)
    with pytest.raises(ValueError, match="command"):
        audit(copy_of_evidence)


def test_adapting_from_a_different_checkpoint_is_detected(copy_of_evidence) -> None:
    def change(records):
        command = records[0]["command"]
        command[command.index("--adapt-from") + 1] = str(Path(command[command.index("--adapt-from") + 1]).with_name("other.pth"))
    rewrite_manifest(copy_of_evidence, change)
    with pytest.raises(ValueError, match="command"):
        audit(copy_of_evidence)


def test_dropping_the_adaptation_flag_is_detected(copy_of_evidence) -> None:
    def change(records):
        command = records[0]["command"]
        index = command.index("--adapt-from")
        del command[index:index + 2]
    rewrite_manifest(copy_of_evidence, change)
    with pytest.raises(ValueError, match="command"):
        audit(copy_of_evidence)


def test_a_manifest_record_from_a_different_code_state_is_detected(copy_of_evidence) -> None:
    rewrite_manifest(copy_of_evidence, lambda records: records[0]["git"].update({"commit": "someothercommit"}))
    with pytest.raises(ValueError, match="manifest record"):
        audit(copy_of_evidence)


def test_an_edited_training_log_is_detected(copy_of_evidence) -> None:
    log = copy_of_evidence / "s1/plain.log"
    log.write_text(log.read_text() + "edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Log"):
        audit(copy_of_evidence)


def test_an_evaluation_that_omits_a_book_is_detected(copy_of_evidence) -> None:
    edit_json(copy_of_evidence / "s1/plain.evaluation.json", lambda data: data["books"].pop("B"))
    with pytest.raises(ValueError, match="books"):
        audit(copy_of_evidence)


def test_a_model_config_changed_after_the_run_is_detected(evidence, copy_of_evidence) -> None:
    config = evidence["base"] / "tiny_config.py"
    original = config.read_bytes()
    try:
        config.write_bytes(original + b"\n# edited\n")
        with pytest.raises(ValueError, match="config"):
            audit(copy_of_evidence)
    finally:
        config.write_bytes(original)


def test_the_config_hash_tolerates_checkout_newline_conventions(evidence, copy_of_evidence) -> None:
    config = evidence["base"] / "tiny_config.py"
    original = config.read_bytes()
    try:
        config.write_bytes(original.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        assert audit(copy_of_evidence)["status"] == "passed"
        config.write_bytes(original.replace(b"\r\n", b"\n"))
        assert audit(copy_of_evidence)["status"] == "passed"
    finally:
        config.write_bytes(original)


def test_a_protocol_whose_runs_differ_from_the_spec_expansion_is_detected(copy_of_evidence) -> None:
    def change(data):
        data["runs"][0]["lr"] = 0.5
    edit_json(copy_of_evidence / "protocol.json", change)
    with pytest.raises(ValueError, match="Protocol run"):
        audit(copy_of_evidence)


def book_scores(tmp_path: Path, *, prediction: str, stop: str, correct: bool, forced_correct: int) -> tuple[dict, Path]:
    source = tmp_path / "book.txt"
    source.write_text("Ann Lee=123\n", encoding="utf-8")
    from phonebook.evaluate_checkpoint import sha256
    scores = {
        "source_sha256": sha256(source),
        "generation": {"correct": int(correct), "total": 1, "accuracy": float(correct), "stop_counts":
                       {"newline": int(stop == "newline"), "budget": int(stop == "budget"), "context": int(stop == "context")},
                       "predictions": [{"prompt": "Ann Lee", "expected": "123", "prediction": prediction,
                                        "stop_reason": stop, "correct": correct}]},
        "teacher_forced": {"token_total": 4, "token_correct": 4, "token_accuracy": 1.0, "sequence_total": 1,
                           "sequence_correct": forced_correct, "sequence_accuracy": float(forced_correct), "loss": 0.1},
    }
    return scores, source


def test_a_limit_stop_cannot_be_scored_correct_even_when_the_text_matches(tmp_path) -> None:
    scores, source = book_scores(tmp_path, prediction="123", stop="budget", correct=True, forced_correct=1)
    with pytest.raises(ValueError, match="disagrees"):
        audit_mod._audit_book("run", "A", scores, source)
    scores, source = book_scores(tmp_path, prediction="123", stop="newline", correct=True, forced_correct=1)
    audit_mod._audit_book("run", "A", scores, source)


def test_teacher_forced_and_generation_exact_recall_must_agree(tmp_path) -> None:
    scores, source = book_scores(tmp_path, prediction="123", stop="newline", correct=True, forced_correct=0)
    with pytest.raises(ValueError, match="do not agree"):
        audit_mod._audit_book("run", "A", scores, source)


def test_an_evaluation_file_copied_from_another_run_is_detected(copy_of_evidence) -> None:
    first, second = copy_of_evidence / "s1/plain.evaluation.json", copy_of_evidence / "s1/replay_l2sp.evaluation.json"
    first_text, second_text = first.read_text(), second.read_text()
    first.write_text(second_text)
    second.write_text(first_text)
    with pytest.raises(ValueError, match="different run"):
        audit(copy_of_evidence)


def edit_csv(path: Path, change) -> None:
    rows = list(csv.reader(path.open(newline="")))
    change(rows)
    with path.open("w", newline="") as file:
        csv.writer(file, lineterminator="\n").writerows(rows)


def test_an_edited_prepared_evaluation_book_hash_is_detected(copy_of_evidence) -> None:
    edit_json(copy_of_evidence / "prepared_data.json", lambda data: data["s1/a1b2_evalA"].update({"eval_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="evaluation book"):
        audit(copy_of_evidence)


def test_prepared_data_must_cover_exactly_the_protocol_modes(copy_of_evidence) -> None:
    edit_json(copy_of_evidence / "prepared_data.json", lambda data: data.pop("s1/a1b2_evalA"))
    with pytest.raises(ValueError, match="Prepared data does not cover"):
        audit(copy_of_evidence)


def test_a_summary_row_with_a_different_hyperparameter_is_detected(copy_of_evidence) -> None:
    def change(rows):
        rows[2][rows[0].index("lr")] = "0.5"
    edit_csv(copy_of_evidence / "summary.csv", change)
    with pytest.raises(ValueError, match="summary row differs"):
        audit(copy_of_evidence)


def test_a_summary_accuracy_that_differs_from_the_evaluation_is_detected(copy_of_evidence) -> None:
    def change(rows):
        rows[1][rows[0].index("A_token_accuracy")] = "0.123456"
    edit_csv(copy_of_evidence / "summary.csv", change)
    with pytest.raises(ValueError, match="summary accuracy"):
        audit(copy_of_evidence)


def test_a_summary_with_a_missing_run_is_detected(copy_of_evidence) -> None:
    edit_csv(copy_of_evidence / "summary.csv", lambda rows: rows.pop())
    with pytest.raises(ValueError, match="summary rows"):
        audit(copy_of_evidence)


@pytest.mark.parametrize("column", ["Train Token Acc", "Val Token Acc"])
def test_logged_token_accuracies_must_match_the_evaluation(copy_of_evidence, column) -> None:
    def change(rows):
        index = rows[0].index(column)
        rows[-1][index] = "0.1234" if rows[-1][index] != "0.1234" else "0.4321"
    edit_csv(copy_of_evidence / "s1/replay_l2sp/metrics.csv", change)
    with pytest.raises(ValueError, match="token accuracy"):
        audit(copy_of_evidence)


def test_local_artifact_check_catches_a_modified_prepared_binary(copy_of_evidence) -> None:
    binary = copy_of_evidence / "s1/data_a0b1_evalA/train.bin"
    binary.write_bytes(binary.read_bytes() + b"\x00\x00")
    with pytest.raises(ValueError, match="Prepared data hash"):
        audit(copy_of_evidence, require_local_artifacts=True)


def test_local_artifact_check_catches_a_replaced_phase_a_checkpoint(evidence, copy_of_evidence) -> None:
    init = json.loads((copy_of_evidence / "protocol.json").read_text())["initializations"]["s1"]["checkpoint"]
    original = Path(init).read_bytes()
    try:
        Path(init).write_bytes(original + b"x")
        with pytest.raises(ValueError, match="initialization"):
            audit(copy_of_evidence, require_local_artifacts=True)
    finally:
        Path(init).write_bytes(original)


def _scores_for_mutation(tmp_path: Path) -> tuple[dict, Path]:
    return book_scores(tmp_path, prediction="123", stop="newline", correct=True, forced_correct=1)


@pytest.mark.parametrize("message,mutate", [
    ("Unexpected generation denominator", lambda s: s["generation"].update({"total": 2})),
    ("Prompt coverage differs", lambda s: s["generation"]["predictions"][0].update({"prompt": "Someone Else"})),
    ("Ground truth differs", lambda s: s["generation"]["predictions"][0].update({"expected": "999"})),
    ("Generation total differs", lambda s: s["generation"].update({"correct": 0, "accuracy": 0.0})),
    ("rate or stop counts", lambda s: s["generation"].update({"accuracy": 0.5})),
    ("rate or stop counts", lambda s: s["generation"]["stop_counts"].update({"newline": 0, "budget": 1})),
    ("Token denominator", lambda s: s["teacher_forced"].update({"token_total": 5, "token_accuracy": 0.8})),
    ("Sequence denominator", lambda s: s["teacher_forced"].update({"sequence_total": 2})),
    ("Token accuracy differs", lambda s: s["teacher_forced"].update({"token_accuracy": 0.5})),
    ("Sequence accuracy differs", lambda s: s["teacher_forced"].update({"sequence_accuracy": 0.5})),
    ("Non-finite cross-entropy", lambda s: s["teacher_forced"].update({"loss": float("nan")})),
    ("Input hash differs", lambda s: s.update({"source_sha256": "0" * 64})),
])
def test_each_per_book_consistency_check_fires(tmp_path, message, mutate) -> None:
    scores, source = _scores_for_mutation(tmp_path)
    audit_mod._audit_book("run", "A", scores, source)  # the unmutated scores are valid
    mutate(scores)
    with pytest.raises(ValueError, match=message):
        audit_mod._audit_book("run", "A", scores, source)


def test_a_spec_edited_in_a_way_that_changes_no_run_is_still_detected_by_its_hash(copy_of_evidence) -> None:
    edit_json(copy_of_evidence / "protocol.json", lambda data: data["spec"].update({"description": "changed after the fact"}))
    with pytest.raises(ValueError, match="recorded hash"):
        audit(copy_of_evidence)


def test_a_recorded_input_hash_that_differs_from_the_file_is_detected(copy_of_evidence) -> None:
    def change(data):
        data["inputs"][next(iter(data["inputs"]))] = "0" * 64
    edit_json(copy_of_evidence / "protocol.json", change)
    with pytest.raises(ValueError, match="Input hash differs") as raised:
        audit(copy_of_evidence)
    assert "evaluated" not in str(raised.value)  # the protocol check fired, not the per-book check


def test_a_protocol_that_lost_a_run_is_detected(copy_of_evidence) -> None:
    edit_json(copy_of_evidence / "protocol.json", lambda data: data["runs"].pop())
    with pytest.raises(ValueError, match="Protocol runs differ"):
        audit(copy_of_evidence)


def test_an_evaluation_pointing_at_another_runs_checkpoint_is_detected(copy_of_evidence) -> None:
    def change(data):
        parts = Path(data["checkpoint"]).parts
        data["checkpoint"] = str(Path(*parts[:-2], "replay_l2sp", "model.pth"))
    edit_json(copy_of_evidence / "s1/plain.evaluation.json", change)
    with pytest.raises(ValueError, match="different run's checkpoint"):
        audit(copy_of_evidence)


def test_an_evaluation_pointing_at_another_runs_vocabulary_file_is_detected(copy_of_evidence) -> None:
    def change(data):
        parts = Path(data["metadata"]).parts
        data["metadata"] = str(Path(*parts[:-2], "data_a1b2_evalA", "meta.pkl"))
    edit_json(copy_of_evidence / "s1/plain.evaluation.json", change)
    with pytest.raises(ValueError, match="different run's vocabulary"):
        audit(copy_of_evidence)


def test_local_artifact_check_catches_a_modified_initial_vocabulary(copy_of_evidence) -> None:
    meta = Path(json.loads((copy_of_evidence / "protocol.json").read_text())["initializations"]["s1"]["meta"])
    original = meta.read_bytes()
    try:
        meta.write_bytes(original + b"x")
        with pytest.raises(ValueError, match="Metadata hash differs: initialization"):
            audit(copy_of_evidence, require_local_artifacts=True)
    finally:
        meta.write_bytes(original)


def test_local_artifact_check_catches_a_modified_evaluation_vocabulary(evidence, copy_of_evidence) -> None:
    meta = evidence["root"] / "s1/data_a0b1_evalA/meta.pkl"
    original = meta.read_bytes()
    try:
        meta.write_bytes(original + b"x")
        with pytest.raises(ValueError, match="Metadata hash differs: s1/plain"):
            audit(copy_of_evidence, require_local_artifacts=True)
    finally:
        meta.write_bytes(original)


def test_local_artifact_check_catches_an_edited_prepared_training_text(copy_of_evidence) -> None:
    train = copy_of_evidence / "s1/inputs_a0b1_evalA/train.txt"
    train.write_bytes(train.read_bytes() + b"extra\n")
    with pytest.raises(ValueError, match="Prepared training text"):
        audit(copy_of_evidence, require_local_artifacts=True)
