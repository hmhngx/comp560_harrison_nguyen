from pathlib import Path

import pytest

from phonebook.run_replay_pilot import needs_extension, training_command, validate_inputs


def test_only_imperfect_joint_recall_triggers_extension():
    result = {"books": {name: {"generation": {"correct": 10, "total": 10}} for name in ["A", "B"]}}
    assert not needs_extension(result)
    result["books"]["B"]["generation"]["correct"] = 9
    assert needs_extension(result)


def test_adaptation_initializes_from_original_checkpoint_and_joint_does_not():
    condition = {"lr": 0.001, "epochs": 500, "adapt": True}
    command = training_command(condition, Path("data"), Path("new-output"), Path("original-phase-a.pth"))
    assert command[command.index("--adapt-from") + 1] == "original-phase-a.pth"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--epochs") + 1] == "500"
    condition["adapt"] = False
    assert "--adapt-from" not in training_command(condition, Path("data"), Path("output"), Path("original-phase-a.pth"))


def test_input_validation_detects_replay_composition_error(tmp_path):
    books = {name: tmp_path / f"{name}.txt" for name in ["A", "B"]}
    for name, path in books.items():
        path.write_text("".join(f"{name}{i}=333-512-{i:04d}\n" for i in range(10)), encoding="utf-8")
    replay = tmp_path / "replay.txt"
    replay.write_text(books["A"].read_text() + books["B"].read_text(), encoding="utf-8")
    validate_inputs(books, replay)
    replay.write_text(books["A"].read_text() * 2 + books["B"].read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="A once and B once"):
        validate_inputs(books, replay)
