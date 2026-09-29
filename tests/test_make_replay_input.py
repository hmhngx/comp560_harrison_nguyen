from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _build_replay_lines(*args, **kwargs) -> list[str]:
    sys.path.insert(0, str(ROOT / "phonebook"))
    import make_replay_input as module

    return module.build_replay_lines(*args, **kwargs)


def test_repeats_a_k_times_then_appends_b_once() -> None:
    a = ["a1", "a2"]
    b = ["b1"]
    assert _build_replay_lines(a, b, k=3) == ["a1", "a2", "a1", "a2", "a1", "a2", "b1"]


def test_k_equals_one_is_a_once_plus_b_once() -> None:
    a = ["a1", "a2"]
    b = ["b1", "b2"]
    assert _build_replay_lines(a, b, k=1) == ["a1", "a2", "b1", "b2"]


def test_preserves_line_order_within_each_source() -> None:
    """Order doesn't affect training dynamics (DataLoader shuffles), but the
    function itself must not silently reorder or dedupe -- that would make
    the actual k-copies count wrong without any visible symptom."""
    a = ["first", "second", "third"]
    b = ["fourth"]
    result = _build_replay_lines(a, b, k=2)
    assert result == ["first", "second", "third", "first", "second", "third", "fourth"]


@pytest.mark.parametrize("bad_k", [0, -1, -5])
def test_rejects_k_less_than_one(bad_k) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        _build_replay_lines(["a1"], ["b1"], k=bad_k)


def test_cli_reads_files_and_writes_expected_line_count(run_script, tmp_path: Path) -> None:
    phonebook_a = tmp_path / "phonebookA.txt"
    phonebook_b = tmp_path / "phonebookB.txt"
    phonebook_a.write_text("alice=111\nbob=222\n", encoding="utf-8")
    phonebook_b.write_text("carol=333\n", encoding="utf-8")
    out_path = tmp_path / "replay_k3.txt"

    result = run_script(
        "phonebook/make_replay_input.py",
        [
            "--phonebook-a", str(phonebook_a),
            "--phonebook-b", str(phonebook_b),
            "--k", "3",
            "--out", str(out_path),
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 * 3 + 1  # 2 A-lines x k=3, plus 1 B-line
    assert lines.count("alice=111") == 3
    assert lines.count("bob=222") == 3
    assert lines.count("carol=333") == 1


def test_cli_rejects_k_less_than_one(run_script, tmp_path: Path) -> None:
    phonebook_a = tmp_path / "phonebookA.txt"
    phonebook_b = tmp_path / "phonebookB.txt"
    phonebook_a.write_text("alice=111\n", encoding="utf-8")
    phonebook_b.write_text("carol=333\n", encoding="utf-8")

    result = run_script(
        "phonebook/make_replay_input.py",
        [
            "--phonebook-a", str(phonebook_a),
            "--phonebook-b", str(phonebook_b),
            "--k", "0",
            "--out", str(tmp_path / "out.txt"),
        ],
        cwd=tmp_path,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "k must be" in combined.lower()
