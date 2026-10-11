from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from phonebook import make_heldout_dataset as mh

ROOT = Path(__file__).resolve().parents[1]


def test_books_are_the_first_and_last_ten_entries() -> None:
    lines = [f"n{i}=1" for i in range(20)]
    a, b = mh.split_books(lines)
    assert a == lines[:10] and b == lines[10:]


def test_split_requires_exactly_twenty_entries() -> None:
    with pytest.raises(ValueError, match="20"):
        mh.split_books([f"n{i}=1" for i in range(19)])


def test_longest_line_counts_the_trailing_newline_like_the_trainer_does() -> None:
    assert mh.longest_line(["ab=1", "abcd=1"]) == len("abcd=1") + 1


def test_a_seed_qualifies_only_when_both_books_have_the_same_longest_line() -> None:
    same = ["aa=1"] * 10 + ["bb=2"] * 10
    assert mh.qualifies(*mh.split_books(same))
    different = ["aa=1"] * 10 + ["bbb=2"] * 10
    assert not mh.qualifies(*mh.split_books(different))


def test_first_qualifying_seeds_scans_upward_and_stops_once_it_has_enough() -> None:
    seen = []

    def make(seed: int) -> list[str]:
        seen.append(seed)
        return ["aa=1"] * 10 + (["bb=2"] if seed % 2 == 0 else ["bbb=2"]) * 10

    assert mh.first_qualifying_seeds(3, make) == [2, 4, 6]
    assert seen == [1, 2, 3, 4, 5, 6]


def test_first_qualifying_seeds_gives_up_after_the_scan_limit() -> None:
    with pytest.raises(ValueError, match="qualifying"):
        mh.first_qualifying_seeds(1, lambda seed: ["aa=1"] * 10 + ["bbb=2"] * 10, limit=5)


def test_generated_phonebooks_are_deterministic_with_twenty_unique_names() -> None:
    first, second = mh.generate_phonebook_lines(2), mh.generate_phonebook_lines(2)
    assert first == second and len(first) == 20
    assert len({line.split("=")[0] for line in first}) == 20
    assert mh.generate_phonebook_lines(3) != first


def test_the_seeds_found_while_planning_behave_as_recorded() -> None:
    # Seed 1: Phase A crashed (A longest 28, B longest 31). Seed 3: Phase A trains but Phase B cannot load (28 vs 29).
    assert [mh.qualifies(*mh.split_books(mh.generate_phonebook_lines(seed))) for seed in (1, 2, 3, 4, 7)] == \
        [False, True, False, True, True]
    assert mh.first_qualifying_seeds(3, mh.generate_phonebook_lines) == [2, 4, 7]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "phonebook" / "make_heldout_dataset.py"), *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_cli_writes_lf_books_named_by_seed(tmp_path: Path) -> None:
    result = run_cli("--seed", "2", "--out-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    for book in ("A", "B"):
        data = (tmp_path / f"heldout_seed2_phonebook{book}.txt").read_bytes()
        assert b"\r" not in data and data.count(b"\n") == 10


def test_cli_refuses_a_seed_whose_books_cannot_share_a_context(tmp_path: Path) -> None:
    result = run_cli("--seed", "1", "--out-dir", str(tmp_path))
    assert result.returncode != 0 and "longest" in result.stderr
    assert not list(tmp_path.iterdir())


def test_cli_refuses_to_overwrite_existing_books(tmp_path: Path) -> None:
    assert run_cli("--seed", "2", "--out-dir", str(tmp_path)).returncode == 0
    again = run_cli("--seed", "2", "--out-dir", str(tmp_path))
    assert again.returncode != 0 and "exists" in again.stderr


def test_cli_can_write_the_first_qualifying_seeds(tmp_path: Path) -> None:
    result = run_cli("--first-qualifying", "3", "--out-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"heldout_seed{seed}_phonebook{book}.txt" for seed in (2, 4, 7) for book in "AB"]
