from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from phonebook import make_capacity_books as mcb
from phonebook import run_experiment as rx

LINE = re.compile(r"[^=\n]+=[2-9]\d\d-[2-9]\d\d-\d{4}")


def entry(name: str, number: str = "200-200-0001") -> str:
    return f"{name}={number}"


def test_ladder_doubles_from_the_start_and_stops_at_the_cap() -> None:
    assert mcb.doubling_ladder(100) == [10, 20, 40, 80]
    assert mcb.doubling_ladder(80) == [10, 20, 40, 80]
    assert mcb.doubling_ladder(10) == [10]


def test_ladder_rejects_a_cap_below_its_first_rung() -> None:
    with pytest.raises(ValueError, match="at least"):
        mcb.doubling_ladder(9)


def test_big_book_is_deterministic_and_entries_are_unique_and_well_formed() -> None:
    first, second = mcb.generate_big_book(1, 80), mcb.generate_big_book(1, 80)
    assert first == second and len(first) == 80
    assert all(LINE.fullmatch(line) for line in first)
    assert len({line.split("=")[0] for line in first}) == 80
    assert len({line.split("=")[1] for line in first}) == 80
    assert mcb.generate_big_book(2, 80) != first


def test_prefixes_are_nested_so_a_larger_book_only_adds_entries() -> None:
    big = mcb.generate_big_book(1, 80)
    rungs = mcb.doubling_ladder(80)
    books = [mcb.prefix(big, n) for n in rungs]
    for small, large in zip(books, books[1:]):
        assert large[: len(small)] == small
    assert [len(book) for book in books] == rungs


def test_prefix_rejects_sizes_outside_the_book() -> None:
    big = [entry(f"n{i}", f"200-200-{i:04d}") for i in range(5)]
    with pytest.raises(ValueError, match="between 1 and 5"):
        mcb.prefix(big, 6)
    with pytest.raises(ValueError, match="between 1 and 5"):
        mcb.prefix(big, 0)


def test_control_book_takes_the_first_candidates_that_fit_and_are_unseen() -> None:
    train = [entry("aa bb", "200-200-0001"), entry("cc dd", "200-200-0002")]
    candidates = [
        entry("aa bb", "300-300-0001"),                 # name seen in training
        entry("ee ff", "200-200-0002"),                 # number seen in training
        entry("a very long name indeed", "300-300-0002"),  # too long for the context
        entry("gg hh", "300-300-0003"),
        entry("ii jj", "300-300-0004"),
        entry("kk ll", "300-300-0005"),
    ]
    max_line = len("gg hh=300-300-0003") + 1
    control = mcb.make_control_book(candidates, train, max_line=max_line, count=2)
    assert control == [entry("gg hh", "300-300-0003"), entry("ii jj", "300-300-0004")]


def test_control_book_fails_loudly_when_too_few_candidates_qualify() -> None:
    train = [entry("aa bb")]
    candidates = [entry("cc dd", "300-300-0001")]
    with pytest.raises(ValueError, match="only 1"):
        mcb.make_control_book(candidates, train, max_line=100, count=2)


def test_control_lines_never_exceed_the_context_of_the_smallest_rung() -> None:
    big = mcb.generate_big_book(3, 160)
    smallest = mcb.prefix(big, 10)
    control = mcb.make_control_book(mcb.generate_big_book(mcb.control_seed(3), 2000), big,
                                    max_line=mcb.longest_line(smallest))
    assert len(control) == mcb.CONTROL_ENTRIES
    assert max(mcb.longest_line([line]) for line in control) <= mcb.longest_line(smallest)
    train_names = {line.split("=")[0] for line in big}
    train_numbers = {line.split("=")[1] for line in big}
    assert not train_names & {line.split("=")[0] for line in control}
    assert not train_numbers & {line.split("=")[1] for line in control}


def test_written_inputs_are_nested_files_that_pass_the_runners_context_check(tmp_path: Path) -> None:
    rungs = [10, 20, 40]
    paths = mcb.write_capacity_inputs(tmp_path, seed=3, rungs=rungs)
    assert set(paths) == {"control", 10, 20, 40}
    books = {n: paths[n].read_text(encoding="ascii").splitlines() for n in paths}
    assert books[10] == books[20][:10] and books[20] == books[40][:20]
    assert all(len(books[n]) == n for n in rungs) and len(books["control"]) == mcb.CONTROL_ENTRIES
    assert all(paths[n].read_bytes().endswith(b"\n") and b"\r" not in paths[n].read_bytes() for n in paths)
    control_lines = [line + "\n" for line in books["control"]]
    for n in rungs:
        rx.check_sequence_lengths(f"n{n}", [line + "\n" for line in books[n]], control_lines, None)


def test_writing_twice_refuses_to_overwrite_a_tracked_input(tmp_path: Path) -> None:
    mcb.write_capacity_inputs(tmp_path, seed=3, rungs=[10, 20])
    with pytest.raises(ValueError, match="already exists"):
        mcb.write_capacity_inputs(tmp_path, seed=3, rungs=[10, 20])


@pytest.mark.parametrize("rungs", [[], [20, 10], [10, 10, 20], [0, 10]])
def test_rungs_must_be_a_strictly_increasing_list_of_positive_sizes(tmp_path: Path, rungs: list[int]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        mcb.write_capacity_inputs(tmp_path, seed=3, rungs=rungs)


def test_command_line_accepts_an_explicit_rung_list(tmp_path: Path) -> None:
    script = Path(mcb.__file__)
    result = subprocess.run([sys.executable, str(script), "--seed", "4", "--rungs", "10,14,20", "--out-dir", str(tmp_path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "capacity_seed4_control.txt", "capacity_seed4_n10.txt", "capacity_seed4_n14.txt", "capacity_seed4_n20.txt"]


def test_command_line_reports_a_refused_overwrite_as_an_error(tmp_path: Path) -> None:
    script = Path(mcb.__file__)
    command = [sys.executable, str(script), "--seed", "4", "--n-max", "20", "--out-dir", str(tmp_path)]
    assert subprocess.run(command, capture_output=True, text=True, check=False).returncode == 0
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode != 0 and "already exists" in again.stderr


def test_command_line_writes_every_rung_and_the_control(tmp_path: Path) -> None:
    script = Path(mcb.__file__)
    result = subprocess.run([sys.executable, str(script), "--seed", "4", "--n-max", "40", "--out-dir", str(tmp_path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "capacity_seed4_control.txt", "capacity_seed4_n10.txt", "capacity_seed4_n20.txt", "capacity_seed4_n40.txt"]
