"""Figures for E4, E5 and E6: every plotted number equals the committed result files (and a recount made here, not by the module),
the table behind each figure is written next to it, and the figures render at the designed size. Pixels are not compared."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import struct

import pytest

from phonebook import report_capacity_replay as fig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "phonebook" / "results"
WEEK = ROOT / "phonebook" / "experiments" / "week_2026-10-31"

# phonebook/RESULTS.md, claim 11 and finding 5: capacity in entries by budget (None = no rung of the ladder passed)
DOCUMENTED_E4 = {
    8: {100: None, 300: None, 500: None, 1000: 14},
    16: {100: None, 300: 20, 500: 40, 1000: 113},
    32: {100: None, 300: 160, 500: 226, 1000: 226},
    64: {100: 20, 300: 320, 500: 320, 1000: 453},
}
KS = (1, 2, 3, 4, 5, 9)
PHONEBOOKS = (2, 4, 7, 8, 9, 12)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    return struct.unpack(">II", data[16:24])


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("figures")
    written = fig.write_all(out, results=RESULTS, week=WEEK)
    return {"dir": out, "paths": written}


# ------------------------------------------------------------------ the palette is the validated one

def test_the_palette_is_exactly_the_one_validated_with_the_dataviz_script() -> None:
    assert fig.SERIES == ("#2a78d6", "#eb6834", "#1baf7a")              # slots 1 to 3, validated with --pairs all
    assert fig.BUDGET_RAMP == ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")   # blue steps 250, 400, 550, 700, validated with --ordinal
    assert fig.SEEDS_RAMP == ("#cde2fb", "#86b6ef", "#2a78d6", "#0d366b")
    assert fig.SURFACE == "#fcfcfb" and fig.GRID == "#e1e0d9" and fig.AXIS == "#c3c2b7"
    assert (fig.INK, fig.INK_2, fig.MUTED) == ("#0b0b0b", "#52514e", "#898781")


# ------------------------------------------------------------------ E4 capacity

def test_capacity_data_is_the_documented_e4_result() -> None:
    data = fig.capacity_data(load_json("e4_capacity.json"), load_json("e6_extension.json"))
    assert data["widths"] == [8, 16, 32, 64] and data["budgets"] == [100, 300, 500, 1000]
    assert data["capacity"] == DOCUMENTED_E4
    assert data["params"] == {8: 3064, 16: 9216, 32: 30720, 64: 110656}


def test_e6_marks_are_lower_bounds_where_the_rung_passed_and_a_failed_rung_at_width_64() -> None:
    data = fig.capacity_data(load_json("e4_capacity.json"), load_json("e6_extension.json"))
    assert data["e6"] == {8: {"rung": 20, "passes": True}, 16: {"rung": 160, "passes": True},
                          32: {"rung": 320, "passes": True}, 64: {"rung": 640, "passes": False}}


def test_the_capacity_table_has_one_row_per_mark_and_leaves_missing_capacities_blank() -> None:
    data = fig.capacity_data(load_json("e4_capacity.json"), load_json("e6_extension.json"))
    rows = fig.capacity_rows(data)
    assert len(rows) == 4 * 4 + 4
    e4 = [r for r in rows if r["source"] == "E4"]
    assert sorted({r["epochs"] for r in e4}) == [100, 300, 500, 1000]
    assert sum(r["entries"] == "" for r in e4) == 5                       # width 8 at 100/300/500, width 16 and 32 at 100
    assert {(r["model_d"], r["meaning"]) for r in rows if r["source"] == "E6"} == {
        (8, "at_least"), (16, "at_least"), (32, "at_least"), (64, "rung_not_passed")}
    assert all(r["epochs"] == 4000 for r in rows if r["source"] == "E6")


# ------------------------------------------------------------------ E4 pass map

def recount_passes(budget: int) -> dict:
    """Seeds perfect by the budget per (width, rung), recounted from the cells table and not from the JSON."""
    counts: dict = {}
    for row in read_csv(RESULTS / "e4_cells.csv"):
        key = (int(row["model_d"]), int(row["n"]))
        counts[key] = counts.get(key, 0) + (row[f"perfect_by_{budget}"] == "True")
    return counts


def recount_capacity(counts: dict, width: int):
    best = None
    for rung in sorted(n for d, n in counts if d == width):
        if counts[(width, rung)] < 2:
            break
        best = rung
    return best


def test_the_pass_map_counts_equal_a_recount_from_the_cells_table() -> None:
    rows = fig.passmap_rows(load_json("e4_capacity.json"))
    assert len(rows) == (5 + 8 + 9 + 10) * 4
    for budget in (100, 300, 500, 1000):
        counts = recount_passes(budget)
        mine = {(r["model_d"], r["rung"]): r["seeds_passed"] for r in rows if r["epochs"] == budget}
        assert mine == counts
        assert {r["seeds"] for r in rows} == {3}
        assert all(r["rung_passes"] == (r["seeds_passed"] >= 2) for r in rows)


def test_the_pass_map_marks_the_capacity_rung_of_each_column_and_nothing_else() -> None:
    rows = fig.passmap_rows(load_json("e4_capacity.json"))
    for width, by_budget in DOCUMENTED_E4.items():
        for budget, capacity in by_budget.items():
            marked = [r["rung"] for r in rows if r["model_d"] == width and r["epochs"] == budget and r["is_capacity_rung"]]
            assert marked == ([] if capacity is None else [capacity])
            assert capacity == recount_capacity(recount_passes(budget), width)


# ------------------------------------------------------------------ E5 replay ratio

def test_the_cells_loader_reads_blanks_as_none_and_the_plain_flag_as_a_boolean(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("phonebook,init,seed,lr,k,plain,condition,stable_joint,first_joint,min_a,epoch_of_min_a,final_A,final_B\n"
                    "2,42,42,0.001,1,False,k1_lr0.001,40,40,9,30,10,10\n"
                    "2,42,42,0.001,,True,plain_lr0.001,,,0,12,0,10\n", encoding="utf-8")
    first, plain = fig.load_replay_cells(path)
    assert first == {"phonebook": 2, "init": 42, "seed": 42, "lr": 0.001, "k": 1, "plain": False, "condition": "k1_lr0.001",
                     "stable_joint": 40, "first_joint": 40, "min_a": 9, "epoch_of_min_a": 30, "final_A": 10, "final_B": 10}
    assert plain["plain"] is True and plain["k"] is None and plain["stable_joint"] is None and plain["min_a"] == 0


def recount_medians(field: str) -> dict:
    per_phonebook: dict = {}
    for row in read_csv(RESULTS / "e5_cells.csv"):
        if row["lr"] == "0.001" and row["seed"] == "42" and row["plain"] == "False":
            per_phonebook.setdefault((int(row["phonebook"]), int(row["k"])), []).append(float(row[field]))
    medians = {key: statistics.median(values) for key, values in per_phonebook.items()}
    return {k: statistics.median([medians[(p, k)] for p in PHONEBOOKS]) for k in KS}


def test_replay_medians_are_the_documented_numbers_and_a_recount_from_the_cells() -> None:
    data = fig.replay_data(fig.load_replay_cells(RESULTS / "e5_cells.csv"))
    assert data["median_stable_joint"] == {1: 40.0, 2: 38.5, 3: 38.5, 4: 40.5, 5: 44.0, 9: 54.0}   # RESULTS.md finding 3
    assert data["median_min_a"] == {1: 9.0, 2: 6.5, 3: 4.0, 4: 3.0, 5: 2.0, 9: 0.5}
    assert data["median_stable_joint"] == recount_medians("stable_joint")
    assert data["median_min_a"] == recount_medians("min_a")
    assert data["phonebooks"] == list(PHONEBOOKS) and data["ks"] == list(KS)


def test_replay_phonebook_medians_equal_the_analysis_json() -> None:
    data = fig.replay_data(fig.load_replay_cells(RESULTS / "e5_cells.csv"))
    expected = load_json("e5_analysis.json")["lr"]["0.001"]["phonebook_medians"]
    for phonebook in PHONEBOOKS:
        for k in KS:
            assert data["stable_joint"][phonebook][k] == expected[str(phonebook)][str(k)]


def test_the_replay_table_has_a_median_row_per_k_and_one_row_per_phonebook_and_k() -> None:
    rows = fig.replay_rows(fig.replay_data(fig.load_replay_cells(RESULTS / "e5_cells.csv")))
    assert len(rows) == (len(PHONEBOOKS) + 1) * len(KS)
    medians = {r["k"]: r["stable_joint"] for r in rows if r["phonebook"] == "median"}
    assert medians == {1: 40.0, 2: 38.5, 3: 38.5, 4: 40.5, 5: 44.0, 9: 54.0}


# ------------------------------------------------------------------ E6 curves

def test_each_e6_curve_has_400_points_and_agrees_with_the_cells_table() -> None:
    curves = fig.curve_data(WEEK)
    cells = {(int(r["model_d"]), int(r["seed"])): r for r in read_csv(RESULTS / "e6_cells.csv")}
    assert sorted(curves) == sorted(cells) and len(curves) == 12
    for (width, seed), curve in curves.items():
        cell = cells[(width, seed)]
        assert curve["entries"] == int(cell["n"])
        assert curve["epochs"] == list(range(10, 4001, 10)) and len(curve["recalled"]) == 400
        assert curve["recalled"][-1] == int(cell["final"])
        assert all(0 <= value <= curve["entries"] for value in curve["recalled"])
        first = next((e for e, v in zip(curve["epochs"], curve["recalled"]) if v == curve["entries"]), None)
        assert first == (int(cell["first_perfect"]) if cell["first_perfect"] != "" else None)


def test_the_curves_table_has_one_row_per_logged_epoch_of_each_curve() -> None:
    rows = fig.curve_rows(fig.curve_data(WEEK))
    assert len(rows) == 12 * 400
    assert set(rows[0]) == {"model_d", "entries", "seed", "epoch", "recalled"}


# ------------------------------------------------------------------ what is written on the figures and where the cells and markers sit

# RESULTS.md / week README, E6 results: first epoch at 100% per seed (1, 2, 3), "never" with the best count reached
DOCUMENTED_NOTES = {
    (8, 1): "100% first at epoch 1,050", (8, 2): "100% first at epoch 1,050", (8, 3): "100% first at epoch 1,730",
    (16, 1): "100% first at epoch 3,270", (16, 2): "100% first at epoch 1,210", (16, 3): "100% first at epoch 3,200",
    (32, 1): "100% first at epoch 940", (32, 2): "100% first at epoch 2,960", (32, 3): "never 100%: best 319 of 320",
    (64, 1): "never 100%: best 637 of 640", (64, 2): "never 100%: best 638 of 640", (64, 3): "never 100%: best 636 of 640",
}


def test_the_note_under_each_e6_panel_says_when_100_percent_was_first_reached_or_the_best_it_got() -> None:
    curves = fig.curve_data(WEEK)
    assert {key: fig.curve_note(curve) for key, curve in curves.items()} == DOCUMENTED_NOTES
    cells = {(int(r["model_d"]), int(r["seed"])): r for r in read_csv(RESULTS / "e6_cells.csv")}
    for key, curve in curves.items():
        expected = int(cells[key]["first_perfect"]) if cells[key]["first_perfect"] != "" else None
        assert fig.first_perfect(curve) == expected


def test_the_pass_map_grid_is_the_recount_and_each_marker_sits_on_its_columns_capacity_rung() -> None:
    rows = fig.passmap_rows(load_json("e4_capacity.json"))
    budgets = [100, 300, 500, 1000]
    for width, by_budget in DOCUMENTED_E4.items():
        rungs, matrix, markers = fig.passmap_grid(rows, width, budgets)
        assert rungs == sorted(rungs) and len(matrix) == len(rungs) and all(len(row) == len(budgets) for row in matrix)
        for j, budget in enumerate(budgets):
            counts = recount_passes(budget)
            assert [matrix[i][j] for i in range(len(rungs))] == [counts[(width, rung)] for rung in rungs]
        assert sorted(col for col, _ in markers) == [j for j, b in enumerate(budgets) if by_budget[b] is not None]
        for col, row in markers:
            assert rungs[row] == by_budget[budgets[col]]


def test_the_numbers_and_names_written_on_the_figures_are_the_ones_in_the_data(tmp_path: Path) -> None:
    capacity = fig.capacity_data(load_json("e4_capacity.json"), load_json("e6_extension.json"))
    texts = fig.figure_e4_capacity(capacity, tmp_path / "a.png")
    assert {"14", "113", "226", "453", "at least 20", "at least 160", "at least 320", "640 did not pass",
            "100 epochs", "300 epochs", "500 epochs", "1000 epochs"} <= set(texts)
    assert not {"at least 640", "at least 453"} & set(texts)
    passmap = fig.figure_e4_passmap(fig.passmap_rows(load_json("e4_capacity.json")), capacity, tmp_path / "b.png")
    assert {"width 8", "width 16", "width 32", "width 64", "0 of 3 seeds", "3 of 3 seeds", "capacity at that budget", "905", "453", "14"} <= set(passmap)
    replay = fig.figure_e5_replay_k(fig.replay_data(fig.load_replay_cells(RESULTS / "e5_cells.csv")), tmp_path / "c.png")
    assert {"38.5", "54", "0.5"} <= set(replay)
    curves = fig.figure_e6_curves(fig.curve_data(WEEK), tmp_path / "d.png")
    assert set(DOCUMENTED_NOTES.values()) <= set(curves)
    assert {"width 8: 20 entries", "width 16: 160 entries", "width 32: 320 entries", "width 64: 640 entries"} <= set(curves)


# ------------------------------------------------------------------ rendering and the command line

EXPECTED_TABLE_ROWS = {"fig_e4_capacity.csv": 20, "fig_e4_passmap.csv": 128, "fig_e5_replay_k.csv": 42, "fig_e6_curves.csv": 4800}


def test_every_figure_renders_at_its_designed_size_with_its_table_beside_it(rendered) -> None:
    out = rendered["dir"]
    assert sorted(p.name for p in rendered["paths"]) == sorted(
        [f"{name}.png" for name in fig.FIGSIZES] + list(EXPECTED_TABLE_ROWS))
    for name, (width, height) in fig.FIGSIZES.items():
        path = out / f"{name}.png"
        assert png_size(path) == (round(width * fig.DPI), round(height * fig.DPI))
        assert path.stat().st_size > 20_000, f"{name} looks blank"
    for table, rows in EXPECTED_TABLE_ROWS.items():
        assert len(read_csv(out / table)) == rows


def test_the_tables_use_lf_line_endings_so_they_are_identical_on_every_checkout(rendered) -> None:
    for table in EXPECTED_TABLE_ROWS:
        assert b"\r" not in (rendered["dir"] / table).read_bytes()


def test_the_command_line_refuses_to_overwrite_and_replaces_with_force(tmp_path: Path) -> None:
    args = ["--out", str(tmp_path), "--results", str(RESULTS), "--week", str(WEEK)]
    fig.main(args)
    before = (tmp_path / "fig_e4_capacity.csv").read_bytes()
    (tmp_path / "fig_e4_capacity.csv").write_bytes(b"changed\n")
    with pytest.raises(SystemExit, match="already exists"):
        fig.main(args)
    assert (tmp_path / "fig_e4_capacity.csv").read_bytes() == b"changed\n"          # nothing was touched
    fig.main(args + ["--force"])
    assert (tmp_path / "fig_e4_capacity.csv").read_bytes() == before
