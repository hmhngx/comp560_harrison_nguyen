"""Tests for the report's E7 figure table (phonebook/report_e7_curves.py) and the report-ready PDF figures (report/figures.py).

The table fig_e7_curves.csv is checked against an independent recompute from the per-run metrics.csv files (written here without importing
the code under test). The figures are checked by reading the plotted numbers back out of the matplotlib objects, and the PDFs for the
things a submission cares about: width, font size, embedded fonts, and no creation date.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
from pathlib import Path

import pytest

from phonebook import report_e7_curves as e7c
from report import figures

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "phonebook" / "results"
EXPERIMENTS = ROOT / "phonebook" / "experiments"
WEEK6, WEEK7 = EXPERIMENTS / "week_2026-10-31", EXPERIMENTS / "week_2026-11-13"
ARMS, WIDTHS, SEEDS = ("E6", "W", "S"), (16, 32, 64), (1, 2, 3)
COLUMNS = ["arm", "model_d", "entries", "seed", "epoch", "recalled"]


def read_table(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def raw_curves() -> dict[tuple[str, int, int], tuple[int, list[tuple[int, int]]]]:
    """{(arm, width, seed): (entries, [(epoch, recalled)])} from the metrics.csv files, recalled = round(Train Seq Acc * entries)."""
    out = {}
    for arm, week, pattern in (("E6", WEEK6, "e6_extend_d*_seed*"), ("W", WEEK7, "e7_w_d*_seed*"), ("S", WEEK7, "e7_s_d*_seed*")):
        for root in sorted(week.glob(pattern)):
            if "__interrupted" in root.name:
                continue
            width, seed = map(int, re.fullmatch(r".*_d(\d+)_seed(\d+)", root.name).groups())
            if width not in WIDTHS:
                continue
            (path,) = root.glob("s*_n*/d*/metrics.csv")
            n = int(re.fullmatch(r"s\d+_n(\d+)", path.parts[-3]).group(1))
            rows = []
            for record in read_table(path):
                recalled = float(record["Train Seq Acc"]) * n
                assert abs(recalled - round(recalled)) < 0.05
                rows.append((int(record["Epoch"]), int(round(recalled))))
            out[(arm, width, seed)] = (n, sorted(rows))
    return out


@pytest.fixture(scope="module")
def raw():
    return raw_curves()


@pytest.fixture(scope="module")
def table():
    return read_table(RESULTS / "fig_e7_curves.csv")


# ---- the table ------------------------------------------------------------------------------------------------------------------------------
def test_the_table_has_the_documented_columns_lf_endings_and_one_row_per_logged_epoch_of_27_curves(table):
    data = (RESULTS / "fig_e7_curves.csv").read_bytes()
    assert data.split(b"\n")[0].decode() == ",".join(COLUMNS) and b"\r" not in data.replace(b"\r\n", b"\n") and b"\r\n" not in data.replace(b"\r\n", b"\n")
    assert len(table) == 27 * 400


def test_every_row_equals_the_independent_recompute_from_the_raw_metrics(table, raw):
    assert len(raw) == 27
    expected = [
        {"arm": arm, "model_d": str(d), "entries": str(n), "seed": str(s), "epoch": str(e), "recalled": str(k)}
        for arm in ARMS for d in WIDTHS for s in SEEDS for (n, rows) in [raw[(arm, d, s)]] for e, k in rows
    ]
    assert table == expected


def test_the_e6_rows_are_the_e6_table_of_the_earlier_figure_at_widths_16_32_and_64(table):
    earlier = [r for r in read_table(RESULTS / "fig_e6_curves.csv") if int(r["model_d"]) in WIDTHS]
    mine = [{k: r[k] for k in ("model_d", "entries", "seed", "epoch", "recalled")} for r in table if r["arm"] == "E6"]
    assert mine == earlier


def test_the_writer_regenerates_the_committed_table_byte_for_byte(tmp_path):
    path = e7c.write_e7_table(tmp_path, week_e6=WEEK6, week_e7=WEEK7)
    assert path.name == "fig_e7_curves.csv"
    assert path.read_bytes() == (RESULTS / "fig_e7_curves.csv").read_bytes().replace(b"\r\n", b"\n")


def test_the_writer_refuses_to_overwrite_and_replaces_with_force(tmp_path):
    e7c.write_e7_table(tmp_path, week_e6=WEEK6, week_e7=WEEK7)
    with pytest.raises(ValueError, match="already exists"):
        e7c.write_e7_table(tmp_path, week_e6=WEEK6, week_e7=WEEK7)
    (tmp_path / "fig_e7_curves.csv").write_text("changed", encoding="utf-8")
    e7c.write_e7_table(tmp_path, week_e6=WEEK6, week_e7=WEEK7, force=True)
    assert (tmp_path / "fig_e7_curves.csv").read_text(encoding="utf-8").startswith("arm,model_d")


def test_too_few_e7_runs_is_an_error(tmp_path):
    shutil.copytree(WEEK7 / "e7_w_d16_seed1", tmp_path / "e7_w_d16_seed1")
    with pytest.raises(ValueError, match="expected 18 E7 runs, found 1"):
        e7c.e7_curve_data(WEEK6, tmp_path)


def test_too_few_e6_runs_at_the_three_widths_is_an_error(tmp_path):
    shutil.copytree(WEEK6 / "e6_extend_d16_seed1", tmp_path / "e6_extend_d16_seed1")
    with pytest.raises(ValueError, match="expected 9 E6 runs at widths 16, 32 and 64, found 1"):
        e7c.e7_curve_data(tmp_path, WEEK7)


def copy_run_with_protocol_change(tmp_path, name: str, change):
    """Copy one E7 run into tmp_path and apply `change` to the first run of its protocol.json; returns the week folder holding it."""
    week = tmp_path / "week"
    shutil.copytree(WEEK7 / name, week / name)
    path = week / name / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    change(protocol["runs"][0])
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return week


@pytest.mark.parametrize("name, change, message", [
    ("e7_w_d16_seed1", lambda run: run.pop("weight_decay"), "arm W .*weight_decay"),
    ("e7_w_d16_seed1", lambda run: run.update(weight_decay=0.1), "arm W .*weight_decay"),
    ("e7_w_d16_seed1", lambda run: run.update(lr_schedule="step:2000:0.1"), "arm W .*lr_schedule"),
    ("e7_s_d16_seed1", lambda run: run.pop("lr_schedule"), "arm S .*lr_schedule"),
    ("e7_s_d16_seed1", lambda run: run.update(lr_schedule="step:1000:0.1"), "arm S .*lr_schedule"),
    ("e7_s_d16_seed1", lambda run: run.update(weight_decay=0.0), "arm S .*weight_decay"),
    ("e7_w_d16_seed1", lambda run: run.update(epochs=3000), "4000 epochs"),
    ("e7_w_d16_seed1", lambda run: run["model"].update(embedding_dim=32), "declares width 32"),
])
def test_a_run_whose_protocol_does_not_match_its_folder_name_is_an_error(tmp_path, name, change, message):
    week = copy_run_with_protocol_change(tmp_path, name, change)
    with pytest.raises(ValueError, match=message):
        e7c.load_e7_runs(week)


def test_an_unmodified_copy_of_each_arm_loads(tmp_path):
    week = copy_run_with_protocol_change(tmp_path, "e7_w_d16_seed1", lambda run: None)
    shutil.copytree(WEEK7 / "e7_s_d16_seed1", week / "e7_s_d16_seed1")
    assert sorted(e7c.load_e7_runs(week)) == [("S", 16, 1), ("W", 16, 1)]


def test_a_folder_without_exactly_one_run_is_an_error(tmp_path):
    week = tmp_path / "week"
    shutil.copytree(WEEK7 / "e7_w_d16_seed1", week / "e7_w_d16_seed1")
    path = week / "e7_w_d16_seed1" / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["runs"] = protocol["runs"] * 2
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="e7_w_d16_seed1: expected 1 run, found 2"):
        e7c.load_e7_runs(week)


def test_the_command_line_writes_the_table_and_refuses_to_overwrite(tmp_path, capsys):
    e7c.main(["--out", str(tmp_path)])
    assert (tmp_path / "fig_e7_curves.csv").is_file() and "wrote" in capsys.readouterr().out
    with pytest.raises(SystemExit) as caught:
        e7c.main(["--out", str(tmp_path)])
    assert "already exists" in str(caught.value)
    e7c.main(["--out", str(tmp_path), "--force"])


# ---- the numbers behind the figures --------------------------------------------------------------------------------------------------------
def raw_dip(rows: list[tuple[int, int]], n: int, settled: int = 50):
    """The fewest entries recalled after the run first sat at 100% for `settled` consecutive logged epochs; None if it never dipped.
    The rule of report/make_numbers.py (SETTLED_RUN = 50), written out again here."""
    run = 0
    for i, (_, recalled) in enumerate(rows):
        run = run + 1 if recalled == n else 0
        if run == settled:
            start = rows[i - settled + 1][0]
            later = [k for e, k in rows if e > start and k < n]
            return min(later) if later else None
    return None


def raw_dip_epoch(rows: list[tuple[int, int]], n: int, settled: int = 50):
    """The first epoch below 100% after the run first sat at 100% for `settled` consecutive logged epochs; None if there is none."""
    run = 0
    for i, (_, recalled) in enumerate(rows):
        run = run + 1 if recalled == n else 0
        if run == settled:
            start = rows[i - settled + 1][0]
            later = [e for e, k in rows if e > start and k < n]
            return later[0] if later else None
    return None


def test_the_curve_summary_is_the_mean_and_the_range_over_the_three_seeds_of_the_recalled_share(table, raw):
    summary = figures.curve_summary(table)
    assert list(summary) == [(arm, d) for arm in ARMS for d in WIDTHS]
    for (arm, d), s in summary.items():
        n = raw[(arm, d, 1)][0]
        per_seed = [[k / n for _, k in raw[(arm, d, seed)][1]] for seed in SEEDS]
        assert s["entries"] == n and s["epochs"] == [e for e, _ in raw[(arm, d, 1)][1]]
        assert s["mean"] == pytest.approx([statistics.fmean(v) for v in zip(*per_seed)], abs=1e-12)
        assert s["low"] == [min(v) for v in zip(*per_seed)] and s["high"] == [max(v) for v in zip(*per_seed)]
        dips = [x for seed in SEEDS if (x := raw_dip(raw[(arm, d, seed)][1], n)) is not None]
        assert s["dip_min"] == (min(dips) if dips else None)
        firsts = [x for seed in SEEDS if (x := raw_dip_epoch(raw[(arm, d, seed)][1], n)) is not None]
        assert s["dip_epoch"] == (min(firsts) if firsts else None)


def test_the_only_dips_after_settling_are_arm_w_at_width_64_and_the_report_numbers_agree(table):
    summary = figures.curve_summary(table)
    assert {key: s["dip_min"] for key, s in summary.items() if s["dip_min"] is not None} == {("W", 64): 25}
    numbers = (ROOT / "report" / "numbers.tex").read_text(encoding="utf-8")
    assert [re.search(rf"\\numdef{{e7w\.dipmin\.w64\.s{s}}}{{(\w+)}}", numbers).group(1) for s in SEEDS] == ["390", "25", "none"]


def raw_passes(raw, arm: str, d: int) -> bool:
    """Two or more of the three seeds had every entry recalled at some logged epoch."""
    return sum(any(k == n for _, k in rows) for seed in SEEDS for n, rows in [raw[(arm, d, seed)]]) >= 2


def extensions():
    return {arm: json.loads((RESULTS / name).read_text(encoding="utf-8"))
            for arm, name in (("E6", "e6_extension.json"), ("W", "e7w_extension.json"), ("S", "e7s_extension.json"))}


def test_the_capacity_figure_data_is_the_documented_result():
    data = figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), extensions())
    assert data["widths"] == [8, 16, 32, 64] and data["budgets"] == [100, 300, 500, 1000]
    assert [data["e4"][d][1000] for d in data["widths"]] == [14, 113, 226, 453]
    assert [data["e4"][d][500] for d in data["widths"]] == [None, 40, 226, 320]
    assert data["e6"] == {8: (20, True), 16: (160, True), 32: (320, True), 64: (640, False)}
    assert data["e7"] == {"W": {16: (160, True), 32: (320, True), 64: (640, True)}, "S": {16: (160, True), 32: (320, True), 64: (640, True)}}
    assert data["train_epochs"] == 4000


def test_every_pass_flag_in_the_capacity_figure_is_what_the_raw_curves_say(raw):
    data = figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), extensions())
    for d in WIDTHS:
        assert data["e6"][d][1] == raw_passes(raw, "E6", d)
        for arm in ("W", "S"):
            assert data["e7"][arm][d][1] == raw_passes(raw, arm, d)


def test_the_arm_s_step_down_epoch_in_the_figure_is_the_one_in_the_protocol():
    protocol = json.loads((WEEK7 / "e7_s_d16_seed1" / "protocol.json").read_text(encoding="utf-8"))
    kind, step, factor = protocol["runs"][0]["lr_schedule"].split(":")
    assert (kind, int(step), float(factor)) == ("step", figures.S_CUT_EPOCH, 0.1)


# ---- the figures -----------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built(table):
    capacity = figures.build_capacity_figure(figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), extensions()))
    curves = figures.build_curves_figure(figures.curve_summary(table))
    return {"capacity": capacity, "curves": curves}


@pytest.mark.parametrize("name", ["capacity", "curves"])
def test_a_figure_fits_the_text_width_and_every_label_is_at_least_8_points(built, name):
    from matplotlib.text import Text
    fig = built[name]
    fig.canvas.draw()                                                          # tick labels exist only once the figure is drawn
    assert fig.get_size_inches()[0] <= 5.5
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible() and t.get_fontsize() < 8]
    assert small == []


def test_the_curves_figure_plots_the_mean_and_the_range_of_each_arm_at_each_width(built, table):
    summary = figures.curve_summary(table)
    axes = built["curves"].axes
    assert len(axes) == 3
    for ax, d in zip(axes, WIDTHS):
        lines = {line.get_label(): line for line in ax.lines if line.get_label() in ("E6", "arm W", "arm S")}
        assert sorted(lines) == ["E6", "arm S", "arm W"]
        for label, arm in (("E6", "E6"), ("arm W", "W"), ("arm S", "S")):
            assert list(lines[label].get_xdata()) == summary[(arm, d)]["epochs"]
            assert list(lines[label].get_ydata()) == pytest.approx(summary[(arm, d)]["mean"], abs=1e-12)
        bands = {c.get_label(): c for c in ax.collections if c.get_label() in ("E6 range", "arm W range", "arm S range")}
        assert sorted(bands) == ["E6 range", "arm S range", "arm W range"]
        for label, arm in (("E6 range", "E6"), ("arm W range", "W"), ("arm S range", "S")):
            ys = bands[label].get_paths()[0].vertices[:, 1]
            assert ys.max() == pytest.approx(max(summary[(arm, d)]["high"])) and ys.min() == pytest.approx(min(summary[(arm, d)]["low"]))


def test_the_curves_figure_says_in_words_the_one_dip_after_settling_that_the_axis_cuts_off(built):
    texts = [" ".join(t.get_text().split()) for ax in built["curves"].axes for t in ax.texts]    # the note wraps onto two lines
    assert [t for t in texts if "down to" in t] == ["arm W: down to 25 of 640"]


def test_the_capacity_figure_plots_the_capacities_and_marks_the_4000_epoch_rungs(built):
    data = figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), extensions())
    (ax,) = built["capacity"].axes
    by_label = {line.get_label(): line for line in ax.lines}
    for budget in data["budgets"]:
        points = [(d, data["e4"][d][budget]) for d in data["widths"] if data["e4"][d][budget] is not None]
        line = by_label[f"{budget:,} epochs"]
        assert list(zip(line.get_xdata(), line.get_ydata())) == points
    for label, entries in (("E6 passed", {8: 20, 16: 160, 32: 320}), ("E7 arm W passed", {16: 160, 32: 320, 64: 640}), ("E7 arm S passed", {16: 160, 32: 320, 64: 640})):
        marks = by_label[label]
        assert [round(y) for y in marks.get_ydata()] == [entries[d] for d in sorted(entries)]
    failed = by_label["E6 did not pass"]
    assert [round(y) for y in failed.get_ydata()] == [640]


# ---- the PDF files ---------------------------------------------------------------------------------------------------------------------------
def test_a_saved_figure_is_a_vector_pdf_with_embedded_truetype_fonts_no_creation_date_and_a_page_within_the_text_width(built, tmp_path):
    path = tmp_path / "f.pdf"
    figures.save_pdf(built["curves"], path)
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and b"/CreationDate" not in data
    assert b"/Subtype /Type3" not in data and b"/FontFile2" in data              # embedded TrueType programs, no bitmap-like Type 3 glyphs
    width = float(re.search(rb"/MediaBox \[ *0 +0 +([\d.]+) +([\d.]+) *\]", data).group(1))
    assert width <= 5.5 * 72 + 0.01


def test_saving_the_same_figure_twice_gives_the_same_bytes(built, tmp_path):
    figures.save_pdf(built["capacity"], tmp_path / "a.pdf")
    figures.save_pdf(built["capacity"], tmp_path / "b.pdf")
    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()


def test_write_all_writes_both_pdfs_refuses_to_overwrite_and_replaces_with_force(tmp_path):
    written = figures.write_all(tmp_path)
    assert sorted(p.name for p in written) == ["fig_capacity.pdf", "fig_e7_curves.pdf"]
    with pytest.raises(ValueError, match="already exists"):
        figures.write_all(tmp_path)
    figures.write_all(tmp_path, force=True)


def test_the_module_runs_as_a_command(tmp_path):
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-m", "report.figures", "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and (tmp_path / "fig_capacity.pdf").is_file(), done.stdout + done.stderr


def test_the_committed_figures_are_vector_pdfs_with_the_page_size_of_a_fresh_render(tmp_path):
    figures.write_all(tmp_path)
    for name in ("fig_capacity.pdf", "fig_e7_curves.pdf"):
        committed, fresh = (ROOT / "report" / "figures" / name).read_bytes(), (tmp_path / name).read_bytes()
        assert re.search(rb"/MediaBox \[[^\]]*\]", committed).group(0) == re.search(rb"/MediaBox \[[^\]]*\]", fresh).group(0)
        assert len(committed) > 1000 and committed.startswith(b"%PDF")


# ---- found in review -------------------------------------------------------------------------------------------------------------------------
def test_the_settled_run_is_the_one_of_the_report_numbers():
    assert figures.SETTLED_RUN == 50


@pytest.mark.parametrize("recalled, expected", [
    ([10] * 49 + [3] + [10] * 10, None),                  # never sat at 100% for 50 logged epochs in a row
    ([3] + [10] * 50, None),                              # a dip before settling does not count; none after
    ([10] * 55, None),                                    # settled, never dipped
    ([5] + [10] * 50 + [4, 10], 4),                       # settled at the 51st logged epoch, then dipped to 4
    ([10] * 50 + [9], 9),                                 # a dip right after the 50th epoch at 100%
    ([10] * 50 + [9, 2, 10], 2),                          # the lowest of several dips
    ([10] * 30 + [1] + [10] * 50 + [6] + [10] * 5, 6),    # the first run was broken at 30, the settled run starts again after it
])
def test_dip_min_is_the_fewest_recalled_after_the_run_settled(recalled, expected):
    assert figures.dip_min(recalled, 10) == expected


def table_rows_with(rows, change):
    out = [dict(r) for r in rows]
    change(out)
    return out


def test_a_curve_with_a_repeated_or_out_of_order_epoch_is_an_error(table):
    firsts = [r for r in table if r["arm"] == "E6" and r["model_d"] == "16" and r["epoch"] == "10"]
    assert len(firsts) == 3
    with pytest.raises(ValueError, match="E6.*16.*increasing"):
        figures.curve_summary(table + [dict(r) for r in firsts])


def test_arms_that_disagree_on_the_number_of_entries_at_a_width_are_an_error(table):
    def change(rows):
        for r in rows:
            if r["arm"] == "W" and r["model_d"] == "16":
                r["entries"] = "161"
    with pytest.raises(ValueError, match="arms disagree.*16"):
        figures.curve_summary(table_rows_with(table, change))


def test_a_width_with_two_seeds_is_an_error(table):
    rows = [r for r in table if not (r["arm"] == "S" and r["model_d"] == "32" and r["seed"] == "3")]
    with pytest.raises(ValueError, match="expected 3 seeds"):
        figures.curve_summary(rows)


def test_more_epoch_budgets_than_colors_is_an_error_not_a_silently_missing_line():
    data = figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), extensions())
    data["budgets"] = data["budgets"] + [2000]
    for per_width in data["e4"].values():
        per_width[2000] = None
    with pytest.raises(ValueError, match="5 epoch budgets"):
        figures.build_capacity_figure(data)


def test_every_plotted_capacity_point_is_inside_the_axes(built):
    (ax,) = built["capacity"].axes
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for line in ax.lines:
        assert all(x0 <= x <= x1 for x in line.get_xdata()) and all(y0 <= y <= y1 for y in line.get_ydata()), line.get_label()


def test_a_failure_while_drawing_the_second_figure_leaves_no_file_behind(tmp_path, monkeypatch):
    def boom(summary):
        raise ValueError("cannot draw")
    monkeypatch.setattr(figures, "build_curves_figure", boom)
    with pytest.raises(ValueError, match="cannot draw"):
        figures.write_all(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_successful_write_leaves_no_temporary_file(tmp_path):
    figures.write_all(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fig_capacity.pdf", "fig_e7_curves.pdf"]


def test_the_command_line_reports_a_missing_results_folder_as_an_error_message(tmp_path):
    with pytest.raises(SystemExit) as caught:
        figures.main(["--out", str(tmp_path / "out"), "--results", str(tmp_path / "missing")])
    assert str(caught.value).startswith("error: ") and "missing" in str(caught.value)


def test_an_arm_with_no_rows_at_a_width_is_an_error(table):
    rows = [r for r in table if not (r["arm"] == "S" and r["model_d"] == "16")]
    with pytest.raises(ValueError, match="no rows for S at width 16"):
        figures.curve_summary(rows)


def test_an_e4_table_that_lacks_a_budget_at_one_width_is_an_error():
    rows = [r for r in read_table(RESULTS / "fig_e4_capacity.csv") if not (r["source"] == "E4" and r["model_d"] == "16" and r["epochs"] == "300")]
    with pytest.raises(ValueError, match="does not have every budget at every width"):
        figures.capacity_figure_data(rows, extensions())


def test_extension_tables_that_end_at_different_epochs_are_an_error():
    ext = extensions()
    for width in ext["W"].values():
        width["passes"].pop("4000")
    with pytest.raises(ValueError, match=r"disagree on the training length: \[3000, 4000\]"):
        figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), ext)


def test_an_e6_table_with_other_widths_than_e4_is_an_error():
    ext = extensions()
    del ext["E6"]["8"]
    with pytest.raises(ValueError, match=r"E4 widths \[8, 16, 32, 64\] and E6 widths \[16, 32, 64\] differ"):
        figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), ext)


# ---- what the figures say in words ---------------------------------------------------------------------------------------------------------
def test_the_curves_figure_axis_starts_where_its_footnote_says(built):
    fig = built["curves"]
    assert [ax.get_ylim()[0] for ax in fig.axes] == [0.9, 0.9, 0.9] and all(ax.get_ylim()[1] > 1.0 for ax in fig.axes)
    assert [t.get_text() for t in fig.texts if "y axis starts" in t.get_text()] == [
        "Line: mean of 3 seeds. Band: lowest to highest seed. The y axis starts at 90%."]
    assert all(ax.get_xlim() == (0, 4000) for ax in fig.axes)


def test_the_curves_legend_states_each_arms_setting_and_the_settings_are_the_protocols():
    from phonebook.make_capacity_spec import E6_WEIGHT_DECAY
    w = json.loads((WEEK7 / "e7_w_d16_seed1" / "protocol.json").read_text(encoding="utf-8"))["runs"][0]["weight_decay"]
    s = json.loads((WEEK7 / "e7_s_d16_seed1" / "protocol.json").read_text(encoding="utf-8"))["runs"][0]["lr_schedule"]
    assert (E6_WEIGHT_DECAY, w, s) == (0.1, 0.0, "step:2000:0.1")
    fig = figures.build_curves_figure(figures.curve_summary(read_table(RESULTS / "fig_e7_curves.csv")))
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["E6 (weight decay 0.1)", "arm W (weight decay 0)",
                                                            "arm S (rate x0.1 after epoch 2,000)", "arm S step-down"]


def test_the_capacity_legends_name_the_budgets_and_the_marks(built):
    first, second = built["capacity"].legends
    assert [t.get_text() for t in first.get_texts()] == ["100 epochs", "300 epochs", "500 epochs", "1,000 epochs"]
    assert first.get_title().get_text() == "E4 capacity by training length"
    assert [t.get_text() for t in second.get_texts()] == ["E6 passed", "E6 did not pass", "E7 arm W passed", "E7 arm S passed"]
    assert second.get_title().get_text() == "4,000-epoch rungs"


def test_each_arm_has_the_same_color_in_both_figures_and_it_is_not_one_of_the_epoch_blues(built):
    from matplotlib.colors import to_hex
    (cap_ax,) = built["capacity"].axes
    cap = {line.get_label(): to_hex(line.get_color()) for line in cap_ax.lines}
    cur = {line.get_label(): to_hex(line.get_color()) for line in built["curves"].axes[0].lines}
    assert cap["E6 passed"] == cap["E6 did not pass"] == cur["E6"]
    assert cap["E7 arm W passed"] == cur["arm W"] and cap["E7 arm S passed"] == cur["arm S"]
    assert len({cur["E6"], cur["arm W"], cur["arm S"]}) == 3
    blues = {cap[f"{b:,} epochs"] for b in (100, 300, 500, 1000)}
    assert len(blues) == 4 and not blues & {cur["E6"], cur["arm W"], cur["arm S"]}


def test_the_three_marks_at_one_width_sit_side_by_side_in_the_order_e6_w_s(built):
    fig = built["capacity"]
    fig.canvas.draw()
    (ax,) = fig.axes
    by_label = {line.get_label(): line for line in ax.lines}
    x = {name: by_label[label].get_transform().transform((16, 160))[0]
         for name, label in (("E6", "E6 passed"), ("W", "E7 arm W passed"), ("S", "E7 arm S passed"))}
    points = 72 / fig.dpi                                                      # points per pixel
    assert 6 <= (x["W"] - x["E6"]) * points <= 12 and 6 <= (x["S"] - x["W"]) * points <= 12


# ---- gaps found by the planted-bug run -----------------------------------------------------------------------------------------------------
def test_a_break_in_the_run_starts_the_count_again_from_zero():
    assert figures.dip_min([10] * 10 + [3] + [10] * 49 + [2], 10) is None       # 49 in a row after the break is not settled
    assert figures.dip_min([10] * 10 + [3] + [10] * 50 + [2], 10) == 2


def synthetic_rows(dips=None, entries=10, count=70):
    """Rows for every arm at width 16 and seeds 1 to 3, epochs 10 to 10 * count, all at 100%; `dips` is {(arm, seed): {index: recalled}}."""
    dips = dips or {}
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            recalled = [entries] * count
            for index, value in dips.get((arm, seed), {}).items():
                recalled[index] = value
            rows += [{"arm": arm, "model_d": "16", "entries": str(entries), "seed": str(seed), "epoch": str(10 * (i + 1)), "recalled": str(k)}
                     for i, k in enumerate(recalled)]
    return rows


def test_the_dip_summary_takes_the_deepest_dip_and_the_earliest_epoch_over_the_seeds():
    s = figures.curve_summary(synthetic_rows(dips={("W", 3): {60: 4}}))
    assert (s[("W", 16)]["dip_min"], s[("W", 16)]["dip_epoch"]) == (4, 610)                      # only the third seed dips
    s = figures.curve_summary(synthetic_rows(dips={("W", 1): {52: 8}, ("W", 2): {60: 3}, ("W", 3): {65: 6}}))
    assert (s[("W", 16)]["dip_min"], s[("W", 16)]["dip_epoch"]) == (3, 530)                      # deepest: seed 2; earliest: seed 1
    assert (s[("E6", 16)]["dip_min"], s[("E6", 16)]["dip_epoch"]) == (None, None)


def test_rows_given_in_another_order_of_runs_give_the_same_summary(table):
    groups = {}
    for row in table:
        groups.setdefault((row["arm"], row["model_d"], row["seed"]), []).append(row)
    shuffled = [row for key in reversed(list(groups)) for row in groups[key]]
    a, b = figures.curve_summary(table), figures.curve_summary(shuffled)
    assert list(a) == list(b) and a == b


def test_one_row_with_another_number_of_entries_is_an_error(table):
    rows = [dict(r) for r in table]
    next(r for r in rows if r["arm"] == "W" and r["model_d"] == "16" and r["epoch"] == "200")["entries"] = "161"
    with pytest.raises(ValueError, match="rows disagree on the number of entries"):
        figures.curve_summary(rows)


def test_a_fourth_seed_or_a_seed_with_other_epochs_is_an_error(table):
    extra = [dict(r, seed="4") for r in table if r["arm"] == "E6" and r["model_d"] == "16" and r["seed"] == "1"]
    with pytest.raises(ValueError, match="expected 3 seeds"):
        figures.curve_summary(table + extra)
    short = [r for r in table if not (r["arm"] == "E6" and r["model_d"] == "16" and r["seed"] == "3" and r["epoch"] == "200")]
    with pytest.raises(ValueError, match="expected 3 seeds"):
        figures.curve_summary(short)


def test_an_epoch_repeated_next_to_itself_is_an_error(table):
    rows = []
    for r in table:
        rows.append(r)
        if r["arm"] == "S" and r["model_d"] == "64" and r["epoch"] == "100":
            rows.append(dict(r))
    with pytest.raises(ValueError, match="increasing"):
        figures.curve_summary(rows)


def test_an_e4_row_from_another_source_is_ignored():
    rows = read_table(RESULTS / "fig_e4_capacity.csv")
    other = rows + [{"model_d": "16", "params": "9216", "epochs": "300", "source": "E5", "entries": "999", "meaning": "x"}]
    assert figures.capacity_figure_data(other, extensions()) == figures.capacity_figure_data(rows, extensions())


@pytest.mark.parametrize("budget", ["100", "300", "500", "1000"])
def test_an_e4_table_missing_any_one_budget_at_a_width_is_an_error(budget):
    rows = [r for r in read_table(RESULTS / "fig_e4_capacity.csv") if not (r["source"] == "E4" and r["model_d"] == "16" and r["epochs"] == budget)]
    with pytest.raises(ValueError, match="does not have every budget at every width"):
        figures.capacity_figure_data(rows, extensions())


@pytest.mark.parametrize("width", ["8", "16", "32", "64"])
def test_an_e6_table_missing_any_one_width_is_an_error(width):
    ext = extensions()
    del ext["E6"][width]
    with pytest.raises(ValueError, match=r"E4 widths .* and E6 widths .* differ"):
        figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), ext)


def test_a_failed_e7_rung_is_refused_not_left_out():
    for arm, width in (("W", "64"), ("S", "16")):
        ext = extensions()
        ext[arm][width]["passes"]["4000"] = False
        data = figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), ext)
        with pytest.raises(ValueError, match=rf"E7 arm {arm} did not pass at width \[{width}\]"):
            figures.build_capacity_figure(data)


def test_the_arm_marks_use_their_own_arms_rungs():
    import matplotlib.pyplot as plt
    data = figures.capacity_figure_data(read_table(RESULTS / "fig_e4_capacity.csv"), extensions())
    data["e7"]["W"][16] = (161, True)
    fig = figures.build_capacity_figure(data)
    try:
        by_label = {line.get_label(): line for line in fig.axes[0].lines}
        assert [round(y) for y in by_label["E7 arm W passed"].get_ydata()] == [161, 320, 640]
        assert [round(y) for y in by_label["E7 arm S passed"].get_ydata()] == [160, 320, 640]
    finally:
        plt.close(fig)


def drawn(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def test_every_tick_label_axis_label_and_title_says_what_it_should(built):
    cap, cur = built["capacity"], built["curves"]
    drawn(cap), drawn(cur)
    (ax,) = cap.axes
    assert [t.get_text() for t in ax.get_xticklabels()] == ["8", "16", "32", "64"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["10", "20", "40", "80", "160", "320", "640"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("embedding width d", "phonebook entries")
    assert all([t.get_text() for t in a.get_xticklabels()] == ["0", "2,000", "4,000"] for a in cur.axes)
    assert [t.get_text() for t in cur.axes[0].get_yticklabels()] == ["90%", "95%", "100%"]
    assert [a.get_yticklabels() for a in cur.axes[1:]] == [[], []]                 # the inner panels share the y axis and do not repeat it
    assert [a.get_ylabel() for a in cur.axes] == ["entries recalled", "", ""] and {a.get_xlabel() for a in cur.axes} == {"epoch"}
    assert [a.get_title(loc="left") for a in cur.axes] == ["d = 16: 160 entries", "d = 32: 320 entries", "d = 64: 640 entries"]
    assert [a.get_ylim() for a in cur.axes] == [(0.9, 1.012)] * 3


def test_the_curve_panels_draw_the_100_percent_line_and_the_step_down_line(built):
    for ax in built["curves"].axes:
        guides = [line for line in ax.lines if line.get_label().startswith("_")]
        assert any(list(line.get_ydata()) == [1.0, 1.0] for line in guides)
        assert any(list(line.get_xdata()) == [figures.S_CUT_EPOCH] * 2 for line in guides)


def test_the_palette_is_fixed_and_the_budget_lines_follow_the_ramp_in_order(built):
    from matplotlib.colors import to_hex
    assert figures.COLORS == {"E6": "#52514e", "W": "#eb6834", "S": "#1baf7a"}
    (ax,) = built["capacity"].axes
    assert [to_hex(line.get_color()) for line in ax.lines if line.get_label().endswith(" epochs")] == [to_hex(c) for c in figures.BUDGET_RAMP]


def test_every_plotted_point_of_the_capacity_figure_is_inside_the_axes_with_the_side_by_side_offsets(built):
    fig = built["capacity"]
    drawn(fig)
    (ax,) = fig.axes
    box = ax.bbox
    for line in ax.lines:
        xy = line.get_transform().transform(list(zip(line.get_xdata(), line.get_ydata())))
        inside = (xy[:, 0] >= box.x0 + 3).all() and (xy[:, 0] <= box.x1 - 3).all() and (xy[:, 1] >= box.y0).all() and (xy[:, 1] <= box.y1).all()
        assert inside, line.get_label()


def test_nothing_is_drawn_outside_the_page_and_no_legend_or_note_overlaps_a_plot(built):
    for name, fig in built.items():
        renderer = drawn(fig)
        width, height = fig.get_size_inches()
        tight = fig.get_tightbbox(renderer)
        assert tight.x0 >= 0 and tight.y0 >= 0 and tight.x1 <= width and tight.y1 <= height, name
        boxes = [ax.get_tightbbox(renderer) for ax in fig.axes] + [leg.get_window_extent(renderer) for leg in fig.legends] \
            + [t.get_window_extent(renderer) for t in fig.texts]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert not (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1), (name, a, b)


def test_the_dip_note_sits_left_of_the_dip_and_low_inside_its_panel(built, table):
    summary = figures.curve_summary(table)
    (note,) = [t for ax in built["curves"].axes for t in ax.texts]
    x, y = note.get_position()
    assert 0 < x < summary[("W", 64)]["dip_epoch"] and 0.9 <= y < 0.92


def test_write_all_accepts_text_paths_and_creates_missing_folders(tmp_path):
    written = figures.write_all(str(tmp_path / "a" / "b"), results=str(RESULTS))
    assert [p.name for p in written] == ["fig_capacity.pdf", "fig_e7_curves.pdf"] and all(p.is_file() for p in written)


def test_no_figure_is_left_open_after_writing_or_failing(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    before = set(plt.get_fignums())
    figures.write_all(tmp_path)
    assert set(plt.get_fignums()) == before

    def boom(summary):
        raise ValueError("cannot draw")
    monkeypatch.setattr(figures, "build_curves_figure", boom)
    with pytest.raises(ValueError, match="cannot draw"):
        figures.write_all(tmp_path / "again")
    assert set(plt.get_fignums()) == before


def test_the_default_output_folder_is_report_figures_and_the_command_prints_what_it_wrote(tmp_path, monkeypatch, capsys):
    assert figures.FIGURES_DIR == ROOT / "report" / "figures"
    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path / "default_out")
    figures.main([])
    assert sorted(p.name for p in (tmp_path / "default_out").iterdir()) == ["fig_capacity.pdf", "fig_e7_curves.pdf"]
    assert capsys.readouterr().out.count("wrote ") == 2


@pytest.mark.parametrize("script", ["report/figures.py", "phonebook/report_e7_curves.py"])
def test_each_module_also_runs_as_a_script_from_another_folder(tmp_path, script):
    import subprocess
    import sys
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, str(ROOT / script), "--out", str(out)], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.count("wrote ") >= 1, done.stdout + done.stderr
    assert any(out.iterdir())


# ---- the same, for the E7 table ---------------------------------------------------------------------------------------------------------
def test_the_table_command_needs_an_output_folder_and_prefixes_its_errors(tmp_path):
    with pytest.raises(SystemExit) as caught:
        e7c.main([])
    assert caught.value.code == 2
    e7c.main(["--out", str(tmp_path)])
    with pytest.raises(SystemExit) as caught:
        e7c.main(["--out", str(tmp_path)])
    assert str(caught.value).startswith("error: ")


def test_the_table_writer_accepts_a_text_path_and_creates_missing_folders(tmp_path):
    path = e7c.write_e7_table(str(tmp_path / "x" / "y"), week_e6=WEEK6, week_e7=WEEK7)
    assert path.is_file() and path.name == "fig_e7_curves.csv"


def test_two_digit_seeds_are_read_and_a_plain_file_or_an_interrupted_folder_is_not(tmp_path):
    week = tmp_path / "week"
    shutil.copytree(WEEK7 / "e7_w_d16_seed1", week / "e7_w_d16_seed12")
    shutil.copytree(WEEK7 / "e7_w_d16_seed1", week / "e7_w_d16_seed1__interrupted")
    (week / "e7_w_d16_seed9").write_text("not a run", encoding="utf-8")
    assert list(e7c.load_e7_runs(week)) == [("W", 16, 12)]


@pytest.mark.parametrize("name, change, message", [
    ("e7_w_d32_seed1", lambda run: run["model"].update(embedding_dim=16), "declares width 16"),
    ("e7_w_d16_seed1", lambda run: run.update(epochs=5000), "5000 epochs"),
])
def test_a_run_declaring_a_smaller_width_or_a_longer_training_is_an_error(tmp_path, name, change, message):
    with pytest.raises(ValueError, match=message):
        e7c.load_e7_runs(copy_run_with_protocol_change(tmp_path, name, change))


def fake_curve(entries=10):
    return {"entries": entries, "epochs": list(range(10, 4001, 10)), "recalled": [entries] * 400}


def fake_curves():
    e7 = {(arm, w, s): fake_curve() for arm in ("W", "S") for w in WIDTHS for s in SEEDS}
    e6 = {(w, s): fake_curve() for w in (8, 16, 32, 64) for s in SEEDS}
    return e6, e7


def patch_loaders(monkeypatch, e6, e7):
    monkeypatch.setattr(e7c, "curve_data", lambda week: e6)
    monkeypatch.setattr(e7c, "load_e7_runs", lambda week: e7)


def test_a_logged_epoch_without_a_recall_count_is_an_error(tmp_path, monkeypatch):
    (tmp_path / "week" / "e7_w_d16_seed1").mkdir(parents=True)
    run = {"model": {"embedding_dim": 16}, "epochs": 4000, "weight_decay": 0.0, "n_a": 10,
           "points": [{"epoch": e, "A": 10, "B": 0} for e in range(10, 4001, 10)]}
    run["points"][5]["A"] = None
    monkeypatch.setattr(e7c, "load_runs", lambda root, budgets=(): [run])
    with pytest.raises(ValueError, match="e7_w_d16_seed1: a logged epoch has no recall count"):
        e7c.load_e7_runs(tmp_path / "week")


def test_clean_curves_are_ordered_arm_width_seed_and_the_width_8_e6_runs_are_left_out(monkeypatch):
    e6, e7 = fake_curves()
    e7[("W", 16, 1)]["recalled"][0] = 0                 # none recalled yet is fine
    patch_loaders(monkeypatch, e6, e7)
    data = e7c.e7_curve_data(Path("e6"), Path("e7"))
    assert list(data) == [(arm, w, s) for arm in ARMS for w in WIDTHS for s in SEEDS]


@pytest.mark.parametrize("change, message", [
    (lambda e6, e7: e7[("W", 16, 2)]["epochs"].pop(), "W width 16 seed 2: the logged epochs are not 10"),
    (lambda e6, e7: e6[(32, 3)].update(epochs=list(range(0, 4000, 10))), "E6 width 32 seed 3: the logged epochs are not 10"),
    (lambda e6, e7: e7[("S", 64, 1)]["recalled"].__setitem__(7, 11), r"S width 64 seed 1: a recalled count is outside 0\.\.10"),
    (lambda e6, e7: e7[("S", 64, 1)]["recalled"].__setitem__(7, -1), r"S width 64 seed 1: a recalled count is outside 0\.\.10"),
    (lambda e6, e7: e7.pop(("S", 64, 1)), "expected 18 E7 runs, found 17"),
    (lambda e6, e7: e7.update({("W", 16, 4): fake_curve()}), "expected 18 E7 runs, found 19"),
    (lambda e6, e7: e6.pop((16, 1)), "expected 9 E6 runs at widths 16, 32 and 64, found 8"),
    (lambda e6, e7: e6.update({(16, 4): fake_curve()}), "expected 9 E6 runs at widths 16, 32 and 64, found 10"),
])
def test_curves_that_are_not_what_the_table_promises_are_refused(monkeypatch, change, message):
    e6, e7 = fake_curves()
    change(e6, e7)
    patch_loaders(monkeypatch, e6, e7)
    with pytest.raises(ValueError, match=message):
        e7c.e7_curve_data(Path("e6"), Path("e7"))


# ---- gaps left by the second planted-bug run -------------------------------------------------------------------------------------------------
def test_a_dip_of_one_entry_breaks_the_run():
    assert figures.dip_min([10] * 30 + [9] + [10] * 30 + [5], 10) is None        # 30 in a row, one short, 30 again: never 50 in a row
    assert figures.dip_min([10] * 50 + [9], 10) == 9                              # but one short after 50 in a row is a dip of one entry


def test_a_row_with_fewer_entries_than_the_first_is_an_error_too(table):
    rows = [dict(r) for r in table]
    next(r for r in rows if r["arm"] == "W" and r["model_d"] == "16" and r["epoch"] == "200")["entries"] = "159"
    with pytest.raises(ValueError, match="rows disagree on the number of entries"):
        figures.curve_summary(rows)


def test_the_capacity_axes_are_log_scaled_with_the_documented_range_and_no_minor_ticks(built):
    fig = built["capacity"]
    drawn(fig)
    (ax,) = fig.axes
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")
    assert (ax.get_xlim(), ax.get_ylim()) == ((7, 75), (10, 1000))
    assert list(ax.xaxis.get_minorticklocs()) == [] and list(ax.yaxis.get_minorticklocs()) == []
