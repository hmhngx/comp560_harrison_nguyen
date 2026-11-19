"""The two figures of the report's results section, as vector PDFs that fit the text width.

    python -m report.figures [--out report/figures] [--results phonebook/results] [--force]

Reads only committed result files: fig_e4_capacity.csv (E4 capacities), fig_e7_curves.csv (the recalled count at every logged epoch of the 27
E6/E7 runs at widths 16, 32 and 64) and the extension tables e6_extension.json, e7w_extension.json and e7s_extension.json (the pass flags).
Writes two PDFs:

    fig_capacity.pdf    how many phonebook entries each width holds: E4's capacity at four epoch budgets (blue ramp), and the 4,000-epoch
                        rungs of E6 (gray), E7 arm W (orange, weight decay 0) and E7 arm S (green, learning rate x0.1 after epoch 2,000),
                        each marked passed or not passed.
    fig_e7_curves.pdf   the share of entries recalled over training at widths 16, 32 and 64: mean of the three seeds with the min-max band,
                        for E6, arm W and arm S. The y axis starts at 90%; the one dip after settling that it cuts off is written out.

Every figure is at most 5.5 in wide (the text width), every label is at least 8 pt, fonts are embedded as TrueType (pdf.fonttype 42, never
Type 3) and no creation date is written, so rendering the same data twice gives the same bytes. Existing files are never overwritten
unless --force is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.report_capacity_replay import AXIS, BUDGET_RAMP, INK, INK_2, RC, SERIES, SURFACE
from phonebook.report_e7_curves import ARMS, S_SCHEDULE, TRAIN_EPOCHS
from phonebook.report_experiments import _pyplot
from report.make_numbers import SETTLED_RUN

RESULTS_DIR = ROOT_DIR / "phonebook" / "results"
FIGURES_DIR = ROOT_DIR / "report" / "figures"

FIGURE_WIDTH_IN = 5.5                                   # the text width: a figure may not be wider
S_CUT_EPOCH = int(S_SCHEDULE.split(":")[1])             # arm S multiplies the learning rate by 0.1 after this epoch ("step:2000:0.1")
Y_LOW, Y_HIGH = 0.9, 1.012                              # the curves figure shows 90% to 100%
CAPACITY_PDF, CURVES_PDF = "fig_capacity.pdf", "fig_e7_curves.pdf"

COLORS = {"E6": INK_2, "W": SERIES[1], "S": SERIES[2]}  # gray control, orange arm W, green arm S (blue is E4's epoch ramp)
NAMES = {"E6": "E6", "W": "arm W", "S": "arm S"}
TICK_SIZE, LABEL_SIZE, SMALL = 8, 9, 8                  # points; nothing on a figure is smaller than SMALL


# ------------------------------------------------------------------ the numbers that are plotted (pure functions)


def load_table(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _after_settling(recalled: list[int], entries: int) -> list[tuple[int, int]]:
    """(index, recalled) of every logged epoch below 100% after the run first sat at 100% for SETTLED_RUN consecutive logged epochs;
    empty if the run never settled or never dipped afterwards. The rule of report/make_numbers.py (which counts from the first epoch of
    that run; the epochs in between are all at 100%, so counting from the last one gives the same dips)."""
    run = 0
    for i, value in enumerate(recalled):
        run = run + 1 if value == entries else 0
        if run == SETTLED_RUN:
            return [(j, k) for j, k in enumerate(recalled) if j > i and k < entries]
    return []


def dip_min(recalled: list[int], entries: int):
    """The fewest entries recalled after the run first settled at 100%; None if it never settled or never dipped afterwards.
    The same number as the report's e7w.dipmin keys."""
    later = _after_settling(recalled, entries)
    return min(k for _, k in later) if later else None


def curve_summary(rows: list[dict]) -> dict:
    """{(arm, width): {entries, epochs, mean, low, high, dip_min, dip_epoch}} from the rows of fig_e7_curves.csv.

    mean, low and high are, at each logged epoch, the mean, minimum and maximum over the seeds of the share of entries recalled;
    dip_min is the lowest per-seed dip_min and dip_epoch the first epoch at which any seed fell below 100% after settling
    (both None when no seed dipped after settling)."""
    runs: dict = {}
    for row in rows:
        key = (row["arm"], int(row["model_d"]))
        slot = runs.setdefault(key, {"entries": int(row["entries"]), "seeds": {}})
        if int(row["entries"]) != slot["entries"]:
            raise ValueError(f"{key}: rows disagree on the number of entries")
        seed = slot["seeds"].setdefault(int(row["seed"]), {"epochs": [], "recalled": []})
        seed["epochs"].append(int(row["epoch"]))
        seed["recalled"].append(int(row["recalled"]))
    summary = {}
    for key in sorted(runs, key=lambda k: (ARMS.index(k[0]), k[1])):
        slot = runs[key]
        seeds = [slot["seeds"][s] for s in sorted(slot["seeds"])]
        epochs = seeds[0]["epochs"]
        if len(seeds) != 3 or any(s["epochs"] != epochs for s in seeds):
            raise ValueError(f"{key}: expected 3 seeds that log the same epochs, found {len(seeds)}")
        if any(later <= earlier for earlier, later in zip(epochs, epochs[1:])):
            raise ValueError(f"{key}: the logged epochs are not in increasing order (a repeated or misplaced row?)")
        n = slot["entries"]
        shares = [[k / n for k in s["recalled"]] for s in seeds]
        later = [(epochs[j], k) for s in seeds for j, k in _after_settling(s["recalled"], n)]
        summary[key] = {"entries": n, "epochs": epochs, "mean": [statistics.fmean(v) for v in zip(*shares)],
                        "low": [min(v) for v in zip(*shares)], "high": [max(v) for v in zip(*shares)],
                        "dip_min": min(k for _, k in later) if later else None, "dip_epoch": min(e for e, _ in later) if later else None}
    for width in sorted({d for _, d in summary}):
        for arm in ARMS:
            if (arm, width) not in summary:
                raise ValueError(f"no rows for {arm} at width {width}")
        entries = {arm: summary[(arm, width)]["entries"] for arm in ARMS}
        if len(set(entries.values())) != 1:
            raise ValueError(f"the arms disagree on the number of entries at width {width}: {entries}")
    return summary


def capacity_figure_data(e4_rows: list[dict], extensions: dict) -> dict:
    """E4 capacities and the 4,000-epoch pass flags of E6 and the two E7 arms.

    e4[width][budget] is the capacity (None when no rung passed); e6[width] and e7[arm][width] are (rung, passed in train_epochs);
    `extensions` is {"E6": e6_extension.json, "W": e7w_extension.json, "S": e7s_extension.json}, loaded."""
    e4: dict = {}
    for row in e4_rows:
        if row["source"] == "E4":
            e4.setdefault(int(row["model_d"]), {})[int(row["epochs"])] = int(row["entries"]) if row["entries"] != "" else None
    widths = sorted(e4)
    budgets = sorted({b for per_width in e4.values() for b in per_width})
    if any(sorted(e4[d]) != budgets for d in widths):
        raise ValueError("the E4 capacity table does not have every budget at every width")
    epochs = {max(int(b) for b in table[d]["passes"]) for table in extensions.values() for d in table}
    if len(epochs) != 1:
        raise ValueError(f"the extension tables disagree on the training length: {sorted(epochs)}")
    (train_epochs,) = epochs

    def marks(table: dict) -> dict:
        return {int(d): (int(v["n"]), bool(v["passes"][str(train_epochs)])) for d, v in table.items()}

    e6 = marks(extensions["E6"])
    if sorted(e6) != widths:
        raise ValueError(f"E4 widths {widths} and E6 widths {sorted(e6)} differ")
    return {"widths": widths, "budgets": budgets, "e4": e4, "e6": e6, "e7": {"W": marks(extensions["W"]), "S": marks(extensions["S"])},
            "train_epochs": train_epochs}


# ------------------------------------------------------------------ drawing


def _style(ax) -> None:
    ax.set_axisbelow(True)
    ax.tick_params(length=3, labelsize=TICK_SIZE)
    ax.grid(axis="y", color=AXIS, linewidth=0.5, alpha=0.6)


def build_capacity_figure(data: dict):
    """Entries (log scale) against width (log2 scale): E4's capacity lines and the passed or not passed 4,000-epoch rungs."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from matplotlib.transforms import offset_copy
    if len(data["budgets"]) != len(BUDGET_RAMP):
        raise ValueError(f"{len(data['budgets'])} epoch budgets but {len(BUDGET_RAMP)} colors for them")
    for arm in ("W", "S"):
        failed = [d for d, (_, passed) in sorted(data["e7"][arm].items()) if not passed]
        if failed:                                              # the figure marks passed E7 rungs only; a failed one would just be missing
            raise ValueError(f"E7 arm {arm} did not pass at width {failed}: the capacity figure has no mark for a failed E7 rung")
    plt = _pyplot()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.8))
        fig.subplots_adjust(left=0.135, right=0.985, top=0.975, bottom=0.31)
        _style(ax)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        budget_lines = []
        for budget, color in zip(data["budgets"], BUDGET_RAMP):
            points = [(d, data["e4"][d][budget]) for d in data["widths"] if data["e4"][d][budget] is not None]
            (line,) = ax.plot([d for d, _ in points], [v for _, v in points], color=color, linewidth=1.4, marker="o", markersize=4.5,
                              label=f"{budget:,} epochs", zorder=3)
            budget_lines.append(line)

        shift = {"E6": -8.0, "W": 0.0, "S": 8.0}                      # points: the three arms share a width, so they sit side by side
        series = [("E6", "E6 passed", "^", True, data["e6"]), ("E6", "E6 did not pass", "X", False, data["e6"]),
                  ("W", "E7 arm W passed", "s", True, data["e7"]["W"]), ("S", "E7 arm S passed", "D", True, data["e7"]["S"])]
        mark_lines = []
        for arm, label, marker, passed, table in series:
            widths = [d for d in sorted(table) if table[d][1] == passed]
            transform = offset_copy(ax.transData, fig=fig, x=shift[arm], units="points")
            (line,) = ax.plot(widths, [table[d][0] for d in widths], linestyle="none", marker=marker,
                              markersize=7 if marker in "^X" else 6, color=COLORS[arm], markeredgecolor=SURFACE, markeredgewidth=0.8,
                              label=label, transform=transform, zorder=4, clip_on=False)
            mark_lines.append(line)
        ax.set_xlim(7, 75)
        ax.set_ylim(10, 1000)
        ax.xaxis.set_major_locator(FixedLocator(data["widths"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ticks = [10, 20, 40, 80, 160, 320, 640]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_xlabel("embedding width d", fontsize=LABEL_SIZE)
        ax.set_ylabel("phonebook entries", fontsize=LABEL_SIZE)
        fig.legend(handles=budget_lines, title="E4 capacity by training length", loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2,
                   fontsize=SMALL, title_fontsize=SMALL, columnspacing=1.0, handlelength=1.6, alignment="left")
        fig.legend(handles=mark_lines, title=f"{data['train_epochs']:,}-epoch rungs", loc="lower right", bbox_to_anchor=(0.99, 0.0), ncol=2,
                   fontsize=SMALL, title_fontsize=SMALL, columnspacing=1.0, handletextpad=0.4)
        return fig


def build_curves_figure(summary: dict):
    """Three panels (widths 16, 32, 64): mean of the seeds and the min-max band of the share recalled, for E6, arm W and arm S."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter
    plt = _pyplot()
    widths = sorted({d for _, d in summary})
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(widths), figsize=(FIGURE_WIDTH_IN, 3.1), sharex=True, sharey=True, squeeze=False)
        axes = list(axes[0])
        fig.subplots_adjust(left=0.125, right=0.965, top=0.9, bottom=0.37, wspace=0.22)
        for col, (ax, d) in enumerate(zip(axes, widths)):
            _style(ax)
            ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=1)
            ax.axvline(S_CUT_EPOCH, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
            for z, arm in enumerate(ARMS):
                s = summary[(arm, d)]
                ax.fill_between(s["epochs"], s["low"], s["high"], color=COLORS[arm], alpha=0.18, linewidth=0, zorder=2 + z,
                                label=f"{NAMES[arm]} range")
                ax.plot(s["epochs"], s["mean"], color=COLORS[arm], linewidth=1.1, solid_joinstyle="round", zorder=5 + z, label=NAMES[arm])
            cut = [(arm, summary[(arm, d)]) for arm in ARMS if summary[(arm, d)]["dip_min"] is not None
                   and summary[(arm, d)]["dip_min"] / summary[(arm, d)]["entries"] < Y_LOW]
            for row, (arm, s) in enumerate(cut):
                ax.text(s["dip_epoch"] - 100, Y_LOW + 0.006 + 0.035 * row, f"{NAMES[arm]}: down to\n{s['dip_min']} of {s['entries']}",
                        ha="right", va="bottom", fontsize=SMALL, color=COLORS[arm], zorder=9,
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0, "alpha": 0.9})
            ax.set_xlim(0, TRAIN_EPOCHS)
            ax.set_ylim(Y_LOW, Y_HIGH)
            ax.set_title(f"d = {d}: {summary[('E6', d)]['entries']} entries", loc="left", fontsize=LABEL_SIZE, fontweight="bold", color=INK, pad=5)
            ax.set_xlabel("epoch", fontsize=LABEL_SIZE)
            ax.xaxis.set_major_locator(FixedLocator([0, S_CUT_EPOCH, TRAIN_EPOCHS]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            if col == 0:
                ax.yaxis.set_major_locator(FixedLocator([0.9, 0.95, 1.0]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
                ax.set_ylabel("entries recalled", fontsize=LABEL_SIZE)
        handles = [Line2D([], [], color=COLORS["E6"], linewidth=1.4), Line2D([], [], color=COLORS["W"], linewidth=1.4),
                   Line2D([], [], color=COLORS["S"], linewidth=1.4), Line2D([], [], color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))]
        labels = ["E6 (weight decay 0.1)", "arm W (weight decay 0)", f"arm S (rate x0.1 after epoch {S_CUT_EPOCH:,})", "arm S step-down"]
        fig.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, fontsize=SMALL, columnspacing=1.5,
                   handlelength=1.8)
        fig.text(0.125, 0.012, "Line: mean of 3 seeds. Band: lowest to highest seed. The y axis starts at 90%.", fontsize=SMALL, color=INK_2)
        return fig


def save_pdf(fig, path: Path) -> None:
    """Vector PDF with TrueType fonts (never Type 3) and no creation date, so the same figure always gives the same bytes."""
    plt = _pyplot()
    with plt.rc_context({**RC, "pdf.fonttype": 42}):
        fig.savefig(path, format="pdf", metadata={"CreationDate": None})


# ------------------------------------------------------------------ writing


def write_all(out: Path, *, results: Path = RESULTS_DIR, force: bool = False) -> list[Path]:
    out = Path(out)
    targets = [out / CAPACITY_PDF, out / CURVES_PDF]
    existing = [t.name for t in targets if t.exists()]
    if existing and not force:
        raise ValueError(f"{', '.join(existing)} already exists in {out}; refusing to overwrite (use --force)")
    results = Path(results)
    extensions = {arm: load_json(results / name) for arm, name in (("E6", "e6_extension.json"), ("W", "e7w_extension.json"), ("S", "e7s_extension.json"))}
    capacity = capacity_figure_data(load_table(results / "fig_e4_capacity.csv"), extensions)
    summary = curve_summary(load_table(results / "fig_e7_curves.csv"))
    plt = _pyplot()
    figs = []
    try:
        figs.append(build_capacity_figure(capacity))                                # both are drawn before anything is written
        figs.append(build_curves_figure(summary))
        out.mkdir(parents=True, exist_ok=True)
        for fig, target in zip(figs, targets):
            save_pdf(fig, target)
    finally:
        for fig in figs:
            plt.close(fig)
    return targets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--force", action="store_true", help="replace existing figure files")
    args = parser.parse_args(argv)
    try:
        written = write_all(args.out, results=args.results, force=args.force)
    except (ValueError, OSError) as error:                    # a missing or unreadable file, or malformed input
        sys.exit(f"error: {error}")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
