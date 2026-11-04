"""Figures for E4 (capacity), E5 (replay ratio) and E6 (longer training), each with the table it plots.

    python phonebook/report_capacity_replay.py --out phonebook/results [--force]

Reads only committed files: the result tables in phonebook/results/ (e4_capacity.json, e6_extension.json, e5_cells.csv) and the
metrics of the 12 E6 evidence roots. Writes, for each of fig_e4_capacity, fig_e4_passmap, fig_e5_replay_k and fig_e6_curves, a PNG and
the CSV table that was plotted (LF line endings), so every value can be read without trusting pixels. Existing files are never
overwritten unless --force is given.

Tables (one row per plotted mark):
- fig_e4_capacity.csv  model_d, params, epochs, source (E4 or E6), entries, meaning: capacity (largest rung that passes with every smaller
                       rung; blank when no rung passed), at_least (E6 rung that passed: the capacity is at least this) or
                       rung_not_passed (the E6 rung did not pass in 4000 epochs).
- fig_e4_passmap.csv   model_d, epochs (budget), rung, seeds_passed, seeds, rung_passes (2 or more seeds), is_capacity_rung.
- fig_e5_replay_k.csv  lr, phonebook ("median" is the median of the phonebooks), k, stable_joint (updates), min_a (fewest entries of A kept).
- fig_e6_curves.csv    model_d, entries, seed, epoch, recalled (entries recalled exactly at that logged epoch).

Colors are the dataviz skill's validated palette: the four epoch budgets are an ordinal blue ramp (validated with --ordinal), the one series
that is the point is blue with every other series gray (emphasis), and the pass-map is a four-step sequential ramp. Plots avoid dual axes, a
number on every point and decoration; the CSV tables are the table view that the low-contrast aqua slot would otherwise require.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.analyze_replay_k import phonebook_medians
from phonebook.report_experiments import _pyplot, load_runs

RESULTS_DIR = ROOT_DIR / "phonebook" / "results"
WEEK_DIR = ROOT_DIR / "phonebook" / "experiments" / "week_2026-10-31"

DPI = 150
FIGSIZES = {"fig_e4_capacity": (11.0, 4.9), "fig_e4_passmap": (12.0, 5.3),
            "fig_e5_replay_k": (10.5, 4.5), "fig_e6_curves": (13.0, 7.8)}
TABLES = {
    "fig_e4_capacity": ["model_d", "params", "epochs", "source", "entries", "meaning"],
    "fig_e4_passmap": ["model_d", "epochs", "rung", "seeds_passed", "seeds", "rung_passes", "is_capacity_rung"],
    "fig_e5_replay_k": ["lr", "phonebook", "k", "stable_joint", "min_a"],
    "fig_e6_curves": ["model_d", "entries", "seed", "epoch", "recalled"],
}

# Chart surface and ink (dataviz palette.md, light mode), validated with scripts/validate_palette.js.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")                       # categorical slots 1 to 3 (valid for small multiples: --pairs all)
BUDGET_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")       # blue steps 250, 400, 550, 700: ordinal (--ordinal)
SEEDS_RAMP = ("#cde2fb", "#86b6ef", "#2a78d6", "#0d366b")        # blue steps 100, 250, 450, 700: sequential, 0 to 3 seeds
RC = {
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False, "axes.grid": False,
}
MIN_SEEDS = 2


# ------------------------------------------------------------------ loading


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_replay_cells(path: Path) -> list[dict]:
    """The cells table of E5 with real types: integers, the learning rate as a float, blanks as None, plain as a bool."""
    integers = ("phonebook", "init", "seed", "k", "stable_joint", "first_joint", "min_a", "epoch_of_min_a", "final_A", "final_B")
    flags = {"True": True, "False": False}
    cells = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            cell = {key: (None if row[key] == "" else int(row[key])) for key in integers}
            cell["lr"] = float(row["lr"])
            cell["plain"] = flags[row["plain"]]
            cell["condition"] = row["condition"]
            cells.append(cell)
    return cells


# ------------------------------------------------------------------ the numbers that are plotted (pure functions)


def capacity_data(e4: dict, e6: dict) -> dict:
    widths = sorted(int(d) for d in e4)
    if widths != sorted(int(d) for d in e6):
        raise ValueError(f"E4 widths {widths} and E6 widths {sorted(int(d) for d in e6)} differ")
    budgets = sorted(int(b) for b in e4[str(widths[0])]["capacity_by_budget"])
    e6_budget = max(int(b) for b in e6[str(widths[0])]["passes"])
    return {
        "widths": widths, "budgets": budgets, "e6_epochs": e6_budget,
        "params": {d: int(e4[str(d)]["params"]) for d in widths},
        "capacity": {d: {int(b): value for b, value in e4[str(d)]["capacity_by_budget"].items()} for d in widths},
        "e6": {d: {"rung": int(e6[str(d)]["n"]), "passes": bool(e6[str(d)]["passes"][str(e6_budget)])} for d in widths},
    }


def capacity_rows(data: dict) -> list[dict]:
    rows = []
    for d in data["widths"]:
        for budget in data["budgets"]:
            value = data["capacity"][d][budget]
            rows.append({"model_d": d, "params": data["params"][d], "epochs": budget, "source": "E4",
                         "entries": "" if value is None else value, "meaning": "capacity"})
    for d in data["widths"]:
        mark = data["e6"][d]
        rows.append({"model_d": d, "params": data["params"][d], "epochs": data["e6_epochs"], "source": "E6", "entries": mark["rung"],
                     "meaning": "at_least" if mark["passes"] else "rung_not_passed"})
    return rows


def passmap_rows(e4: dict) -> list[dict]:
    rows = []
    for d in sorted(int(key) for key in e4):
        for budget in sorted(int(b) for b in e4[str(d)]["detail"]):
            detail = e4[str(d)]["detail"][str(budget)]
            for rung in sorted(int(n) for n in detail["rungs"]):
                info = detail["rungs"][str(rung)]
                rows.append({"model_d": d, "epochs": budget, "rung": rung, "seeds_passed": int(info["passed"]),
                             "seeds": int(info["seeds"]), "rung_passes": bool(info["ok"]),
                             "is_capacity_rung": detail["capacity"] is not None and rung == int(detail["capacity"])})
    return rows


def replay_data(cells: list[dict], *, lr: float = 0.001, seed: int = 42) -> dict:
    stable = phonebook_medians(cells, lr=lr, seed=seed)
    dip = phonebook_medians(cells, lr=lr, seed=seed, field="min_a")
    phonebooks = sorted(stable)
    ks = sorted(stable[phonebooks[0]])
    for name, table in (("stable_joint", stable), ("min_a", dip)):
        for phonebook in phonebooks:
            if sorted(table[phonebook]) != ks:
                raise ValueError(f"phonebook {phonebook} lacks some of k={ks} in {name}")
            if any(math.isinf(value) for value in table[phonebook].values()):
                raise ValueError(f"phonebook {phonebook} has a run that never reached stable joint recall; it cannot be plotted")
    return {"lr": lr, "seed": seed, "phonebooks": phonebooks, "ks": ks, "stable_joint": stable, "min_a": dip,
            "median_stable_joint": {k: statistics.median([stable[p][k] for p in phonebooks]) for k in ks},
            "median_min_a": {k: statistics.median([dip[p][k] for p in phonebooks]) for k in ks}}


def replay_rows(data: dict) -> list[dict]:
    rows = [{"lr": data["lr"], "phonebook": str(p), "k": k, "stable_joint": data["stable_joint"][p][k], "min_a": data["min_a"][p][k]}
            for p in data["phonebooks"] for k in data["ks"]]
    rows += [{"lr": data["lr"], "phonebook": "median", "k": k, "stable_joint": data["median_stable_joint"][k],
              "min_a": data["median_min_a"][k]} for k in data["ks"]]
    return rows


def curve_data(week: Path) -> dict:
    """{(width, seed): {entries, epochs, recalled}} for the 12 E6 runs, read from their evidence roots."""
    pattern = re.compile(r"e6_extend_d(\d+)_seed(\d+)")
    curves = {}
    for root in sorted(Path(week).glob("e6_extend_d*_seed*")):
        match = pattern.fullmatch(root.name)
        if match is None or not root.is_dir():
            continue
        (run,) = load_runs(root, budgets=())
        width, seed = int(match.group(1)), int(match.group(2))
        if run["model"]["embedding_dim"] != width:
            raise ValueError(f"{root.name}: the run declares width {run['model']['embedding_dim']}")
        if any(point["A"] is None for point in run["points"]):
            raise ValueError(f"{root.name}: a logged epoch has no recall count")
        curves[(width, seed)] = {"entries": run["n_a"], "epochs": [p["epoch"] for p in run["points"]],
                                 "recalled": [p["A"] for p in run["points"]]}
    return dict(sorted(curves.items()))


def curve_rows(curves: dict) -> list[dict]:
    return [{"model_d": width, "entries": c["entries"], "seed": seed, "epoch": epoch, "recalled": recalled}
            for (width, seed), c in curves.items() for epoch, recalled in zip(c["epochs"], c["recalled"])]


def first_perfect(curve: dict):
    return next((e for e, v in zip(curve["epochs"], curve["recalled"]) if v == curve["entries"]), None)


def curve_note(curve: dict) -> str:
    """The line written above each E6 panel."""
    first = first_perfect(curve)
    if first is not None:
        return f"100% first at epoch {first:,}"
    return f"never 100%: best {max(curve['recalled'])} of {curve['entries']}"


def passmap_grid(rows: list[dict], width: int, budgets: list[int]):
    """(rungs ascending, matrix[rung][budget] of seeds passed, markers [(column, row)] of the capacity rung of each budget)."""
    mine = [r for r in rows if r["model_d"] == width]
    rungs = sorted({r["rung"] for r in mine})
    lookup = {(r["rung"], r["epochs"]): r for r in mine}
    matrix = [[lookup[(rung, b)]["seeds_passed"] for b in budgets] for rung in rungs]
    markers = [(j, rungs.index(rung)) for j, b in enumerate(budgets) for rung in rungs if lookup[(rung, b)]["is_capacity_rung"]]
    return rungs, matrix, markers


# ------------------------------------------------------------------ drawing


def _texts(fig) -> list[str]:
    """Every non-empty string written on the figure (titles, labels, annotations, legend), after it was drawn."""
    from matplotlib.text import Text
    return sorted({t.get_text() for t in fig.findobj(Text) if t.get_text().strip()})


def _axes_style(ax, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, linestyle="-")
    ax.tick_params(length=3, labelsize=8.5)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)


def _log2_widths(ax, widths: list[int], labels: list[str]) -> None:
    from matplotlib.ticker import NullLocator
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(widths[0] / 1.35, widths[-1] * 1.35)


def _log_entries(ax) -> None:
    from matplotlib.ticker import FuncFormatter, NullLocator
    ticks = [10, 20, 50, 100, 200, 500, 1000]
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(8, 1100)


def figure_e4_capacity(data: dict, path: Path) -> list[str]:
    from matplotlib.lines import Line2D
    plt = _pyplot()
    widths, budgets = data["widths"], data["budgets"]
    with plt.rc_context(RC):
        fig, (left, right) = plt.subplots(1, 2, figsize=FIGSIZES["fig_e4_capacity"], sharey=True,
                                          gridspec_kw={"width_ratios": [3, 2.3], "wspace": 0.06})
        fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.2)
        labels = [f"{d}\n{data['params'][d]:,}" for d in widths]
        for i, budget in enumerate(budgets):
            points = [(d, data["capacity"][d][budget]) for d in widths if data["capacity"][d][budget] is not None]
            left.plot([p[0] for p in points], [p[1] for p in points], color=BUDGET_RAMP[i], linewidth=1.8, marker="o", markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=1.2, solid_capstyle="round", zorder=3 + i, label=f"{budget} epochs")
        _log2_widths(left, widths, labels)
        _log_entries(left)
        _axes_style(left)
        left.set_ylabel("entries memorized exactly (log scale)")
        left.set_xlabel("embedding width, then parameters (2 layers, 2 heads)")
        _title(left, "E4: capacity at four training budgets")
        left.legend(title="training budget", loc="upper left", fontsize=8.5, title_fontsize=8.5, labelcolor=INK_2, handlelength=1.8)
        left.text(0.99, 0.03, "no point: no rung of the ladder passed (capacity below 10)", transform=left.transAxes, ha="right",
                  va="bottom", fontsize=8, color=MUTED)

        for d in widths:
            start, mark = data["capacity"][d][budgets[-1]], data["e6"][d]
            if mark["passes"]:
                right.plot([d, d], [start, mark["rung"]], color=SERIES[0], linewidth=1.8, solid_capstyle="round", zorder=2)
                right.plot(d, mark["rung"], marker="^", color=SERIES[0], markersize=7.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                           linestyle="none", zorder=4)
                right.annotate(f"at least {mark['rung']}", (d, mark["rung"]), xytext=(0, 8), textcoords="offset points", ha="center",
                               va="bottom", fontsize=8.5, color=INK_2)
            else:
                right.plot(d, mark["rung"], marker="X", color=MUTED, markersize=7.5, markeredgecolor=SURFACE, markeredgewidth=1.0,
                           linestyle="none", zorder=4)
                right.annotate(f"{mark['rung']} did not pass", (d, mark["rung"]), xytext=(0, 8), textcoords="offset points",
                               ha="center", va="bottom", fontsize=8.5, color=INK_2)
            right.plot(d, start, marker="o", color=BUDGET_RAMP[-1], markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                       linestyle="none", zorder=5)
            right.annotate(f"{start}", (d, start), xytext=(0, -9), textcoords="offset points", ha="center", va="top",
                           fontsize=8.5, color=INK_2)
        _log2_widths(right, widths, [str(d) for d in widths])
        _axes_style(right)
        right.tick_params(labelleft=False)
        right.set_xlabel("embedding width")
        _title(right, f"E6: the next rung up, trained for {data['e6_epochs']:,} epochs")
        handles = [Line2D([], [], marker="o", color=BUDGET_RAMP[-1], linestyle="none", markersize=6.5, label=f"E4 capacity, {budgets[-1]:,} epochs"),
                   Line2D([], [], marker="^", color=SERIES[0], linestyle="none", markersize=7.5, label="rung passed: capacity is at least this"),
                   Line2D([], [], marker="X", color=MUTED, linestyle="none", markersize=7.5, label="rung did not pass")]
        right.legend(handles=handles, loc="lower right", fontsize=8, labelcolor=INK_2, handletextpad=0.4)
        fig.savefig(path, dpi=DPI, metadata={"Software": None})
        texts = _texts(fig)
        plt.close(fig)
        return texts


def figure_e4_passmap(rows: list[dict], data: dict, path: Path) -> list[str]:
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt = _pyplot()
    widths, budgets = data["widths"], data["budgets"]
    cmap, norm = ListedColormap(SEEDS_RAMP), BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], len(SEEDS_RAMP))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(widths), figsize=FIGSIZES["fig_e4_passmap"], gridspec_kw={"wspace": 0.3})
        fig.subplots_adjust(left=0.065, right=0.99, top=0.8, bottom=0.2)
        for ax, d in zip(axes, widths):
            rungs, matrix, markers = passmap_grid(rows, d, budgets)
            ax.pcolormesh(np.arange(len(budgets) + 1), np.arange(len(rungs) + 1), np.array(matrix), cmap=cmap, norm=norm,
                          edgecolors=SURFACE, linewidth=2.0)
            for col, row in markers:
                ax.plot(col + 0.5, row + 0.5, marker="o", markersize=6.5, markerfacecolor=SURFACE, markeredgecolor=INK,
                        markeredgewidth=1.4, linestyle="none", zorder=5)
            ax.set_xticks(np.arange(len(budgets)) + 0.5)
            ax.set_xticklabels([str(b) for b in budgets])
            ax.set_yticks(np.arange(len(rungs)) + 0.5)
            ax.set_yticklabels([str(n) for n in rungs])
            ax.tick_params(length=0, labelsize=8.5)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel("training budget (epochs)")
            _title(ax, f"width {d}")
        axes[0].set_ylabel("entries in the book (rung)")
        fig.text(0.065, 0.93, "A rung passes at a budget when at least 2 of its 3 book seeds recall every entry exactly by then.",
                 fontsize=9, color=INK_2)
        handles = [Patch(facecolor=SEEDS_RAMP[i], label=f"{i} of 3 seeds") for i in range(len(SEEDS_RAMP))]
        handles.append(Line2D([], [], marker="o", markersize=6.5, markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.4,
                              linestyle="none", label="capacity at that budget"))
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.02),
                   title="seeds that recalled every entry exactly by the budget", title_fontsize=8.5)
        fig.savefig(path, dpi=DPI, metadata={"Software": None})
        texts = _texts(fig)
        plt.close(fig)
        return texts


def figure_e5_replay_k(data: dict, path: Path) -> list[str]:
    from matplotlib.lines import Line2D
    plt = _pyplot()
    ks = data["ks"]
    panels = (("stable_joint", "median_stable_joint", "updates until both books stay recalled", "Time to joint recall", (0, 66), (1, 2, 9)),
              ("min_a", "median_min_a", "fewest entries of A recalled (of 10)", "Depth of the dip in A", (0, 10.6), (1, 9)))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZES["fig_e5_replay_k"], gridspec_kw={"wspace": 0.22})
        fig.subplots_adjust(left=0.075, right=0.985, top=0.8, bottom=0.17)
        for ax, (key, median_key, ylabel, title, ylim, labelled) in zip(axes, panels):
            for p in data["phonebooks"]:
                ax.plot(ks, [data[key][p][k] for k in ks], color=MUTED, linewidth=1.0, solid_capstyle="round", zorder=2)
            ax.plot(ks, [data[median_key][k] for k in ks], color=SERIES[0], linewidth=2.2, marker="o", markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, solid_capstyle="round", zorder=4)
            for k in labelled:
                last = k == ks[-1]
                ax.annotate(f"{data[median_key][k]:g}", (k, data[median_key][k]), xytext=(9, 0) if last else (0, 9),
                            textcoords="offset points", ha="left" if last else "center", va="center" if last else "bottom",
                            fontsize=9, color=INK)
            ax.set_xticks(ks)
            ax.set_xlim(ks[0] - 0.5, ks[-1] + 1.1)
            ax.set_ylim(*ylim)
            _axes_style(ax)
            ax.set_xlabel("k, copies of B per copy of A")
            ax.set_ylabel(ylabel)
            _title(ax, title)
        handles = [Line2D([], [], color=MUTED, linewidth=1.0, label=f"one phonebook (median of its 3 Phase-A models)"),
                   Line2D([], [], color=SERIES[0], linewidth=2.2, marker="o", markersize=6.5, label=f"median of the {len(data['phonebooks'])} phonebooks")]
        axes[0].legend(handles=handles, loc="lower right", fontsize=8.5, labelcolor=INK_2)
        fig.text(0.075, 0.93, f"E5 at lr = {data['lr']:g}: six independent phonebooks, one copy of A and k copies of B in the replay data.",
                 fontsize=9, color=INK_2)
        fig.savefig(path, dpi=DPI, metadata={"Software": None})
        texts = _texts(fig)
        plt.close(fig)
        return texts


def figure_e6_curves(curves: dict, path: Path) -> list[str]:
    from matplotlib.ticker import FuncFormatter
    plt = _pyplot()
    widths = sorted({d for d, _ in curves})
    seeds = sorted({s for _, s in curves})
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(seeds), len(widths), figsize=FIGSIZES["fig_e6_curves"], sharex=True, sharey=True,
                                 gridspec_kw={"wspace": 0.14, "hspace": 0.16}, squeeze=False)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.08)
        for col, d in enumerate(widths):
            for row, seed in enumerate(seeds):
                ax, c = axes[row][col], curves[(d, seed)]
                fraction = [v / c["entries"] for v in c["recalled"]]
                ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=1)
                ax.axvline(1000, color=AXIS, linewidth=0.8, zorder=1)
                ax.plot(c["epochs"], fraction, color=SERIES[0], linewidth=1.2, solid_joinstyle="round", zorder=3)
                first = first_perfect(c)
                if first is not None:
                    ax.plot(first, 1.0, marker="o", markersize=6, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.2,
                            linestyle="none", zorder=5)
                ax.text(3980, 1.006, curve_note(c), ha="right", va="bottom", fontsize=8, color=INK_2)
                ax.set_xlim(0, 4000)
                ax.set_ylim(0.9, 1.026)
                ax.set_axisbelow(True)
                ax.tick_params(length=3, labelsize=8)
                if row == 0:
                    ax.set_title(f"width {d}: {c['entries']} entries", loc="left", fontsize=10, fontweight="bold", color=INK, pad=6)
                if col == 0:
                    ax.set_ylabel(f"seed {seed}", fontsize=9)
                if row == len(seeds) - 1:
                    ax.set_xlabel("epoch", fontsize=9)
        axes[0][0].set_yticks([0.9, 0.95, 1.0])
        axes[0][0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
        axes[0][0].set_xticks([0, 1000, 2000, 3000, 4000])
        fig.text(0.07, 0.93, "E6: share of the entries recalled exactly, for the first rung that failed in E4, trained for 4,000 epochs. "
                             "The y axis starts at 90%.", fontsize=9, color=INK_2)
        fig.text(0.07, 0.905, "Dot: first epoch with every entry recalled. Gray lines: 100% and E4's 1,000-epoch budget.", fontsize=8.5, color=MUTED)
        fig.savefig(path, dpi=DPI, metadata={"Software": None})
        texts = _texts(fig)
        plt.close(fig)
        return texts


# ------------------------------------------------------------------ writing


def write_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_all(out: Path, *, results: Path = RESULTS_DIR, week: Path = WEEK_DIR, force: bool = False) -> list[Path]:
    out = Path(out)
    targets = [out / f"{name}{suffix}" for name in FIGSIZES for suffix in (".png", ".csv")]
    existing = [t.name for t in targets if t.exists()]
    if existing and not force:
        raise ValueError(f"{', '.join(existing)} already exists in {out}; refusing to overwrite (use --force)")
    e4, e6 = load_json(Path(results) / "e4_capacity.json"), load_json(Path(results) / "e6_extension.json")
    capacity = capacity_data(e4, e6)
    passmap = passmap_rows(e4)
    replay = replay_data(load_replay_cells(Path(results) / "e5_cells.csv"))
    curves = curve_data(week)
    if len(curves) != 12:
        raise ValueError(f"expected the 12 E6 runs, found {len(curves)} under {week}")
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "fig_e4_capacity.csv", capacity_rows(capacity), TABLES["fig_e4_capacity"])
    write_table(out / "fig_e4_passmap.csv", passmap, TABLES["fig_e4_passmap"])
    write_table(out / "fig_e5_replay_k.csv", replay_rows(replay), TABLES["fig_e5_replay_k"])
    write_table(out / "fig_e6_curves.csv", curve_rows(curves), TABLES["fig_e6_curves"])
    figure_e4_capacity(capacity, out / "fig_e4_capacity.png")
    figure_e4_passmap(passmap, capacity, out / "fig_e4_passmap.png")
    figure_e5_replay_k(replay, out / "fig_e5_replay_k.png")
    figure_e6_curves(curves, out / "fig_e6_curves.png")
    return targets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--week", type=Path, default=WEEK_DIR)
    parser.add_argument("--force", action="store_true", help="replace existing figure files")
    args = parser.parse_args(argv)
    try:
        written = write_all(args.out, results=args.results, week=args.week, force=args.force)
    except ValueError as error:
        sys.exit(f"error: {error}")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
