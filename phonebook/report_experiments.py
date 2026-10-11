"""Tables and figures from experiment evidence, using only committed files:
protocol.json, each run's metrics.csv, and its strict evaluation JSON.

    python phonebook/report_experiments.py grid --root EVIDENCE_ROOT --out phonebook/results --prefix e1
    python phonebook/report_experiments.py e2 --root EVIDENCE_ROOT --reference phonebook/results/e1_decision.json --out phonebook/results
    python phonebook/report_experiments.py e1b --e1 E1_ROOT --e1b E1B_ROOT --out phonebook/results
    python phonebook/report_experiments.py e3 --root E3_ROOT --e1 E1_ROOT --out phonebook/results
    python phonebook/report_experiments.py compare --a SHORT_ROOT --b LONG_ROOT --pair SCENARIO:COND_A:COND_B --max-epoch 100 --out report.json

Evidence roots are the folders written by phonebook/run_experiment.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.analyze_experiment import (
    curve, decision_table, load_metrics, overall_verdict, select_lambda, summarize_curve)

BUDGETS = (100, 300)
TIMING_COLUMN = "Training Time (sec)"


def endpoint_columns(budgets: tuple[int, ...] = BUDGETS) -> list[str]:
    return (["scenario", "condition", "lr", "epochs", "l2sp_lambda", "a_copies", "b_copies"]
            + [f"{book}_at_{budget}" for budget in budgets for book in "AB"]
            + ["A_final_strict", "B_final_strict", "first_b_complete", "a_at_first_b_complete", "first_joint",
               "stable_joint", "min_a", "epoch_of_min_a", "best_joint_sum", "epoch_of_best_joint_sum"])


ENDPOINT_COLUMNS = endpoint_columns()


def load_runs(root: Path, budgets: tuple[int, ...] = BUDGETS) -> list[dict]:
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    runs = []
    for run in protocol["runs"]:
        evaluation = json.loads((root / run["scenario"] / f"{run['condition']}.evaluation.json").read_text(encoding="utf-8"))
        books = evaluation["books"]
        n_a, n_b = books["A"]["generation"]["total"], books["B"]["generation"]["total"]
        rows = load_metrics(root / run["scenario"] / run["condition"] / "metrics.csv")
        points = curve(rows, a_copies=run["train"]["a_copies"], b_copies=run["train"]["b_copies"],
                       eval_book=run["eval_book"], n_a=n_a, n_b=n_b)
        runs.append({**run, "n_a": n_a, "n_b": n_b, "points": points,
                     "strict": {name: books[name]["generation"]["correct"] for name in books},
                     "summary": summarize_curve(points, n_a=n_a, n_b=n_b, budgets=budgets)})
    return runs


def endpoint_row(run: dict, budgets: tuple[int, ...] = BUDGETS) -> dict:
    summary = run["summary"]
    row = {"scenario": run["scenario"], "condition": run["condition"], "lr": run["lr"], "epochs": run["epochs"],
           "l2sp_lambda": run["l2sp_lambda"], "a_copies": run["train"]["a_copies"], "b_copies": run["train"]["b_copies"],
           "A_final_strict": run["strict"]["A"], "B_final_strict": run["strict"]["B"]}
    for budget in budgets:
        at = summary["at"].get(budget)
        row[f"A_at_{budget}"] = at["A"] if at else ""
        row[f"B_at_{budget}"] = at["B"] if at else ""
    for key in ("first_b_complete", "a_at_first_b_complete", "first_joint", "stable_joint", "min_a",
                "epoch_of_min_a", "best_joint_sum", "epoch_of_best_joint_sum"):
        row[key] = "" if summary[key] is None else summary[key]
    return row


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def l2sp_records(runs: list[dict], budget: int) -> list[dict]:
    return [{"scenario": run["scenario"], "lambda": run["l2sp_lambda"],
             "A": run["summary"]["at"][budget]["A"], "B": run["summary"]["at"][budget]["B"]}
            for run in runs if run["l2sp_lambda"] > 0]


def e1_decision(runs: list[dict], *, min_scenarios: int = 2) -> dict:
    n_a, n_b = runs[0]["n_a"], runs[0]["n_b"]
    out = {"rule": f"one lambda gives A={n_a}/{n_a} and B={n_b}/{n_b} (matches) or A>={n_a - 1} and B={n_b} "
                   f"(nearly) on at least {min_scenarios} checkpoints", "budgets": {}}
    for budget in BUDGETS:
        records = l2sp_records(runs, budget)
        table = decision_table(records, n_a=n_a, n_b=n_b, min_scenarios=min_scenarios)
        out["budgets"][str(budget)] = {"overall": overall_verdict(table), "by_lambda": table}
    out["lambda_for_replication"] = select_lambda(l2sp_records(runs, 100))
    return out


def _at(runs: list[dict], scenario: str, condition: str, budget: int) -> dict:
    run = next(r for r in runs if r["scenario"] == scenario and r["condition"] == condition)
    return run["summary"]["at"][budget]


def e2_replication(runs: list[dict], reference: dict, *, min_scenarios: int = 2) -> dict:
    """The pre-registered E2 criteria: E1's conclusions replicate on the new phonebooks."""
    scenarios = list(dict.fromkeys(run["scenario"] for run in runs))
    n_a, n_b = runs[0]["n_a"], runs[0]["n_b"]
    replay = [s for s in scenarios if _at(runs, s, "replay_a1b1", 100) == {"A": n_a, "B": n_b}]
    plain = [s for s in scenarios if _at(runs, s, "plain", 300)["B"] == n_b and _at(runs, s, "plain", 300)["A"] <= 2]
    own = e1_decision(runs, min_scenarios=min_scenarios)
    new = {budget: own["budgets"][budget]["overall"] for budget in ("100", "300")}
    old = {budget: reference["budgets"][budget]["overall"] for budget in ("100", "300")}
    out = {
        "replay_full_recall": {"scenarios": replay, "holds": len(replay) >= min_scenarios},
        "plain_forgets_a_and_learns_b": {"scenarios": plain, "holds": len(plain) >= min_scenarios},
        "l2sp_verdict": {"e1": old, "e2": new, "holds": new == old},
    }
    out["replicates"] = all(part["holds"] for part in out.values())
    return out


def union_l2sp_records(e1_runs: list[dict], e1b_runs: list[dict], budget: int) -> list[dict]:
    """One record per (scenario, lambda) from E1 and E1b; the 1000-update budget exists only in E1b."""
    records: dict[tuple, dict] = {}
    for runs in ([e1b_runs] if budget not in BUDGETS else [e1_runs, e1b_runs]):
        for run in runs:
            at = run["summary"]["at"].get(budget)
            if run["l2sp_lambda"] > 0 and at:
                records.setdefault((run["scenario"], run["l2sp_lambda"]),
                                   {"scenario": run["scenario"], "lambda": run["l2sp_lambda"], "A": at["A"], "B": at["B"]})
    return list(records.values())


def e1b_decision(e1_runs: list[dict], e1b_runs: list[dict], *, min_scenarios: int = 2) -> dict:
    n_a, n_b = e1_runs[0]["n_a"], e1_runs[0]["n_b"]
    out = {"rule": "E1's rule applied over the union of E1 and E1b lambdas (1000 updates: E1b lambdas only)", "budgets": {}}
    for budget in (*BUDGETS, 1000):
        table = decision_table(union_l2sp_records(e1_runs, e1b_runs, budget), n_a=n_a, n_b=n_b, min_scenarios=min_scenarios)
        out["budgets"][str(budget)] = {"overall": overall_verdict(table), "by_lambda": table}
    return out


# ------------------------------------------------------------------- E3: replay ratio


def e3_cells(runs: list[dict]) -> list[dict]:
    cells = []
    for run in runs:
        summary = run["summary"]
        at100 = summary["at"].get(100) or {"A": "", "B": ""}
        cells.append({"scenario": run["scenario"], "lr": run["lr"], "k": run["train"]["b_copies"],
                      "first_joint": summary["first_joint"], "stable_joint": summary["stable_joint"],
                      "min_a": summary["min_a"], "epoch_of_min_a": summary["epoch_of_min_a"],
                      "A_at_100": at100["A"], "B_at_100": at100["B"],
                      "final_A": summary["final"]["A"], "final_B": summary["final"]["B"]})
    return cells


def median_or_none(values: list) -> float | int | None:
    """Median over checkpoints where None (never reached) counts as infinitely slow; None if the median is infinite."""
    ordered = sorted(float("inf") if value is None else value for value in values)
    middle = statistics.median(ordered)
    if middle == float("inf"):
        return None
    return int(middle) if middle == int(middle) else middle


def e3_table(cells: list[dict]) -> list[dict]:
    rows = []
    for lr in sorted({cell["lr"] for cell in cells}, reverse=True):
        for k in sorted({cell["k"] for cell in cells if cell["lr"] == lr}):
            mine = [cell for cell in cells if cell["lr"] == lr and cell["k"] == k]
            rows.append({"lr": lr, "k": k, "checkpoints": len(mine),
                         "median_first_joint": median_or_none([c["first_joint"] for c in mine]),
                         "median_stable_joint": median_or_none([c["stable_joint"] for c in mine]),
                         "censored": sum(c["stable_joint"] is None for c in mine),
                         "median_min_a": median_or_none([c["min_a"] for c in mine])})
    return rows


def best_k_per_lr(cells: list[dict]) -> dict:
    """For each lr: the k with the smallest median stable_joint (ties to the smaller k); None if all are censored."""
    best = {}
    table = e3_table(cells)
    for lr in sorted({row["lr"] for row in table}, reverse=True):
        candidates = [row for row in table if row["lr"] == lr and row["median_stable_joint"] is not None]
        best[lr] = min(candidates, key=lambda row: (row["median_stable_joint"], row["k"]))["k"] if candidates else None
    return best


def last_week_replay_b_counts(experiments_dir: Path) -> dict[str, int]:
    """B recall at lr=0.0001 and 100 updates from last week's committed pilot summaries."""
    base = Path(experiments_dir) / "week_2026-09-30"
    sources = {"d0_seed42": base / "run_2026-10-04" / "summary.csv",
               "d0_seed123": base / "replication_phaseA_seed123" / "summary.csv",
               "d0_seed999": base / "replication_phaseA_seed999" / "summary.csv"}
    counts = {}
    for scenario, path in sources.items():
        with path.open(newline="", encoding="utf-8") as file:
            rows = {row["condition"]: row for row in csv.DictReader(file)}
        counts[scenario] = int(rows["replay_lr0.0001_ep100"]["B_generation_correct"])
    return counts


# ---------------------------------------------------------------- consistency checks


def common_epoch_comparison(path_a: Path, path_b: Path, *, max_epoch: int) -> dict:
    """Rows logged at epochs present in both files (up to max_epoch) must agree in every column but timing."""
    def read(path: Path) -> dict[int, dict]:
        with Path(path).open(newline="", encoding="utf-8") as file:
            return {int(row["Epoch"]): row for row in csv.DictReader(file)}

    first, second = read(path_a), read(path_b)
    shared = sorted(epoch for epoch in first.keys() & second.keys() if epoch <= max_epoch)
    if not shared:
        raise ValueError("the two metrics files share no epoch within the limit")
    mismatches = []
    for epoch in shared:
        for column, value in first[epoch].items():
            if column != TIMING_COLUMN and value != second[epoch][column]:
                mismatches.append({"epoch": epoch, "column": column, "a": value, "b": second[epoch][column]})
    return {"compared_epochs": len(shared), "mismatches": mismatches}


def compare_runs(root_a: Path, root_b: Path, pairs: list[tuple[str, str, str]], *, max_epoch: int) -> dict:
    """pairs: (scenario, condition in root_a, condition in root_b); the scenario id is the same in both roots."""
    report = {}
    for scenario, condition_a, condition_b in pairs:
        report[f"{scenario}/{condition_a} vs {scenario}/{condition_b}"] = common_epoch_comparison(
            Path(root_a) / scenario / condition_a / "metrics.csv",
            Path(root_b) / scenario / condition_b / "metrics.csv", max_epoch=max_epoch)
    return report


# ------------------------------------------------------------------------ figures

A_COLOR, B_COLOR = "#2a78d6", "#eb6834"


def panel_title(scenario: str) -> str:
    return scenario.replace("d0_seed", "Phase-A seed ").replace("ds", "Phonebook seed ") if scenario.startswith(("d0_seed", "ds")) else scenario


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figure_curves(runs: list[dict], path: Path, lambdas: tuple[float, ...] = (0.1, 0.3, 1, 3)) -> None:
    plt = _pyplot()
    scenarios = list(dict.fromkeys(run["scenario"] for run in runs))
    shades = plt.cm.viridis([i / max(len(lambdas) - 1, 1) for i in range(len(lambdas))])
    fig, axes = plt.subplots(2, len(scenarios), figsize=(4.2 * len(scenarios), 6.4), sharex=True, sharey=True, squeeze=False)
    for col, scenario in enumerate(scenarios):
        mine = {run["condition"]: run for run in runs if run["scenario"] == scenario}
        for row, (key, title) in enumerate((("A", "Old book A: entries still recalled"), ("B", "New book B: entries recalled"))):
            ax = axes[row][col]
            styles = [("plain", "#555555", "--", "plain fine-tuning"), ("replay_a1b1", "#0ca30c", "-", "balanced replay")]
            for condition, color, style, label in styles:
                if condition in mine:
                    pts = mine[condition]["points"]
                    ax.plot([p["epoch"] for p in pts], [p[key] for p in pts], color=color, linestyle=style, linewidth=1.8, label=label)
            for shade, lam in zip(shades, lambdas):
                run = next((r for r in mine.values() if r["l2sp_lambda"] == lam), None)
                if run:
                    ax.plot([p["epoch"] for p in run["points"]], [p[key] for p in run["points"]], color=shade, linewidth=1.5,
                            label=f"L2-SP λ={lam:g}")
            ax.set_ylim(-0.3, 10.3)
            ax.set_yticks(range(0, 11, 2))
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(panel_title(scenario), fontsize=11)
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
            if row == 1:
                ax.set_xlabel("updates")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8.5, frameon=False)
    fig.suptitle("Exact recall per update at lr=0.001 (strict: whole answer plus newline)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_dose_response(runs: list[dict], path: Path) -> None:
    plt = _pyplot()
    scenarios = list(dict.fromkeys(run["scenario"] for run in runs))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.4 * len(scenarios), 3.9), sharey=True, squeeze=False)
    plain_x, replay_x = 0.001, 1000
    for ax, scenario in zip(axes[0], scenarios):
        mine = [run for run in runs if run["scenario"] == scenario]
        l2sp = sorted((r for r in mine if r["l2sp_lambda"] > 0), key=lambda r: r["l2sp_lambda"])
        for budget, style in ((100, "-"), (300, ":")):
            for key, color, label in (("A", A_COLOR, "A (old book)"), ("B", B_COLOR, "B (new book)")):
                xs = [r["l2sp_lambda"] for r in l2sp]
                ys = [r["summary"]["at"][budget][key] for r in l2sp]
                plain = next(r for r in mine if r["condition"] == "plain")["summary"]["at"][budget][key]
                replay = next(r for r in mine if r["condition"] == "replay_a1b1")["summary"]["at"][budget][key]
                ax.plot([plain_x] + xs, [plain] + ys, color=color, linestyle=style, marker="o",
                        markersize=3.5, linewidth=1.6, label=f"{label}, {budget} updates")
                nudge = 0.7 if key == "A" else 1.4  # keep the two replay markers from hiding each other
                ax.plot([replay_x * nudge], [replay], color=color, linestyle="none", marker="D", markersize=5,
                        markerfacecolor=color if budget == 100 else "none")
        ax.set_xscale("log")
        ax.set_xticks([plain_x, 0.01, 0.1, 1, 10, 100, replay_x])
        ax.set_xticklabels(["plain", "0.01", "0.1", "1", "10", "100", "replay"], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(-0.3, 10.3)
        ax.set_yticks(range(0, 11, 2))
        ax.grid(alpha=0.25)
        ax.set_title(panel_title(scenario), fontsize=11)
        ax.set_xlabel("L2-SP strength λ")
    axes[0][0].set_ylabel("entries recalled exactly (of 10)")
    axes[0][-1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Retention versus new learning as λ grows (lr=0.001)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_fine_grid(e1_runs: list[dict], e1b_runs: list[dict], path: Path) -> None:
    plt = _pyplot()
    scenarios = list(dict.fromkeys(run["scenario"] for run in e1b_runs))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.4 * len(scenarios), 3.9), sharey=True, squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        for budget, style in ((100, "-"), (300, ":"), (1000, "--")):
            records = sorted((r for r in union_l2sp_records(e1_runs, e1b_runs, budget)
                              if r["scenario"] == scenario and 0.1 <= r["lambda"] <= 3), key=lambda r: r["lambda"])
            for key, color, label in (("A", A_COLOR, "A (old book)"), ("B", B_COLOR, "B (new book)")):
                ax.plot([r["lambda"] for r in records], [r[key] for r in records], color=color, linestyle=style,
                        marker="o", markersize=3.5, linewidth=1.5, label=f"{label}, {budget} updates")
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 0.5, 1, 3])
        ax.set_xticklabels(["0.1", "0.3", "0.5", "1", "3"], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(-0.3, 10.3)
        ax.set_yticks(range(0, 11, 2))
        ax.grid(alpha=0.25)
        ax.set_title(panel_title(scenario), fontsize=11)
        ax.set_xlabel("L2-SP strength λ")
    axes[0][0].set_ylabel("entries recalled exactly (of 10)")
    axes[0][-1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Finer λ grid and longer horizon (lr=0.001)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_e3_heatmaps(table: list[dict], path: Path) -> None:
    plt = _pyplot()
    import numpy as np
    lrs = sorted({row["lr"] for row in table}, reverse=True)
    ks = sorted({row["k"] for row in table})

    def grid(key: str):
        data = np.full((len(lrs), len(ks)), np.nan)
        for row in table:
            data[lrs.index(row["lr"]), ks.index(row["k"])] = np.nan if row[key] is None else row[key]
        return data

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    panels = ((axes[0], "median_stable_joint", "viridis_r", "Updates until both books stay exactly recalled\n(median of 3 checkpoints; gray = never within 500)"),
              (axes[1], "median_min_a", "viridis", "Fewest A entries recalled at any update\n(median of 3 checkpoints)"))
    for ax, key, cmap_name, title in panels:
        data = grid(key)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e6e6e6")
        image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, aspect="auto")
        for i in range(len(lrs)):
            for j in range(len(ks)):
                value = data[i, j]
                if np.isnan(value):
                    ax.text(j, i, "never" if key == "median_stable_joint" else "-", ha="center", va="center", fontsize=9, color="#333333")
                    continue
                red, green, blue, _ = cmap(image.norm(value))
                ax.text(j, i, f"{value:g}", ha="center", va="center", fontsize=9,
                        color="white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"k={k}" for k in ks], fontsize=9)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"lr={lr:g}" for lr in lrs], fontsize=9)
        ax.set_xlabel("copies of B per copy of A")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- drivers


def write_json(path: Path, obj) -> None:
    Path(path).write_bytes((json.dumps(obj, indent=2) + "\n").encode("utf-8"))


def report_grid(root: Path, out: Path, prefix: str) -> tuple[dict, list[dict]]:
    """Endpoint table, pre-registered decision and the two standard figures for an E1-style grid."""
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    write_csv(out / f"{prefix}_endpoints.csv", [endpoint_row(run) for run in runs], ENDPOINT_COLUMNS)
    decision = e1_decision(runs)
    write_json(out / f"{prefix}_decision.json", decision)
    figure_curves(runs, out / f"{prefix}_curves.png")
    figure_dose_response(runs, out / f"{prefix}_dose_response.png")
    return decision, runs


def report_e2(root: Path, out: Path, reference: Path) -> dict:
    _, runs = report_grid(root, out, "e2")
    verdict = e2_replication(runs, json.loads(Path(reference).read_text(encoding="utf-8")))
    write_json(out / "e2_replication.json", verdict)
    return verdict


def report_e1b(e1_root: Path, e1b_root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    budgets = (*BUDGETS, 1000)
    e1_runs, e1b_runs = load_runs(e1_root), load_runs(e1b_root, budgets=budgets)
    write_csv(out / "e1b_endpoints.csv", [endpoint_row(run, budgets) for run in e1b_runs], endpoint_columns(budgets))
    decision = e1b_decision(e1_runs, e1b_runs)
    write_json(out / "e1b_decision.json", decision)
    scenarios = list(dict.fromkeys(run["scenario"] for run in e1b_runs))
    pairs = [(scenario, f"l2sp_lam{lam}", f"l2sp_lam{lam}") for scenario in scenarios for lam in ("0.3", "1")]
    write_json(out / "e1b_vs_e1.json", compare_runs(e1b_root, e1_root, pairs, max_epoch=300))
    figure_fine_grid(e1_runs, e1b_runs, out / "e1b_fine_grid.png")
    return decision


E3_CELL_COLUMNS = ["scenario", "lr", "k", "first_joint", "stable_joint", "min_a", "epoch_of_min_a",
                   "A_at_100", "B_at_100", "final_A", "final_B"]
E3_TABLE_COLUMNS = ["lr", "k", "checkpoints", "median_first_joint", "median_stable_joint", "censored", "median_min_a"]


def _blank_none(rows: list[dict]) -> list[dict]:
    return [{key: ("" if value is None else value) for key, value in row.items()} for row in rows]


def report_e3(root: Path, e1_root: Path, out: Path, experiments_dir: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root, budgets=(100,))
    cells = e3_cells(runs)
    table = e3_table(cells)
    best = best_k_per_lr(cells)
    write_csv(out / "e3_cells.csv", _blank_none(cells), E3_CELL_COLUMNS)
    write_csv(out / "e3_table.csv", _blank_none(table), E3_TABLE_COLUMNS)
    write_json(out / "e3_best_k.json", {f"{lr:g}": k for lr, k in best.items()})
    figure_e3_heatmaps(table, out / "e3_heatmaps.png")
    scenarios = list(dict.fromkeys(run["scenario"] for run in runs))
    mine = {scenario: next(r for r in runs if r["scenario"] == scenario and r["condition"] == "k1_lr0.0001")
            ["summary"]["at"][100]["B"] for scenario in scenarios}
    last_week = last_week_replay_b_counts(experiments_dir)
    checks = {
        "k1_lr0.001_vs_e1_replay": compare_runs(root, e1_root, [(s, "k1_lr0.001", "replay_a1b1") for s in scenarios], max_epoch=300),
        "k1_lr0.0001_b_at_100_updates": {"this_week": mine, "last_week": last_week, "match": mine == last_week},
    }
    write_json(out / "e3_cross_checks.json", checks)
    return {"best_k_per_lr": {f"{lr:g}": k for lr, k in best.items()}, "cross_checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    grid = sub.add_parser("grid", help="tables, decision and figures for an E1-style grid (E1 and E2 share it)")
    grid.add_argument("--root", type=Path, required=True)
    grid.add_argument("--out", type=Path, required=True)
    grid.add_argument("--prefix", required=True)
    e2 = sub.add_parser("e2", help="E2 grid plus the pre-registered replication verdict")
    e2.add_argument("--root", type=Path, required=True)
    e2.add_argument("--out", type=Path, required=True)
    e2.add_argument("--reference", type=Path, required=True, help="E1's e1_decision.json")
    e1b = sub.add_parser("e1b", help="E1b decision over the union of the E1 and E1b grids")
    e1b.add_argument("--e1", type=Path, required=True)
    e1b.add_argument("--e1b", type=Path, required=True)
    e1b.add_argument("--out", type=Path, required=True)
    e3 = sub.add_parser("e3", help="E3 tables, best k per learning rate, heatmaps and cross-checks")
    e3.add_argument("--root", type=Path, required=True)
    e3.add_argument("--e1", type=Path, required=True)
    e3.add_argument("--out", type=Path, required=True)
    e3.add_argument("--experiments-dir", type=Path, default=ROOT_DIR / "phonebook" / "experiments")
    compare = sub.add_parser("compare", help="logged rows of two runs must agree at every shared epoch")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    compare.add_argument("--pair", action="append", required=True, metavar="SCENARIO:COND_A:COND_B")
    compare.add_argument("--max-epoch", type=int, required=True)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "grid":
        print(json.dumps(report_grid(args.root, args.out, args.prefix)[0], indent=2))
        return
    if args.command == "e2":
        print(json.dumps(report_e2(args.root, args.out, args.reference), indent=2))
        return
    if args.command == "e1b":
        print(json.dumps(report_e1b(args.e1, args.e1b, args.out), indent=2))
        return
    if args.command == "e3":
        print(json.dumps(report_e3(args.root, args.e1, args.out, args.experiments_dir), indent=2))
        return
    pairs = [tuple(spec.split(":")) for spec in args.pair]
    if any(len(pair) != 3 for pair in pairs):
        parser.error("--pair must look like SCENARIO:COND_A:COND_B")
    report = compare_runs(args.a, args.b, pairs, max_epoch=args.max_epoch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    bad = {name: result for name, result in report.items() if result["mismatches"]}
    for name, result in report.items():
        print(f"{'MISMATCH' if result['mismatches'] else 'ok'}: {name} ({result['compared_epochs']} epochs compared)")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
