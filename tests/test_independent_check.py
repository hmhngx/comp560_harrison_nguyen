"""The independent recompute must agree with the analysis modules on the same evidence, and notice when it does not."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from phonebook import analyze_capacity as ac
from phonebook import analyze_replay_k as ar
from phonebook import independent_check as ic

ROOT = Path(__file__).resolve().parents[1]
METRIC_HEADER = ["Epoch", "Train Loss", "Val Loss", "Train Token Acc", "Val Token Acc",
                 "Train Seq Acc", "Val Seq Acc", "Training Time (sec)", "Param Count"]


def write_run(root: Path, protocol_run: dict, curve: list[tuple], *, books: dict, params: int = 9216) -> None:
    folder = root / protocol_run["scenario"]
    (folder / protocol_run["condition"]).mkdir(parents=True, exist_ok=True)
    evaluation = {"books": {name: {"generation": {"total": total, "correct": correct}} for name, (total, correct) in books.items()}}
    (folder / f"{protocol_run['condition']}.evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    with (folder / protocol_run["condition"] / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(METRIC_HEADER)
        for epoch, train, val in curve:
            writer.writerow([epoch, "1.0", "1.0", "0.5", "0.5", f"{train:.4f}", f"{val:.4f}", "0.1", params])


def write_protocol(root: Path, seed: int, runs: list[dict]) -> None:
    (root / "protocol.json").write_text(json.dumps({"spec": {"seed": seed}, "runs": runs}), encoding="utf-8")


# ------------------------------------------------------------------ E4

def capacity_root(root: Path, first_perfect: dict, *, d: int = 16, epochs: int = 6) -> Path:
    """first_perfect: {(seed, n): epoch of first full recall or None}; the curve climbs and then stays perfect."""
    runs = []
    for (seed, n), first in first_perfect.items():
        scenario = f"s{seed}_n{n}"
        run = {"id": f"{scenario}/d{d}", "scenario": scenario, "condition": f"d{d}", "mode": "a1b0_evalB",
               "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B", "lr": 0.001, "epochs": epochs,
               "batch_size": 64, "accuracy_interval": 1, "l2sp_lambda": 0.0, "adapt": False, "n_train": n,
               "updates_per_epoch": 1, "seq_len": 30, "model": {"embedding_dim": d, "n_heads": 2, "n_layers": 2}}
        runs.append(run)
        curve = []
        for epoch in range(1, epochs + 1):
            recalled = n if first is not None and epoch >= first else n // 2
            curve.append((epoch, recalled / n, 0.0))
        write_run(root, run, curve, books={"A": (n, curve and int(curve[-1][1] * n)), "B": (20, 0)})
    write_protocol(root, 42, runs)
    return root


E4_DATA = {(1, 10): 2, (2, 10): 3, (3, 10): 4,
           (1, 20): 3, (2, 20): 5, (3, 20): None,
           (1, 40): 6, (2, 40): 6, (3, 40): None,        # passes at budget 6 only because "by 6" includes epoch 6
           (1, 80): None, (2, 80): None, (3, 80): None,
           (1, 160): 1, (2, 160): 1, (3, 160): 1}        # a lucky pass above the first failing rung must not count


def test_the_independent_capacity_agrees_with_the_analysis_module(tmp_path: Path) -> None:
    root = capacity_root(tmp_path / "e4", E4_DATA)
    budgets = (2, 4, 6)
    mine = ic.e4([root], budgets)
    cells = ac.load_capacity_cells(root, budgets)
    theirs = {entry_d: entry["capacity_by_budget"] for entry_d, entry in ac.summarize_models(cells, budgets).items()}
    assert mine == theirs
    assert mine[16] == {2: None, 4: 10, 6: 40}


def test_a_changed_analysis_result_is_reported_as_a_difference(tmp_path: Path) -> None:
    root = capacity_root(tmp_path / "e4", E4_DATA)
    mine = ic.e4([root], (2, 4, 6))
    analysis = {"16": {"capacity_by_budget": {"2": None, "4": 10, "6": 20}}}
    differences = ic.compare_e4(mine, analysis)
    assert len(differences) == 1 and "budget 6" in differences[0]
    assert ic.compare_e4(mine, {"16": {"capacity_by_budget": {"2": None, "4": 10, "6": 40}}}) == []


def test_a_model_missing_from_the_analysis_is_a_difference(tmp_path: Path) -> None:
    mine = ic.e4([capacity_root(tmp_path / "e4", E4_DATA)], (2,))
    assert ic.compare_e4(mine, {}) != []


# ------------------------------------------------------------------ E5

PHONEBOOKS = [2, 4, 7, 8, 9, 12]
INITS = (42, 123, 999)


def replay_run(scenario: str, condition: str, a_copies: int, b_copies: int, lr: float) -> dict:
    return {"id": f"{scenario}/{condition}", "scenario": scenario, "condition": condition,
            "mode": f"a{a_copies}b{b_copies}_evalA", "train": {"a_copies": a_copies, "b_copies": b_copies},
            "eval_book": "A", "lr": lr, "epochs": 10, "batch_size": 128, "accuracy_interval": 1, "l2sp_lambda": 0.0,
            "adapt": True, "n_train": 10 * (a_copies + b_copies), "updates_per_epoch": 1, "seq_len": 29}


def replay_root(root: Path, seed: int, plan: list[tuple]) -> Path:
    """plan: (phonebook, init, k, stable epoch or None, lowest A); before the stable epoch B is incomplete."""
    runs = []
    for phonebook, init, k, stable, low_a in plan:
        scenario, condition = f"pb{phonebook}_s{init}", f"k{k}_lr0.001"
        run = replay_run(scenario, condition, 1, k, 0.001)
        runs.append(run)
        curve = []
        for epoch in range(1, 11):
            if stable is not None and epoch >= stable:
                a, b = 10, 10
            elif stable is not None and epoch == stable - 1:
                a, b = 10, 5                                   # A recalled, B not yet
            else:
                a, b = low_a, 5
            curve.append((epoch, (a + k * b) / (10 * (1 + k)), a / 10))
        final = curve[-1]
        write_run(root, run, curve, books={"A": (10, round(final[2] * 10)), "B": (10, 10 if stable is not None else 5)}, params=110_528)
    write_protocol(root, seed, runs)
    return root


def main_plan() -> list[tuple]:
    plan = []
    for phonebook in PHONEBOOKS:
        for init in INITS:
            base = 4 + phonebook % 3 + (init % 2)
            plan.append((phonebook, init, 1, None if phonebook == 9 else base, 8))
            plan.append((phonebook, init, 3, base - (3 if phonebook != 12 else 0), 3))
    return plan


def noise_plan(seed: int) -> list[tuple]:
    return [(phonebook, 42, k, 4 + phonebook % 3 + (seed % 2) * (3 if phonebook == 7 else 1) - (3 if k == 3 else 0),
             8 if k == 1 else 3)
            for phonebook in (2, 4, 7) for k in (1, 3)]


@pytest.fixture
def e5_roots(tmp_path: Path) -> list[Path]:
    return [replay_root(tmp_path / "main", 42, main_plan())] + \
           [replay_root(tmp_path / f"noise{seed}", seed, noise_plan(seed)) for seed in (1, 2)]


def test_the_independent_replay_analysis_agrees_with_the_analysis_module(e5_roots, tmp_path: Path) -> None:
    kwargs = dict(lrs=[0.001], primary_lr=0.001, ks=[1, 3], phonebooks=PHONEBOOKS)
    cells = [cell for root in e5_roots for cell in ar.load_replay_cells(root)]
    ar.write_report(cells, tmp_path / "out", prefix="e5", **kwargs)
    theirs = json.loads((tmp_path / "out" / "e5_analysis.json").read_text(encoding="utf-8"))
    mine = ic.e5(e5_roots, **kwargs)
    assert ic.compare_e5(mine, theirs) == []
    assert mine["noise_band"] == theirs["noise"]["band"] and mine["noise_band"] > 0


def test_the_synthetic_replay_evidence_has_the_properties_the_comparison_relies_on(e5_roots) -> None:
    mine = ic.e5(e5_roots, lrs=[0.001], primary_lr=0.001, ks=[1, 3], phonebooks=PHONEBOOKS)
    entry = mine["lr"]["0.001"]
    assert entry["paired"]["3"]["faster"] >= 5            # k=3 is faster than k=1 in most phonebooks
    assert entry["median_stable_joint"]["3"] < entry["median_stable_joint"]["1"]
    assert entry["paired"]["3"]["beats"] in (True, False) and entry["plateau"] is not None


def test_a_tampered_analysis_value_is_reported(e5_roots, tmp_path: Path) -> None:
    kwargs = dict(lrs=[0.001], primary_lr=0.001, ks=[1, 3], phonebooks=PHONEBOOKS)
    cells = [cell for root in e5_roots for cell in ar.load_replay_cells(root)]
    ar.write_report(cells, tmp_path / "out", prefix="e5", **kwargs)
    theirs = json.loads((tmp_path / "out" / "e5_analysis.json").read_text(encoding="utf-8"))
    theirs["lr"]["0.001"]["median_stable_joint"]["3"] += 1
    theirs["noise"]["band"] += 1
    differences = ic.compare_e5(ic.e5(e5_roots, **kwargs), theirs)
    assert len(differences) == 2


def test_a_count_that_is_not_a_whole_number_of_entries_is_rejected(tmp_path: Path) -> None:
    root = replay_root(tmp_path / "main", 42, [(2, 42, 1, 3, 8)])
    metrics = root / "pb2_s42" / "k1_lr0.001" / "metrics.csv"
    metrics.write_text(metrics.read_text(encoding="utf-8").replace("0.6500", "0.6530", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="whole number"):
        ic.e5([root], lrs=[0.001], primary_lr=0.001, ks=[1], phonebooks=[2])


def test_the_script_does_not_import_the_modules_it_checks() -> None:
    source = (ROOT / "phonebook" / "independent_check.py").read_text(encoding="utf-8")
    for module in ("analyze_capacity", "analyze_replay_k", "analyze_experiment", "report_experiments", "run_experiment"):
        assert module not in source.replace("analyze_capacity.py", "").replace("analyze_replay_k.py", "").split('"""', 2)[2]


def test_the_command_line_exits_nonzero_on_a_disagreement(tmp_path: Path) -> None:
    root = capacity_root(tmp_path / "e4", E4_DATA)
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text(json.dumps({"16": {"capacity_by_budget": {"2": None, "4": 10, "6": 40}}}), encoding="utf-8")
    bad.write_text(json.dumps({"16": {"capacity_by_budget": {"2": None, "4": 10, "6": 20}}}), encoding="utf-8")
    command = [sys.executable, str(ROOT / "phonebook" / "independent_check.py"), "e4", "--root", str(root),
               "--budgets", "2", "4", "6"]
    assert subprocess.run(command + ["--against", str(good)], capture_output=True, text=True).returncode == 0
    failed = subprocess.run(command + ["--against", str(bad)], capture_output=True, text=True)
    assert failed.returncode == 1 and "budget 6" in failed.stdout


# ------------------------------------------------------------------ review fixes

def test_a_budget_beyond_the_run_length_is_rejected_by_the_independent_check_too(tmp_path: Path) -> None:
    root = capacity_root(tmp_path / "e4", E4_DATA)           # six epochs per run
    with pytest.raises(ValueError, match="budget 60"):
        ic.e4([root], (2, 60))


def test_censored_runs_are_counted_the_same_way_by_both_computations(e5_roots, tmp_path: Path) -> None:
    kwargs = dict(lrs=[0.001], primary_lr=0.001, ks=[1, 3], phonebooks=PHONEBOOKS)
    mine = ic.e5(e5_roots, **kwargs)["lr"]["0.001"]
    assert mine["censored"] == {"1": 3, "3": 0}               # phonebook 9 never reaches joint recall at k=1 on any init
    cells = [cell for root in e5_roots for cell in ar.load_replay_cells(root)]
    ar.write_report(cells, tmp_path / "out", prefix="e5", **kwargs)
    theirs = json.loads((tmp_path / "out" / "e5_analysis.json").read_text(encoding="utf-8"))
    assert theirs["lr"]["0.001"]["censored"] == mine["censored"]
    theirs["lr"]["0.001"]["censored"]["1"] = 0
    assert any("censored" in line for line in ic.compare_e5(ic.e5(e5_roots, **kwargs), theirs))
