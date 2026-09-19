from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_name(*args, **kwargs) -> str:
    sys.path.insert(0, str(ROOT))
    import phonebook.run_lr_sweep as sweep_module

    return sweep_module.run_name(*args, **kwargs)


@pytest.mark.parametrize(
    "lr, epochs, seed, l2sp_lambda, expected",
    [
        (0.0001, 100, 42, 0.0, "lr0.0001_ep100_seed42"),
        (0.0001, 100, 42, 0.1, "lr0.0001_ep100_seed42_l2sp0.1"),
        (0.0001, 100, 42, 1e-4, "lr0.0001_ep100_seed42_l2sp0.0001"),
    ],
)
def test_run_name_suffix_matches_expected_string(lr, epochs, seed, l2sp_lambda, expected) -> None:
    assert _run_name(lr, epochs, seed, l2sp_lambda) == expected


def test_run_name_gives_distinct_names_across_the_actual_sweep_grid() -> None:
    """The 5 lambda values actually used in this week's sweep
    (0, 1e-4, 1e-3, 1e-2, 1e-1) must not collide with each other."""
    names = {_run_name(0.0001, 100, 42, l) for l in [0, 1e-4, 1e-3, 1e-2, 1e-1]}
    assert len(names) == 5


def test_rejects_negative_l2sp_lambdas_before_planning_any_runs(run_script, tmp_path: Path) -> None:
    result = run_script(
        "phonebook/run_lr_sweep.py",
        ["--lrs", "0.0001", "--l2sp-lambdas", "0.1", "-0.1", "--dry-run"],
        cwd=tmp_path,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--l2sp-lambdas" in combined
    # Must fail before planning/printing any runs -- not mid-sweep.
    assert "Planned runs" not in combined


def test_accepts_all_non_negative_l2sp_lambdas(run_script, tmp_path: Path) -> None:
    result = run_script(
        "phonebook/run_lr_sweep.py",
        ["--lrs", "0.0001", "--l2sp-lambdas", "0", "0.1", "--dry-run"],
        cwd=tmp_path,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Planned runs: 2" in combined


def test_skip_existing_requires_metrics_csv_to_reach_the_expected_epoch(run_script, tmp_path: Path) -> None:
    """metrics.csv is opened and flushed incrementally starting partway
    through a run (see train.py), so a run killed by --timeout or a crash
    leaves a real metrics.csv on disk whose last logged epoch falls short of
    the requested total. Existence alone is not a safe "done" signal."""
    out_root = tmp_path / "sweep_out"
    partial_dir = out_root / "lr0.0001_ep100_seed42"
    partial_dir.mkdir(parents=True)
    (partial_dir / "metrics.csv").write_text("Epoch,Train Loss\n10,4.0\n", encoding="utf-8")

    result = run_script(
        "phonebook/run_lr_sweep.py",
        ["--lrs", "0.0001", "--out-root", str(out_root), "--dry-run"],
        cwd=tmp_path,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "SKIPPED" not in combined, "a partial run (last epoch 10 of 100) must not be treated as done"


def test_skip_existing_skips_when_metrics_csv_reaches_the_expected_epoch(run_script, tmp_path: Path) -> None:
    out_root = tmp_path / "sweep_out"
    complete_dir = out_root / "lr0.0001_ep100_seed42"
    complete_dir.mkdir(parents=True)
    # No model.pth here on purpose: this must match every already-committed
    # historical sweep directory, which never has one (gitignored, per
    # policy) -- skip-existing has to work from metrics.csv content alone.
    (complete_dir / "metrics.csv").write_text(
        "Epoch,Train Loss\n10,4.0\n100,1.2\n", encoding="utf-8"
    )

    result = run_script(
        "phonebook/run_lr_sweep.py",
        ["--lrs", "0.0001", "--out-root", str(out_root), "--dry-run"],
        cwd=tmp_path,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "SKIPPED" in combined


def test_manifest_records_fresh_git_state_per_run_not_once_for_the_whole_sweep(
    monkeypatch, tmp_path: Path
) -> None:
    """A sweep can run for a while across many combos; if the tree changes
    mid-sweep, later manifest entries must reflect *that run's* git state,
    not a single snapshot taken before the first run started."""
    sys.path.insert(0, str(ROOT))
    import phonebook.run_lr_sweep as sweep_module

    call_count = {"n": 0}

    def fake_get_git_state(cwd):
        call_count["n"] += 1
        return {"commit": f"fake-commit-{call_count['n']}", "dirty": False}

    def fake_subprocess_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sweep_module, "get_git_state", fake_get_git_state)
    monkeypatch.setattr(sweep_module.subprocess, "run", fake_subprocess_run)

    out_root = tmp_path / "sweep_out"
    monkeypatch.setattr(
        sys, "argv",
        ["run_lr_sweep.py", "--lrs", "0.0001", "--seeds", "42", "123", "--out-root", str(out_root)],
    )
    sweep_module.main()

    records = [json.loads(line) for line in (out_root / "manifest.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["git_commit"] != records[1]["git_commit"], (
        "each run must capture its own git state; reusing one pre-sweep snapshot "
        "means a mid-sweep commit/edit would be misattributed to later runs"
    )
