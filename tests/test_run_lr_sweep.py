from __future__ import annotations

from pathlib import Path


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
