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
