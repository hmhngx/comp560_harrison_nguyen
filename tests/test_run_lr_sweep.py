from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(script_rel: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    script = ROOT / script_rel
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def test_rejects_negative_l2sp_lambdas_before_planning_any_runs(tmp_path: Path) -> None:
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


def test_accepts_all_non_negative_l2sp_lambdas(tmp_path: Path) -> None:
    result = run_script(
        "phonebook/run_lr_sweep.py",
        ["--lrs", "0.0001", "--l2sp-lambdas", "0", "0.1", "--dry-run"],
        cwd=tmp_path,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Planned runs: 2" in combined
