from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]

# train.py draws a loss plot; a GUI backend can fail intermittently (a Tk error was seen once), and no test needs a window.
# Set before matplotlib is imported, and inherited by every training subprocess the tests start.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def run_script() -> Callable[[str, list[str], Path | None], subprocess.CompletedProcess[str]]:
    def _run_script(
        script_rel: str, args: list[str], cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        script = ROOT / script_rel
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=str(cwd or ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

    return _run_script


@pytest.fixture
def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None
