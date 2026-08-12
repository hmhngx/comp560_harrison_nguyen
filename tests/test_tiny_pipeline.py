from __future__ import annotations

import importlib.util
import subprocess
import sys
import csv
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def run_script(script_rel: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    script = ROOT / script_rel
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.integration
def test_tiny_end_to_end_pipeline_with_small_transformer(tmp_path: Path) -> None:
    if not has_torch():
        pytest.skip("torch is not installed in this environment")

    # 1) Tiny input with enough lines for a non-empty validation split.
    tiny_input = tmp_path / "tiny_eval.txt"
    tiny_input.write_text(
        "a=A\n"
        "b=B\n"
        "c=C\n"
        "ab=AB\n"
        "ba=BA\n"
        "ac=AC\n"
        "ca=CA\n"
        "bc=BC\n",
        encoding="utf-8",
    )

    data_dir = tmp_path / "tiny_data"
    tiny_out_dir = tmp_path / "tiny_out"
    compat_out_dir = tmp_path / "compat_out"
    adapt_out_dir = tmp_path / "adapt_out"

    # 2) Prepare tokenized data.
    prep = run_script(
        "prepare_inputs.py",
        [str(tiny_input), "--out-dir", str(data_dir), "--train-split", "0.75"],
        cwd=tmp_path,
    )
    assert prep.returncode == 0, prep.stderr

    # 3) Train with tiny config: dim=32, heads=2, layers=2, epochs=1.
    train_tiny = run_script(
        "train.py",
        [
            str(ROOT / "config" / "config_tiny_test.py"),
            "--device",
            "cpu",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(tiny_out_dir),
        ],
        cwd=tmp_path,
    )
    assert train_tiny.returncode == 0, train_tiny.stderr
    assert (tiny_out_dir / "model.pth").exists()
    assert (tiny_out_dir / "metrics.csv").exists()
    tiny_output = (train_tiny.stdout + train_tiny.stderr).lower()
    assert "train acc" in tiny_output
    assert "val acc" in tiny_output

    with (tiny_out_dir / "metrics.csv").open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)

    assert "Train Token Acc" in header
    assert "Val Token Acc" in header
    assert "Train Seq Acc" in header
    assert "Val Seq Acc" in header

    # 4) Train a compatibility checkpoint for the current inference scripts,
    # which are fixed to 128-dim, 4 heads, 4 layers.
    train_compat = run_script(
        "train.py",
        [
            "--device",
            "cpu",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(compat_out_dir),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--embedding-dim",
            "128",
            "--n-heads",
            "4",
            "--n-layers",
            "4",
        ],
        cwd=tmp_path,
    )
    assert train_compat.returncode == 0, train_compat.stderr
    assert (compat_out_dir / "model.pth").exists()

    # 4b) Adapt model from the compatibility checkpoint with fresh run settings.
    adapt = run_script(
        "train.py",
        [
            "--device",
            "cpu",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(adapt_out_dir),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--embedding-dim",
            "128",
            "--n-heads",
            "4",
            "--n-layers",
            "4",
            "--adapt-from",
            str(compat_out_dir / "model.pth"),
        ],
        cwd=tmp_path,
    )
    assert adapt.returncode == 0, adapt.stderr
    assert (adapt_out_dir / "model.pth").exists()
    assert "adapting model from checkpoint" in (adapt.stdout + adapt.stderr).lower()

    # 5) Single prompt inference should execute successfully.
    infer_one = run_script(
        "generate_one.py",
        ["a", "--out-dir", str(compat_out_dir), "--data-dir", str(data_dir)],
        cwd=tmp_path,
    )
    assert infer_one.returncode == 0, infer_one.stderr
    infer_one_text = (infer_one.stdout + infer_one.stderr).lower()
    assert "traceback" not in infer_one_text

    # 6) File-based evaluation should execute and print report structure.
    evaluate = run_script(
        "generate.py",
        [str(tiny_input), "--out-dir", str(compat_out_dir), "--data-dir", str(data_dir)],
        cwd=tmp_path,
    )
    assert evaluate.returncode == 0, evaluate.stderr
    report_text = evaluate.stdout + evaluate.stderr
    assert "accuracy report" in report_text.lower()
