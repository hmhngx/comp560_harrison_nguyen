from __future__ import annotations

import subprocess
import csv
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_tiny_end_to_end_pipeline_with_small_transformer(run_script, has_torch, tmp_path: Path) -> None:
    if not has_torch:
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

    # 4c) --l2sp-lambda must change the training trajectory relative to an
    # otherwise-identical adapt run with the penalty off. One epoch isn't
    # enough to observe this in the *reported* loss (the penalty can only
    # affect a step after params have already drifted from old_params by at
    # least one prior step), so this uses --epochs 3.
    def run_adapt(out_dir: Path, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
        return run_script(
            "train.py",
            [
                "--device", "cpu",
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),
                "--epochs", "3",
                "--batch-size", "4",
                "--embedding-dim", "128",
                "--n-heads", "4",
                "--n-layers", "4",
                "--adapt-from", str(compat_out_dir / "model.pth"),
                *extra_args,
            ],
            cwd=tmp_path,
        )

    no_l2sp_out_dir = tmp_path / "adapt_no_l2sp_out"
    l2sp_out_dir = tmp_path / "adapt_l2sp_out"
    adapt_no_l2sp = run_adapt(no_l2sp_out_dir, [])
    assert adapt_no_l2sp.returncode == 0, adapt_no_l2sp.stderr
    adapt_l2sp = run_adapt(l2sp_out_dir, ["--l2sp-lambda", "1000.0"])
    assert adapt_l2sp.returncode == 0, adapt_l2sp.stderr
    assert (l2sp_out_dir / "model.pth").exists()

    def final_train_loss(out_dir: Path) -> str:
        with (out_dir / "metrics.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        return rows[-1][1]  # "Train Loss" column

    assert final_train_loss(no_l2sp_out_dir) != final_train_loss(l2sp_out_dir), (
        "a large --l2sp-lambda must measurably change the training trajectory "
        "relative to an otherwise-identical adapt run with the penalty off"
    )

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


@pytest.mark.integration
def test_l2sp_lambda_changes_final_metrics_on_the_real_phonebook_pipeline(
    run_script, has_torch, tmp_path: Path
) -> None:
    """Every other --l2sp-lambda behavioral test uses either a tiny 4-dim
    synthetic model (test_l2sp.py) or an 8-line dataset (the test above) --
    never the real 128d/4head/4layer phonebook architecture or the real
    Phase-A checkpoint the actual sweep adapts from. A regression that only
    shows up at that scale (e.g. a checkpoint-shape assumption baked into
    old_params, or the penalty being numerically negligible relative to real
    gradients) would pass every other test in this suite."""
    if not has_torch:
        pytest.skip("torch is not installed in this environment")

    checkpoint = ROOT / "phonebook" / "out_phaseA" / "model.pth"
    data_dir = ROOT / "phonebook" / "data_phaseB"
    config = ROOT / "phonebook" / "config" / "phonebook.py"
    if not checkpoint.exists() or not data_dir.exists():
        pytest.skip(
            "phonebook/out_phaseA and phonebook/data_phaseB are gitignored build "
            "artifacts (generated per phonebook/README.md's documented pipeline), "
            "not present in a fresh checkout"
        )

    def run_adapt(out_dir: Path, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
        return run_script(
            "train.py",
            [
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),
                "--adapt-from", str(checkpoint),
                str(config),
                "--lr", "0.0001",
                "--epochs", "5",
                "--seed", "42",
                *extra_args,
            ],
            cwd=tmp_path,
        )

    # A realistic sweep-scale lambda (0.1) needs dozens of epochs before the
    # penalty is large enough to show up at real lr=0.0001 (confirmed in
    # phonebook/sweep_lr/README.md's Part 4: the effect window doesn't start
    # until ~epoch 48). This test only needs to prove the mechanism has real
    # teeth at real scale within a few fast epochs, so it uses a much larger
    # lambda -- same "exaggerate to get a fast, robust signal" approach as
    # the boundary test in test_l2sp.py.
    no_l2sp_out_dir = tmp_path / "real_adapt_no_l2sp"
    l2sp_out_dir = tmp_path / "real_adapt_l2sp"
    adapt_no_l2sp = run_adapt(no_l2sp_out_dir, [])
    assert adapt_no_l2sp.returncode == 0, adapt_no_l2sp.stderr
    adapt_l2sp = run_adapt(l2sp_out_dir, ["--l2sp-lambda", "100.0"])
    assert adapt_l2sp.returncode == 0, adapt_l2sp.stderr

    def final_val_loss(out_dir: Path) -> str:
        with (out_dir / "metrics.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        return rows[-1][2]  # "Val Loss" column

    assert final_val_loss(no_l2sp_out_dir) != final_val_loss(l2sp_out_dir), (
        "--l2sp-lambda must measurably change training on the real phonebook "
        "architecture and the real Phase-A checkpoint, not just on tiny fixtures"
    )
