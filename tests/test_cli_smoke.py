from __future__ import annotations

from pathlib import Path

import pytest


def test_argparse_scripts_show_help(run_script) -> None:
    scripts = [
        "make_inputs_add.py",
        "make_inputs_capital.py",
        "make_inputs_comp_data.py",
        "make_inputs_rev.py",
        "prepare_inputs.py",
    ]

    for script in scripts:
        result = run_script(script, ["-h"])
        combined = (result.stdout + result.stderr).lower()
        assert result.returncode == 0, f"help failed for {script}: {combined}"
        assert "usage" in combined, f"no usage text for {script}"


def test_torch_scripts_show_help_or_fail_with_missing_dependency(run_script, has_torch) -> None:
    scripts = [
        "generate.py",
        "generate_one.py",
        "generate_reverse.py",
        "train.py",
    ]

    for script in scripts:
        result = run_script(script, ["-h"])
        combined = (result.stdout + result.stderr).lower()
        if has_torch:
            assert result.returncode == 0, f"help failed for {script}: {combined}"
            assert "usage" in combined, f"no usage text for {script}"
            if script == "train.py":
                assert "--adapt-from" in combined
                assert "--accuracy-interval" in combined
                assert "--seed" in combined
                assert "--l2sp-lambda" in combined
        else:
            assert result.returncode != 0
            assert "no module named 'torch'" in combined


def test_generate_all_fails_cleanly_without_checkpoint(run_script, has_torch, tmp_path: Path) -> None:
    if not has_torch:
        pytest.skip("torch is not installed in this environment")

    result = run_script("generate_all_additions.py", [], cwd=tmp_path)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "could not find model weights" in combined.lower()


def test_generate_one_fails_cleanly_without_checkpoint(run_script, has_torch, tmp_path: Path) -> None:
    if not has_torch:
        pytest.skip("torch is not installed in this environment")

    result = run_script("generate_one.py", ["ab"], cwd=tmp_path)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "could not find model weights" in combined.lower()


def test_generate_fails_cleanly_with_missing_data_and_weights(run_script, has_torch, tmp_path: Path) -> None:
    if not has_torch:
        pytest.skip("torch is not installed in this environment")

    result = run_script(
        "generate.py",
        ["--out-dir", str(tmp_path / "missing_out"), "--data-dir", str(tmp_path / "missing_data")],
        cwd=tmp_path,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "not found" in combined.lower()


def test_train_fails_cleanly_with_missing_meta(run_script, has_torch, tmp_path: Path) -> None:
    if not has_torch:
        pytest.skip("torch is not installed in this environment")

    result = run_script(
        "train.py",
        ["--data-dir", str(tmp_path / "missing_data"), "--out-dir", str(tmp_path / "out")],
        cwd=tmp_path,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "vocabulary file not found" in combined.lower()


def test_train_fails_cleanly_with_l2sp_lambda_but_no_adapt_from(run_script, has_torch, tmp_path: Path) -> None:
    if not has_torch:
        pytest.skip("torch is not installed in this environment")

    # Deliberately uses a nonexistent data-dir too, to prove this is rejected
    # by argument validation alone, before any file access -- not a
    # coincidental failure from the missing data.
    result = run_script(
        "train.py",
        [
            "--data-dir", str(tmp_path / "missing_data"),
            "--out-dir", str(tmp_path / "out"),
            "--l2sp-lambda", "0.1",
        ],
        cwd=tmp_path,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--l2sp-lambda" in combined
    assert "--adapt-from" in combined


def test_train_fails_cleanly_with_negative_l2sp_lambda(run_script, has_torch, tmp_path: Path) -> None:
    if not has_torch:
        pytest.skip("torch is not installed in this environment")

    # Negative lambda is rejected even with a syntactically-valid --adapt-from
    # path, to prove this is a distinct check from the missing-checkpoint one.
    result = run_script(
        "train.py",
        [
            "--data-dir", str(tmp_path / "missing_data"),
            "--out-dir", str(tmp_path / "out"),
            "--l2sp-lambda", "-0.1",
            "--adapt-from", str(tmp_path / "checkpoint.pth"),
        ],
        cwd=tmp_path,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--l2sp-lambda" in combined
