"""Two properties the long capacity runs depend on.

1. A failing loss-plot step must not cost a finished run its weights (a Tk error was seen once in the suite).
2. A shorter run is the exact prefix of a longer one even when each epoch holds several shuffled batches,
   which is what lets one long run be read at several epoch budgets.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from completion_core.prep import build_char_vocab, encode_chars, save_outputs
import train

TEXT = "a=12\nb=21\nc=11\nd=22\ne=12\nf=21\ng=11\nh=22\n"


@pytest.fixture
def tiny_data(tmp_path: Path) -> Path:
    stoi, itos = build_char_vocab(TEXT, "_")
    data = tmp_path / "data"
    save_outputs(data, encode_chars(TEXT, stoi), encode_chars(TEXT, stoi), stoi, itos)
    return data


def run(monkeypatch, data: Path, out: Path, *, epochs: int, batch_size: int = 3, interval: int = 1, seed: int = 123) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", [
            "train.py", "--device", "cpu", "--data-dir", str(data), "--out-dir", str(out),
            "--embedding-dim", "8", "--n-heads", "2", "--n-layers", "1", "--batch-size", str(batch_size),
            "--epochs", str(epochs), "--accuracy-interval", str(interval), "--seed", str(seed), "--lr", "0.01"])
        train.main()


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.mark.integration
def test_a_failing_plot_step_does_not_lose_the_trained_model(monkeypatch, tiny_data, tmp_path) -> None:
    import matplotlib.pyplot as plt

    def broken(*args, **kwargs):
        raise RuntimeError("Tk could not start")

    monkeypatch.setattr(plt, "figure", broken)
    run(monkeypatch, tiny_data, tmp_path / "out", epochs=2)
    assert (tmp_path / "out" / "model.pth").is_file() and (tmp_path / "out" / "metrics.csv").is_file()


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [3, 8], ids=["three_batches_with_a_partial_last", "one_batch"])
def test_a_shorter_run_is_the_exact_prefix_of_a_longer_one(monkeypatch, tiny_data, tmp_path, batch_size) -> None:
    run(monkeypatch, tiny_data, tmp_path / "short", epochs=4, batch_size=batch_size)
    run(monkeypatch, tiny_data, tmp_path / "long", epochs=8, batch_size=batch_size)
    short, long_ = rows(tmp_path / "short" / "metrics.csv"), rows(tmp_path / "long" / "metrics.csv")
    assert [row["Epoch"] for row in short] == ["1", "2", "3", "4"]
    for early, late in zip(short, long_):
        assert {k: v for k, v in early.items() if k != "Training Time (sec)"} == \
               {k: v for k, v in late.items() if k != "Training Time (sec)"}


@pytest.mark.integration
def test_a_different_seed_changes_the_run_so_the_prefix_test_can_fail(monkeypatch, tiny_data, tmp_path) -> None:
    run(monkeypatch, tiny_data, tmp_path / "one", epochs=4, seed=1)
    run(monkeypatch, tiny_data, tmp_path / "two", epochs=4, seed=2)
    drop = lambda path: [{k: v for k, v in row.items() if k != "Training Time (sec)"} for row in rows(path)]
    assert drop(tmp_path / "one" / "metrics.csv") != drop(tmp_path / "two" / "metrics.csv")
