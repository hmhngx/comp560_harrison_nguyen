"""--adapt-from must refuse a checkpoint whose recorded architecture differs from the model being built.

A checkpoint trained with 2 heads has exactly the same tensor shapes as one trained with 4 heads
(d_model unchanged), so load_state_dict cannot tell them apart; only the recorded model_config can.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import train
from completion_core.prep import build_char_vocab, encode_chars, save_outputs

TEXT = "a=12\nb=21\nc=11\nd=22\ne=12\nf=21\ng=11\nh=22\n"


@pytest.fixture
def tiny_data(tmp_path: Path) -> Path:
    stoi, itos = build_char_vocab(TEXT, "_")
    data = tmp_path / "data"
    save_outputs(data, encode_chars(TEXT, stoi), encode_chars(TEXT, stoi), stoi, itos)
    return data


def run(monkeypatch, data: Path, out: Path, *, n_heads: int, adapt_from: Path | None = None) -> None:
    argv = ["train.py", "--device", "cpu", "--data-dir", str(data), "--out-dir", str(out),
            "--embedding-dim", "8", "--n-heads", str(n_heads), "--n-layers", "1", "--batch-size", "8",
            "--epochs", "1", "--accuracy-interval", "1", "--seed", "7", "--lr", "0.01"]
    if adapt_from is not None:
        argv += ["--adapt-from", str(adapt_from)]
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        train.main()


@pytest.mark.integration
def test_a_checkpoint_with_a_different_head_count_is_rejected(monkeypatch, tiny_data, tmp_path) -> None:
    run(monkeypatch, tiny_data, tmp_path / "two_heads", n_heads=2)
    with pytest.raises(ValueError, match="n_heads"):
        run(monkeypatch, tiny_data, tmp_path / "four_heads", n_heads=4, adapt_from=tmp_path / "two_heads" / "model.pth")


@pytest.mark.integration
def test_a_matching_checkpoint_still_adapts(monkeypatch, tiny_data, tmp_path, capsys) -> None:
    run(monkeypatch, tiny_data, tmp_path / "base", n_heads=2)
    run(monkeypatch, tiny_data, tmp_path / "next", n_heads=2, adapt_from=tmp_path / "base" / "model.pth")
    assert "adapting model from checkpoint" in capsys.readouterr().out


@pytest.mark.integration
def test_a_legacy_checkpoint_without_model_config_still_loads_with_a_warning(monkeypatch, tiny_data, tmp_path, capsys) -> None:
    run(monkeypatch, tiny_data, tmp_path / "base", n_heads=2)
    payload = torch.load(tmp_path / "base" / "model.pth", map_location="cpu")
    legacy = tmp_path / "legacy.pth"
    torch.save({"state_dict": payload["state_dict"]}, legacy)
    run(monkeypatch, tiny_data, tmp_path / "next", n_heads=2, adapt_from=legacy)
    out = capsys.readouterr().out
    assert "adapting model from checkpoint" in out
    assert "cannot verify" in out.lower()
