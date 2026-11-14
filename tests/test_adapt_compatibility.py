"""--adapt-from must refuse a checkpoint whose recorded architecture differs from the model being built.

A checkpoint trained with 2 heads has exactly the same tensor shapes as one trained with 4 heads
(d_model unchanged), so load_state_dict cannot tell them apart; only the recorded model_config can.
A field the checkpoint does not record is reported in a warning, not skipped in silence, and a run that is
refused leaves no output folder behind.
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
    rejected = tmp_path / "four_heads"
    with pytest.raises(ValueError, match="n_heads"):
        run(monkeypatch, tiny_data, rejected, n_heads=4, adapt_from=tmp_path / "two_heads" / "model.pth")
    assert not rejected.exists(), "a refused run must not leave an empty output folder behind"


@pytest.mark.integration
def test_a_matching_checkpoint_still_adapts(monkeypatch, tiny_data, tmp_path, capsys) -> None:
    run(monkeypatch, tiny_data, tmp_path / "base", n_heads=2)
    run(monkeypatch, tiny_data, tmp_path / "next", n_heads=2, adapt_from=tmp_path / "base" / "model.pth")
    assert "adapting model from checkpoint" in capsys.readouterr().out
    assert (tmp_path / "next" / "model.pth").is_file() and (tmp_path / "next" / "metrics.csv").is_file()


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


@pytest.mark.integration
def test_a_missing_adapt_checkpoint_creates_no_output_folder(monkeypatch, tiny_data, tmp_path) -> None:
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        run(monkeypatch, tiny_data, out, n_heads=2, adapt_from=tmp_path / "nowhere" / "model.pth")
    assert not out.exists()


# ---- the check called directly, without a training run: each recorded field, and fields that are missing

FULL = {"d_model": 8, "n_heads": 2, "n_layers": 1, "seq_len": 5}


def _checkpoint(tmp_path: Path, model_config: dict | None) -> Path:
    path = tmp_path / "checkpoint.pth"
    payload = {"state_dict": {"weight": torch.zeros(1)}}
    if model_config is not None:
        payload["model_config"] = model_config
    torch.save(payload, path)
    return path


def _check(path: Path) -> None:
    train.check_adapt_compatibility(path, "cpu", embedding_dim=8, n_heads=2, n_layers=1, seq_len=5)


def test_a_checkpoint_that_matches_on_all_four_fields_passes_silently(tmp_path, capsys) -> None:
    _check(_checkpoint(tmp_path, dict(FULL)))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("key", sorted(FULL))
def test_a_mismatch_in_any_one_field_is_rejected_and_names_only_that_field(tmp_path, key) -> None:
    with pytest.raises(ValueError) as caught:
        _check(_checkpoint(tmp_path, dict(FULL, **{key: FULL[key] + 1})))
    message = str(caught.value)
    assert f"{key}: checkpoint {FULL[key] + 1} vs current {FULL[key]}" in message
    assert message.count(": checkpoint ") == 1


def test_every_mismatched_field_is_listed(tmp_path) -> None:
    with pytest.raises(ValueError) as caught:
        _check(_checkpoint(tmp_path, dict(FULL, n_heads=4, seq_len=9)))
    message = str(caught.value)
    assert "n_heads: checkpoint 4 vs current 2" in message and "seq_len: checkpoint 9 vs current 5" in message
    assert "d_model" not in message and "n_layers" not in message


def test_a_checkpoint_with_no_model_config_warns_that_it_cannot_be_verified(tmp_path, capsys) -> None:
    _check(_checkpoint(tmp_path, None))
    assert "records no model_config" in capsys.readouterr().out


@pytest.mark.parametrize("missing", [("n_heads",), ("d_model", "seq_len"), ("d_model", "n_heads", "n_layers", "seq_len")])
def test_a_model_config_that_leaves_fields_out_warns_and_names_them(tmp_path, capsys, missing) -> None:
    path = _checkpoint(tmp_path, {key: value for key, value in FULL.items() if key not in missing})
    _check(path)  # nothing recorded can be compared, so nothing is refused, but it must not pass in silence
    out = capsys.readouterr().out
    assert str(path) in out, "no warning was printed"
    warning = out.split(str(path), 1)[1]
    assert "cannot verify" in warning
    assert all(key in warning for key in missing)
    assert not any(key in warning for key in FULL if key not in missing)


def test_a_mismatch_is_refused_even_when_other_fields_are_missing(tmp_path) -> None:
    with pytest.raises(ValueError, match="n_heads: checkpoint 4 vs current 2"):
        _check(_checkpoint(tmp_path, {"n_heads": 4}))
