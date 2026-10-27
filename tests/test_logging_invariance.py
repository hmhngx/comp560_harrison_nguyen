"""Accuracy measurement must not change training order or learned weights."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from completion_core.prep import build_char_vocab, encode_chars, save_outputs
import train


@pytest.fixture
def tiny_training_data(tmp_path: Path) -> Path:
    text = "a=12\nb=21\nc=11\nd=22\ne=12\nf=21\ng=11\nh=22\n"
    stoi, itos = build_char_vocab(text, "_")
    data = tmp_path / "data"
    save_outputs(data, encode_chars(text, stoi), encode_chars(text, stoi), stoi, itos)
    return data


class RecordingLoader:
    def __init__(self, loader, order):
        self.loader, self.order = loader, order

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, y in self.loader:
            self.order.append(x.tolist())
            yield x, y


def run_training(monkeypatch, data: Path, output: Path, *, batch_size: int,
                 interval: int, epochs: int = 10, extra_args=()):
    orders, states = [], []
    original_epoch = train.run_epoch

    def record_epoch(**kwargs):
        is_training = kwargs["optimizer"] is not None
        if is_training:
            order = []
            orders.append(order)
            kwargs["loader"] = RecordingLoader(kwargs["loader"], order)
        loss = original_epoch(**kwargs)
        if is_training:
            states.append({name: value.detach().clone()
                           for name, value in kwargs["model"].state_dict().items()})
        return loss

    with monkeypatch.context() as patch:
        patch.setattr(train, "run_epoch", record_epoch)
        patch.setattr(sys, "argv", [
            "train.py", "--device", "cpu", "--data-dir", str(data), "--out-dir", str(output),
            "--embedding-dim", "8", "--n-heads", "2", "--n-layers", "1",
            "--batch-size", str(batch_size), "--epochs", str(epochs),
            "--accuracy-interval", str(interval), "--seed", "123", "--lr", "0.01", *extra_args,
        ])
        train.main()
    payload = torch.load(output / "model.pth", map_location="cpu", weights_only=True)
    return orders, states, payload


@pytest.mark.parametrize("batch_size", [16, 3], ids=["full_batch", "partial_last_batch"])
@pytest.mark.integration
def test_accuracy_interval_changes_neither_batches_nor_any_epoch_weights(
    monkeypatch, tiny_training_data, tmp_path, batch_size,
):
    reference = run_training(monkeypatch, tiny_training_data, tmp_path / "interval1",
                             batch_size=batch_size, interval=1)
    for interval in (5, 10):
        candidate = run_training(monkeypatch, tiny_training_data, tmp_path / f"interval{interval}",
                                 batch_size=batch_size, interval=interval)
        assert candidate[0] == reference[0], f"logging every {interval} epochs changed training batches"
        for epoch, (expected, actual) in enumerate(zip(reference[1], candidate[1]), start=1):
            assert expected.keys() == actual.keys()
            for name in expected:
                assert torch.equal(expected[name], actual[name]), f"epoch {epoch}, parameter {name}"
        assert candidate[2]["training_config"] == {"seed": 123, "loader_rng": "isolated"}


@pytest.mark.integration
def test_short_training_matches_the_prefix_even_with_an_extra_final_evaluation(
    monkeypatch, tiny_training_data, tmp_path,
):
    short = run_training(monkeypatch, tiny_training_data, tmp_path / "short",
                         batch_size=3, interval=10, epochs=5)
    long = run_training(monkeypatch, tiny_training_data, tmp_path / "long",
                        batch_size=3, interval=10, epochs=10)
    assert short[0] == long[0][:5]
    for expected, actual in zip(short[1], long[1]):
        assert all(torch.equal(expected[name], actual[name]) for name in expected)


def test_evaluation_iterators_leave_global_and_training_rng_unchanged():
    dataset = TensorDataset(torch.arange(8))
    training, train_eval, validation = train.build_data_loaders(
        dataset, dataset, batch_size=3, seed=123,
    )
    list(training)
    global_before = torch.get_rng_state().clone()
    training_before = training.generator.get_state().clone()
    list(train_eval)
    list(validation)
    list(validation)
    assert torch.equal(global_before, torch.get_rng_state())
    assert torch.equal(training_before, training.generator.get_state())


@pytest.mark.integration
def test_legacy_mode_matches_original_training_weights(monkeypatch, tiny_training_data, tmp_path):
    legacy = run_training(monkeypatch, tiny_training_data, tmp_path / "legacy",
                          batch_size=3, interval=5, extra_args=("--loader-rng", "legacy"))

    def original_loaders(train_dataset, val_dataset, *, batch_size, seed, loader_rng):
        return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
                DataLoader(val_dataset, batch_size=batch_size, shuffle=False))

    with monkeypatch.context() as patch:
        patch.setattr(train, "build_data_loaders", original_loaders)
        original = run_training(patch, tiny_training_data, tmp_path / "original",
                                batch_size=3, interval=5, extra_args=("--loader-rng", "legacy"))
    assert legacy[0] == original[0]
    for expected, actual in zip(original[1], legacy[1]):
        assert all(torch.equal(expected[name], actual[name]) for name in expected)
    assert legacy[2]["training_config"] == {"seed": 123, "loader_rng": "legacy"}


def test_loader_rng_config_and_cli_override_are_validated(monkeypatch, tmp_path):
    config = tmp_path / "config.py"
    config.write_text("loader_rng = 'legacy'\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config)])
    assert train.parse_args().loader_rng == "legacy"
    monkeypatch.setattr(sys, "argv", ["train.py", str(config), "--loader-rng=isolated"])
    assert train.parse_args().loader_rng == "isolated"
    config.write_text("loader_rng = 'typo'\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config)])
    with pytest.raises(ValueError, match="loader-rng"):
        train.parse_args()
    monkeypatch.setattr(sys, "argv", ["train.py", "--loader-rng", "typo"])
    with pytest.raises(SystemExit) as error:
        train.parse_args()
    assert error.value.code == 2
