"""grad_clip = 0 must mean "no clipping" in the main trainer, as it already does in run_partial_replay.step.

PyTorch's clip_grad_norm_(params, max_norm=0) rescales every gradient to zero, so passing 0 straight
through silently freezes learning (only decoupled weight decay would still act). A negative or NaN value is a
configuration error: run_epoch must refuse it, not skip the clipping without a word.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from completion_core.modeling import CompletionTransformer
from completion_core.training import run_epoch

VOCAB = 6


def _tiny():
    torch.manual_seed(0)
    model = CompletionTransformer(vocab_size=VOCAB, seq_len=5, d_model=8, n_heads=2, n_layers=1)
    x = torch.randint(0, VOCAB, (4, 4))
    y = torch.randint(0, VOCAB, (4, 4))
    return model, DataLoader(TensorDataset(x, y), batch_size=4)


def _flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().clone() for p in model.parameters()])


def _one_epoch(grad_clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    model, loader = _tiny()
    before = _flat(model)
    # Plain SGD without weight decay: any parameter change can only come from the gradients.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    run_epoch(
        model=model, loader=loader, vocab_size=VOCAB, criterion=nn.CrossEntropyLoss(ignore_index=-100),
        optimizer=optimizer, grad_clip=grad_clip, device="cpu", log_interval=100, epoch=1, total_epochs=1,
    )
    return before, _flat(model)


def test_grad_clip_zero_does_not_freeze_the_model() -> None:
    before, after = _one_epoch(0.0)
    assert not torch.equal(before, after), "grad_clip=0 zeroed every gradient"


def test_grad_clip_zero_is_identical_to_a_clip_threshold_that_never_binds() -> None:
    _, off = _one_epoch(0.0)
    _, huge = _one_epoch(1e12)
    assert torch.equal(off, huge)


def test_a_binding_clip_threshold_still_changes_the_update() -> None:
    # Guards the guard: if clipping were skipped for every value, this would fail.
    _, off = _one_epoch(0.0)
    _, tight = _one_epoch(1e-6)
    assert not torch.equal(off, tight)


@pytest.mark.parametrize("bad", [-1.0, -1e-12, float("nan")])
def test_a_negative_or_nan_grad_clip_is_refused_rather_than_ignored(bad) -> None:
    with pytest.raises(ValueError, match="grad_clip"):
        _one_epoch(bad)


def test_evaluation_never_reads_grad_clip_so_it_is_not_validated_there() -> None:
    model, loader = _tiny()
    loss = run_epoch(
        model=model, loader=loader, vocab_size=VOCAB, criterion=nn.CrossEntropyLoss(ignore_index=-100),
        optimizer=None, grad_clip=None, device="cpu", log_interval=100, epoch=1, total_epochs=1,
    )
    assert loss > 0


def test_a_refused_grad_clip_changes_nothing() -> None:
    model, loader = _tiny()
    before = _flat(model)
    with pytest.raises(ValueError):
        run_epoch(
            model=model, loader=loader, vocab_size=VOCAB, criterion=nn.CrossEntropyLoss(ignore_index=-100),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1), grad_clip=-1.0, device="cpu",
            log_interval=100, epoch=1, total_epochs=1,
        )
    assert torch.equal(before, _flat(model)), "the refusal must come before any update"
