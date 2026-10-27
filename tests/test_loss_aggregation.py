from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from completion_core.training import evaluate_epoch_metrics, l2sp_penalty, run_epoch


class LookupLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[6., -4.], [-4., 6.]], dtype=torch.float64))

    def forward(self, x):
        return self.logits[x]


def examples():
    # Three records, unequal supervised lengths, deliberately different losses.
    return TensorDataset(
        torch.tensor([[0, 0, 1], [1, 0, 1], [1, 1, 0]]),
        torch.tensor([[0, 0, -100], [1, 0, -100], [0, 0, 0]]),
    )


def epoch(model, loader, criterion=None, optimizer=None, **kwargs):
    return run_epoch(model, loader, 2, criterion or nn.CrossEntropyLoss(ignore_index=-100),
                     optimizer, 0.75, "cpu", 999, 1, 1, **kwargs)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_evaluation_loss_is_mean_over_all_supervised_tokens(batch_size):
    model = LookupLM()
    loader = DataLoader(examples(), batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # Five correct targets (loss log(1+exp(-10))), two incorrect (+10 each).
    expected = math.log1p(math.exp(-10)) + 20 / 7
    assert epoch(model, loader) == pytest.approx(expected, abs=1e-12)
    metrics = evaluate_epoch_metrics(model, loader, 2, criterion, "cpu")
    assert metrics.loss == pytest.approx(expected, abs=1e-12)
    assert (metrics.token_correct, metrics.token_total) == (5, 7)
    assert (metrics.sequence_correct, metrics.sequence_total) == (2, 3)


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.AdamW])
@pytest.mark.parametrize("penalty", [0., 0.3])
def test_training_reports_online_token_loss_without_changing_any_updates(optimizer_type, penalty):
    actual = LookupLM()
    reference = copy.deepcopy(actual)
    anchor = {name: p.detach().clone() + 0.25 for name, p in actual.named_parameters()}
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    opt_actual = optimizer_type(actual.parameters(), lr=0.05, weight_decay=0.1)
    opt_reference = optimizer_type(reference.parameters(), lr=0.05, weight_decay=0.1)
    weighted_loss = 0.
    tokens = 0
    loader = DataLoader(examples(), batch_size=2)
    # Independent reference: original batch objective, clipping and optimizer steps.
    for x, y in loader:
        task = criterion(reference(x).view(-1, 2), y.view(-1))
        count = (y != -100).sum().item()
        weighted_loss += task.item() * count
        tokens += count
        objective = task + penalty * l2sp_penalty(reference, anchor)
        opt_reference.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm_(reference.parameters(), 0.75)
        opt_reference.step()
    reported = epoch(actual, loader, optimizer=opt_actual, old_params=anchor, l2sp_lambda=penalty)
    assert reported == pytest.approx(weighted_loss / tokens, abs=1e-12)
    for name, tensor in actual.state_dict().items():
        assert torch.equal(tensor, reference.state_dict()[name]), name
    for p_actual, p_reference in zip(actual.parameters(), reference.parameters(), strict=True):
        assert torch.equal(p_actual.grad, p_reference.grad)
        for key, value in opt_actual.state[p_actual].items():
            other = opt_reference.state[p_reference][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), key
            else:
                assert value == other


@pytest.mark.parametrize("entry", ["epoch", "metrics"])
@pytest.mark.parametrize("case", ["empty", "all_masked", "sum", "weighted", "other_ignore_index"])
def test_undefined_means_and_unsupported_criteria_fail_explicitly(entry, case):
    model = LookupLM()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loader = DataLoader(examples(), batch_size=2)
    if case == "empty":
        loader = []
    elif case == "all_masked":
        loader = [(torch.zeros((1, 2), dtype=torch.long), torch.full((1, 2), -100))]
    elif case == "sum":
        criterion = nn.CrossEntropyLoss(reduction="sum")
    elif case == "weighted":
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 2.], dtype=torch.float64))
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        if entry == "epoch":
            epoch(model, loader, criterion=criterion, optimizer=torch.optim.AdamW(model.parameters()))
        else:
            evaluate_epoch_metrics(model, loader, 2, criterion, "cpu")
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
