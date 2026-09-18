from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from completion_core.training import l2sp_penalty, run_epoch

ROOT = Path(__file__).resolve().parents[1]


class TinyLMModel(nn.Module):
    """Minimal stand-in for CompletionTransformer: same (LongTensor -> logits) contract."""

    def __init__(self, vocab_size: int, d: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(x))


def _make_model(vocab_size: int = 5, d: int = 4, seed: int = 0) -> TinyLMModel:
    torch.manual_seed(seed)
    return TinyLMModel(vocab_size, d)


# ---------------------------------------------------------------------------
# l2sp_penalty: hand-computed values, no training loop involved.
# ---------------------------------------------------------------------------

def test_l2sp_penalty_matches_hand_computed_sum_of_squares() -> None:
    model = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.copy_(torch.tensor([3.0]))

    old_params = {
        "weight": torch.tensor([[0.0, 1.0]]),
        "bias": torch.tensor([1.0]),
    }
    # (1-0)^2 + (2-1)^2 + (3-1)^2 = 1 + 1 + 4 = 6
    result = l2sp_penalty(model, old_params)
    assert torch.equal(result, torch.tensor(6.0))


def test_l2sp_penalty_weighted_by_lambda_matches_hand_computed_value() -> None:
    model = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.copy_(torch.tensor([3.0]))

    old_params = {
        "weight": torch.tensor([[0.0, 1.0]]),
        "bias": torch.tensor([1.0]),
    }
    l2sp_lambda = 0.1
    # lambda * sum-of-squares = 0.1 * 6 = 0.6
    result = l2sp_lambda * l2sp_penalty(model, old_params)
    assert torch.isclose(result, torch.tensor(0.6))


def test_l2sp_penalty_ignores_params_missing_from_old_params() -> None:
    model = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.copy_(torch.tensor([100.0]))  # huge drift, but excluded below

    old_params = {"weight": torch.tensor([[0.0, 1.0]])}  # "bias" deliberately absent
    # Only weight contributes: (1-0)^2 + (2-1)^2 = 2
    result = l2sp_penalty(model, old_params)
    assert torch.equal(result, torch.tensor(2.0))


def test_l2sp_penalty_is_zero_when_params_match_snapshot_exactly() -> None:
    model = nn.Linear(2, 1, bias=True)
    old_params = {name: p.detach().clone() for name, p in model.named_parameters()}
    result = l2sp_penalty(model, old_params)
    assert torch.equal(result, torch.tensor(0.0))


# ---------------------------------------------------------------------------
# run_epoch: the critical invariant -- the *returned/logged* loss must be the
# pure task loss, identical with or without the penalty. Only the gradient
# step may differ.
# ---------------------------------------------------------------------------

def test_run_epoch_returned_loss_is_unaffected_by_l2sp_penalty() -> None:
    vocab_size = 5
    model = _make_model(vocab_size)
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    x = torch.tensor([[0, 1, 2], [3, 4, 0]])
    y = torch.tensor([[1, 2, 3], [4, 0, 1]])
    loader = [(x, y)]
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = optim.SGD(model.parameters(), lr=0.5)
    loss_no_penalty = run_epoch(
        model=model, loader=loader, vocab_size=vocab_size, criterion=criterion,
        optimizer=optimizer, grad_clip=10.0, device="cpu",
        log_interval=9999, epoch=1, total_epochs=1,
        old_params=None, l2sp_lambda=0.0,
    )
    params_no_penalty = {k: v.clone() for k, v in model.state_dict().items()}

    # Reset to the identical starting point, then re-run with a strong penalty
    # pulling toward a snapshot that's deliberately far from the current params.
    model.load_state_dict(initial_state)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    old_params = {name: p.detach().clone() + 5.0 for name, p in model.named_parameters()}
    loss_with_penalty = run_epoch(
        model=model, loader=loader, vocab_size=vocab_size, criterion=criterion,
        optimizer=optimizer, grad_clip=10.0, device="cpu",
        log_interval=9999, epoch=1, total_epochs=1,
        old_params=old_params, l2sp_lambda=2.0,
    )
    params_with_penalty = {k: v.clone() for k, v in model.state_dict().items()}

    assert loss_no_penalty == loss_with_penalty, (
        "the returned epoch loss must be the pure task loss regardless of l2sp_lambda"
    )
    assert any(
        not torch.equal(params_no_penalty[k], params_with_penalty[k])
        for k in initial_state
    ), "the penalty must actually influence the gradient step, or l2sp_lambda is a no-op"


def test_run_epoch_val_pass_semantics_unchanged_when_old_params_omitted() -> None:
    """optimizer=None (the val pass) must still work with the new kwargs defaulted."""
    vocab_size = 5
    model = _make_model(vocab_size)
    x = torch.tensor([[0, 1, 2], [3, 4, 0]])
    y = torch.tensor([[1, 2, 3], [4, 0, 1]])
    loader = [(x, y)]
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    loss = run_epoch(
        model=model, loader=loader, vocab_size=vocab_size, criterion=criterion,
        optimizer=None, grad_clip=10.0, device="cpu",
        log_interval=9999, epoch=1, total_epochs=1,
    )
    after = {k: v.clone() for k, v in model.state_dict().items()}
    assert isinstance(loss, float)
    assert all(torch.equal(before[k], after[k]) for k in before), "val pass must never update weights"


# ---------------------------------------------------------------------------
# Boundary condition: a large l2sp_lambda should visibly suppress drift from
# the frozen snapshot, relative to l2sp_lambda=0 under the same optimizer.
# ---------------------------------------------------------------------------

def test_large_l2sp_lambda_suppresses_drift_relative_to_no_penalty() -> None:
    vocab_size = 5
    base_model = _make_model(vocab_size)
    old_params = {name: p.detach().clone() for name, p in base_model.named_parameters()}

    # Targets chosen to disagree with the model's initial predictions, so
    # there's a real task-loss gradient pulling parameters away from old_params.
    x = torch.tensor([[0, 1, 2], [3, 4, 0]])
    y = torch.tensor([[4, 4, 4], [4, 4, 4]])
    loader = [(x, y)]
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    lr = 0.05
    steps = 10

    def total_drift(model: nn.Module) -> float:
        return sum(
            (p.detach() - old_params[name]).pow(2).sum().item()
            for name, p in model.named_parameters()
        )

    model_small = copy.deepcopy(base_model)
    opt_small = optim.SGD(model_small.parameters(), lr=lr)
    for _ in range(steps):
        run_epoch(
            model=model_small, loader=loader, vocab_size=vocab_size, criterion=criterion,
            optimizer=opt_small, grad_clip=10.0, device="cpu",
            log_interval=9999, epoch=1, total_epochs=1,
            old_params=old_params, l2sp_lambda=0.0,
        )
    drift_no_penalty = total_drift(model_small)

    model_large = copy.deepcopy(base_model)
    opt_large = optim.SGD(model_large.parameters(), lr=lr)
    for _ in range(steps):
        run_epoch(
            model=model_large, loader=loader, vocab_size=vocab_size, criterion=criterion,
            optimizer=opt_large, grad_clip=10.0, device="cpu",
            log_interval=9999, epoch=1, total_epochs=1,
            old_params=old_params, l2sp_lambda=5.0,
        )
    drift_large_lambda = total_drift(model_large)

    assert drift_no_penalty > 1e-6, "test is meaningless if the no-penalty baseline didn't drift at all"
    assert drift_large_lambda < 0.5 * drift_no_penalty, (
        f"large l2sp_lambda should visibly suppress drift: "
        f"no_penalty={drift_no_penalty!r} vs large_lambda={drift_large_lambda!r}"
    )


# ---------------------------------------------------------------------------
# CLI wiring: --l2sp-lambda must reach cfg.l2sp_lambda. This is the exact bug
# class flagged in review -- add_argument without the sys.argv override line
# silently makes the flag a no-op.
# ---------------------------------------------------------------------------

def _parse_train_args(monkeypatch, argv: list[str]):
    """Import train.py (a repo-root script, not a package) and call its
    parse_args() under a monkeypatched sys.argv. sys.path.insert is a no-op
    after the first call (module caching) but is cheap and kept local to this
    one helper rather than repeated at every call site."""
    sys.path.insert(0, str(ROOT))
    import train as train_module

    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    return train_module.parse_args()


def test_parse_args_l2sp_lambda_flag_sets_cfg(monkeypatch) -> None:
    # --adapt-from is required whenever --l2sp-lambda > 0 (see the dedicated
    # rejection test below); the path need not exist yet at parse time.
    cfg = _parse_train_args(
        monkeypatch, ["--l2sp-lambda", "0.01", "--adapt-from", "some/checkpoint.pth"]
    )
    assert cfg.l2sp_lambda == 0.01


def test_parse_args_l2sp_lambda_defaults_to_zero(monkeypatch) -> None:
    cfg = _parse_train_args(monkeypatch, [])
    assert cfg.l2sp_lambda == 0.0


def test_parse_args_rejects_l2sp_lambda_without_adapt_from(monkeypatch) -> None:
    try:
        _parse_train_args(monkeypatch, ["--l2sp-lambda", "0.01"])
        assert False, "expected ValueError for --l2sp-lambda without --adapt-from"
    except ValueError as exc:
        assert "--adapt-from" in str(exc)


def test_parse_args_rejects_negative_l2sp_lambda_without_adapt_from(monkeypatch) -> None:
    try:
        _parse_train_args(monkeypatch, ["--l2sp-lambda", "-0.1"])
        assert False, "expected ValueError for negative --l2sp-lambda"
    except ValueError as exc:
        assert "--l2sp-lambda" in str(exc)


def test_parse_args_rejects_negative_l2sp_lambda_even_with_adapt_from(monkeypatch) -> None:
    """A negative lambda is never valid, regardless of --adapt-from -- distinct
    failure mode from the missing-checkpoint case, must not be masked by it."""
    try:
        _parse_train_args(
            monkeypatch, ["--l2sp-lambda", "-0.1", "--adapt-from", "some/checkpoint.pth"]
        )
        assert False, "expected ValueError for negative --l2sp-lambda"
    except ValueError as exc:
        assert "--l2sp-lambda" in str(exc)


def test_parse_args_accepts_zero_l2sp_lambda_with_no_adapt_from(monkeypatch) -> None:
    """Boundary: 0.0 is the valid off-default and must not trip either check,
    with or without --adapt-from."""
    cfg = _parse_train_args(monkeypatch, ["--l2sp-lambda", "0"])
    assert cfg.l2sp_lambda == 0.0
