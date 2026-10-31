from __future__ import annotations

import pytest

from completion_core.modeling import CompletionTransformer
from phonebook import param_count as pc


def measured(vocab_size: int, seq_len: int, d_model: int, n_heads: int, n_layers: int) -> int:
    model = CompletionTransformer(vocab_size, seq_len, d_model, n_heads, n_layers)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("vocab,seq,d,heads,layers", [
    (67, 30, 64, 2, 2),
    (67, 31, 64, 2, 2),
    (67, 29, 8, 2, 2),
    (67, 40, 16, 2, 2),
    (67, 35, 32, 2, 2),
    (60, 25, 16, 2, 1),
    (67, 30, 32, 2, 3),
])
def test_closed_form_equals_the_instantiated_model(vocab: int, seq: int, d: int, heads: int, layers: int) -> None:
    assert pc.expected_param_count(vocab, seq, d, layers) == measured(vocab, seq, d, heads, layers)


def test_head_count_does_not_change_the_parameter_count() -> None:
    counts = {measured(67, 30, 16, heads, 2) for heads in (1, 2, 4)}
    assert counts == {pc.expected_param_count(67, 30, 16, 2)}


def test_it_reproduces_the_counts_quoted_in_the_results_notes() -> None:
    # RESULTS.md: 110,528 parameters for the 20-entry phonebook (context 29); the 200-entry probe printed 110,656 (context 31).
    assert pc.expected_param_count(67, 29, 64, 2) == 110_528
    assert pc.expected_param_count(67, 31, 64, 2) == 110_656


def test_only_the_context_length_moves_the_count_between_books_of_one_model() -> None:
    step = pc.expected_param_count(67, 31, 8, 2) - pc.expected_param_count(67, 30, 8, 2)
    assert step == 8  # one more position row of width d_model


@pytest.mark.parametrize("args", [(0, 30, 64, 2), (67, 0, 64, 2), (67, 30, 0, 2), (67, 30, 64, 0), (67, 30, 64.0, 2), (True, 30, 64, 2)])
def test_inputs_must_be_positive_integers(args: tuple) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        pc.expected_param_count(*args)
