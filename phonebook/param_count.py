"""Closed-form trainable-parameter count of completion_core.modeling.CompletionTransformer.

    embeddings   vocab*d + seq*d
    each layer   attention 4d^2+4d, feed-forward (width 4d) 8d^2+5d, two norms 4d  ->  12d^2+13d
    final norm   2d
    output head  vocab*d (no bias)

    total = 2*vocab*d + seq*d + layers*(12*d^2 + 13*d) + 2*d

The head count does not enter: heads only split the same d x d projections. The training script
sizes the context (seq) from the longest line of the training book, so for a fixed model the count
moves only through seq, by d per extra position.
"""
from __future__ import annotations


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def expected_param_count(vocab_size: int, seq_len: int, d_model: int, n_layers: int) -> int:
    v, s = _positive_int(vocab_size, "vocab_size"), _positive_int(seq_len, "seq_len")
    d, layers = _positive_int(d_model, "d_model"), _positive_int(n_layers, "n_layers")
    return 2 * v * d + s * d + layers * (12 * d * d + 13 * d) + 2 * d
