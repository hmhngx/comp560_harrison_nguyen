from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import torch

from completion_core.modeling import CompletionTransformer
from completion_core.vocabulary import Vocabulary


@torch.no_grad()
def generate_completion(
    lhs: str,
    model: CompletionTransformer,
    vocab: Vocabulary,
    max_new_tokens: int | None,
    max_supported_len: int,
    device: str,
) -> CompletionResult:
    """Generate until newline or a stated limit, without using the answer length."""
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive or None (automatic).")
    if max_supported_len < 1:
        raise ValueError("max_supported_len must be positive.")
    lhs_tokens = vocab.tokenize_string(lhs)
    tokens = [vocab.stoi[tok] for tok in lhs_tokens] + [vocab.equal_id]

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    budget = max_new_tokens if max_new_tokens is not None else max_supported_len
    generated: list[int] = []

    while len(generated) < budget and x.size(1) < max_supported_len:
        logits = model(x)
        next_id = torch.argmax(logits[0, -1, :]).item()
        generated.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == vocab.eot_id:
            return CompletionResult(
                text="".join(vocab.itos[i] for i in generated[:-1]),
                token_ids=tuple(generated), stop_reason="newline",
            )

    return CompletionResult(
        text="".join(vocab.itos[i] for i in generated),
        token_ids=tuple(generated),
        stop_reason="context" if x.size(1) >= max_supported_len else "budget",
    )


@dataclass(frozen=True)
class CompletionResult:
    text: str
    token_ids: tuple[int, ...]
    stop_reason: Literal["newline", "budget", "context"]


def is_exact_completion(result: CompletionResult, ground_truth: str, vocab: Vocabulary) -> bool:
    """Require the exact target tokens and newline, including legitimate 2-char padding."""
    expected = tuple(vocab.stoi[token] for token in vocab.tokenize_string(ground_truth))
    return result.stop_reason == "newline" and result.token_ids == expected + (vocab.eot_id,)


def complete_sequence(
    lhs: str, model: CompletionTransformer, vocab: Vocabulary,
    max_new_tokens: int, max_supported_len: int, device: str,
) -> str:
    """Compatibility string interface for interactive and task-specific callers."""
    result = generate_completion(lhs, model, vocab, max_new_tokens, max_supported_len, device)
    return result.text.replace("_", "")


def evaluate_file(
    eval_file: Path,
    model: CompletionTransformer,
    vocab: Vocabulary,
    max_new_tokens: int | None,
    max_supported_len: int,
    device: str,
) -> tuple[int, int]:
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    print(f"Evaluating: {eval_file}\n")

    correct = 0
    total = 0

    with eval_file.open(encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "=" not in line:
                print(f"  [SKIP] Line {line_num}: missing '=' - '{raw_line.strip()}'")
                continue

            lhs, ground_truth = line.split("=", maxsplit=1)
            lhs_clean = lhs.replace("_", "")
            gt_clean = ground_truth.replace("_", "")

            result = generate_completion(
                lhs=lhs_clean,
                model=model,
                vocab=vocab,
                max_new_tokens=max_new_tokens,
                max_supported_len=max_supported_len,
                device=device,
            )

            if is_exact_completion(result, gt_clean, vocab):
                correct += 1
            else:
                print(
                    f"  [MISS] Prompt: {lhs_clean + '=':<6}  "
                    f"Expected: {gt_clean:<5}  Got: {result.text!r} "
                    f"(stop={result.stop_reason})"
                )
            total += 1

    return correct, total
