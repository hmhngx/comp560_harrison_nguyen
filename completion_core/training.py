from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from completion_core.vocabulary import Vocabulary


@dataclass
class EpochMetrics:
    loss: float
    token_accuracy: float
    sequence_accuracy: float
    token_correct: int
    token_total: int
    sequence_correct: int
    sequence_total: int


class SequenceDataset(Dataset):
    def __init__(self, bin_path, vocab: Vocabulary) -> None:
        raw_tokens = np.fromfile(bin_path, dtype=np.uint16).astype(np.int64)
        self.sequences = self._split_into_lines(raw_tokens, vocab.eot_id)
        self.max_len = max(len(s) for s in self.sequences)
        self.vocab = vocab

    @staticmethod
    def _split_into_lines(tokens: np.ndarray, eot_id: int) -> list[torch.Tensor]:
        eot_indices = np.where(tokens == eot_id)[0]
        sequences: list[torch.Tensor] = []
        start = 0
        for end in eot_indices:
            sequences.append(torch.tensor(tokens[start : end + 1]))
            start = end + 1
        return sequences

    def _build_targets(self, seq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = torch.full_like(x, fill_value=-100)
        shifted = seq[1:].clone()

        eq_positions = (x == self.vocab.equal_id).nonzero(as_tuple=True)[0]
        if len(eq_positions) == 0:
            raise ValueError("Sequence is missing the '=' delimiter.")

        eq_pos = eq_positions[0].item()
        y[eq_pos:] = shifted[eq_pos:]
        return y

    def _pad_to_length(
        self, tensor: torch.Tensor, target_len: int, pad_value: int
    ) -> torch.Tensor:
        shortfall = target_len - len(tensor)
        if shortfall <= 0:
            return tensor
        padding = torch.full((shortfall,), pad_value, dtype=torch.long)
        return torch.cat([tensor, padding])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        target_len = self.max_len - 1

        x = seq[:-1].clone()
        y = self._build_targets(seq, x)

        x = self._pad_to_length(x, target_len, self.vocab.pad_id)
        y = self._pad_to_length(y, target_len, pad_value=-100)

        return x, y


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    vocab_size: int,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    grad_clip: float,
    device: str,
    log_interval: int,
    epoch: int,
    total_epochs: int,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch {epoch}/{total_epochs} | "
                        # f"Batch {batch_idx}/{len(loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch_metrics(
    model: nn.Module,
    loader: DataLoader,
    vocab_size: int,
    criterion: nn.Module,
    device: str,
) -> EpochMetrics:
    model.eval()

    total_loss = 0.0
    token_correct = 0
    token_total = 0
    sequence_correct = 0
    sequence_total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            supervised_mask = y_batch != -100

            token_correct += ((preds == y_batch) & supervised_mask).sum().item()
            token_total += supervised_mask.sum().item()

            has_supervised = supervised_mask.any(dim=1)
            masked_match = ((preds == y_batch) | (~supervised_mask)).all(dim=1)
            sequence_correct += (masked_match & has_supervised).sum().item()
            sequence_total += has_supervised.sum().item()

    token_accuracy = (token_correct / token_total) if token_total > 0 else 0.0
    sequence_accuracy = (sequence_correct / sequence_total) if sequence_total > 0 else 0.0

    return EpochMetrics(
        loss=total_loss / len(loader),
        token_accuracy=token_accuracy,
        sequence_accuracy=sequence_accuracy,
        token_correct=token_correct,
        token_total=token_total,
        sequence_correct=sequence_correct,
        sequence_total=sequence_total,
    )
