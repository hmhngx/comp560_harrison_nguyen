from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_lines(path: Path, strip: bool = False, drop_empty: bool = False) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Check the path and try again."
        )

    with path.open(encoding="utf-8") as fh:
        lines = fh.readlines()

    if strip:
        lines = [line.strip() for line in lines]
    if drop_empty:
        lines = [line for line in lines if line]

    return lines


def split_by_fraction[T](items: list[T], train_split: float) -> tuple[list[T], list[T]]:
    split = int(len(items) * train_split)
    return items[:split], items[split:]


def build_char_vocab(text: str, pad_token: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    if pad_token in chars:
        chars.remove(pad_token)
    chars.append(pad_token)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode_chars(text: str, stoi: dict[str, int]) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.uint16)


def build_token_vocab(tokenized_lines: list[list[str]], pad_token: str) -> tuple[dict[str, int], dict[int, str]]:
    unique_tokens = set()
    for tokens in tokenized_lines:
        unique_tokens.update(tokens)

    sorted_tokens = sorted(list(unique_tokens))
    if pad_token in sorted_tokens:
        sorted_tokens.remove(pad_token)
    sorted_tokens.append(pad_token)

    stoi = {tok: i for i, tok in enumerate(sorted_tokens)}
    itos = {i: tok for i, tok in enumerate(sorted_tokens)}
    return stoi, itos


def encode_token_lines(tokenized_lines: list[list[str]], stoi: dict[str, int]) -> np.ndarray:
    flat_ids: list[int] = []
    for tokens in tokenized_lines:
        for token in tokens:
            flat_ids.append(stoi[token])
    return np.array(flat_ids, dtype=np.uint16)


def save_outputs(
    out_dir: Path,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    stoi: dict[str, int],
    itos: dict[int, str],
) -> None:
    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError("Training and evaluation data must both be non-empty; check the split or input files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    meta = {"vocab_size": len(stoi), "stoi": stoi, "itos": itos}
    with (out_dir / "meta.pkl").open("wb") as fh:
        pickle.dump(meta, fh)
