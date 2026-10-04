"""Evaluate A and B independently, with strict generation and teacher-forced metrics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader

from completion_core.inference import generate_completion, is_exact_completion
from completion_core.modeling import load_model_and_config
from completion_core.prep import encode_chars
from completion_core.training import SequenceDataset, evaluate_epoch_metrics
from completion_core.vocabulary import Vocabulary
from phonebook.prepare_phonebook import load_phonebook_lines


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_book(model, vocab: Vocabulary, source: Path, seq_len: int, device: str = "cpu") -> dict:
    lines = load_phonebook_lines(source)
    if not lines:
        raise ValueError(f"Evaluation book is empty: {source}")
    if vocab.is_2char:
        raise ValueError("Phonebook evaluation requires a one-character vocabulary.")
    if max(len(line) for line in lines) > seq_len:
        raise ValueError(f"Evaluation records exceed the checkpoint context: {source}")
    with tempfile.TemporaryDirectory() as temp:
        binary = Path(temp) / "book.bin"
        encode_chars("".join(lines), vocab.stoi).tofile(binary)
        loader = DataLoader(SequenceDataset(binary, vocab), batch_size=len(lines), shuffle=False)
        teacher = evaluate_epoch_metrics(model, loader, vocab.vocab_size,
                                         torch.nn.CrossEntropyLoss(ignore_index=-100), device)
    predictions = []
    stops = {"newline": 0, "budget": 0, "context": 0}
    for line in lines:
        lhs, target = line.rstrip("\n").split("=", 1)
        result = generate_completion(lhs, model, vocab, None, seq_len, device)
        correct = is_exact_completion(result, target, vocab)
        stops[result.stop_reason] += 1
        predictions.append({"prompt": lhs, "expected": target, "prediction": result.text,
                            "stop_reason": result.stop_reason, "correct": correct})
    correct = sum(row["correct"] for row in predictions)
    return {"source": str(source), "source_sha256": sha256(source),
            "teacher_forced": asdict(teacher),
            "generation": {"correct": correct, "total": len(lines), "accuracy": correct / len(lines),
                           "stop_counts": stops, "predictions": predictions}}


def evaluate_checkpoint(checkpoint: Path, meta: Path, books: dict[str, Path]) -> dict:
    vocab = Vocabulary.from_pickle(meta)
    model, cfg = load_model_and_config(checkpoint, vocab, "cpu")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "metadata": str(meta), "metadata_sha256": sha256(meta),
            "model_config": asdict(cfg), "device": "cpu", "generation_budget": "automatic_context_limit",
            "books": {name: evaluate_book(model, vocab, path, cfg.seq_len) for name, path in books.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--phonebook-a", type=Path, default=Path("phonebook/inputs/phonebookA.txt"))
    parser.add_argument("--phonebook-b", type=Path, default=Path("phonebook/inputs/phonebookB.txt"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.meta, {"A": args.phonebook_a, "B": args.phonebook_b})
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
