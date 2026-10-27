"""Evaluate pinned replay partitions against an explicitly checked training text.

This validates a prepared input's membership, not that a checkpoint was trained
using that input. Training/command linkage is a subsequent runner integration.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from completion_core.inference import CompletionResult, generate_completion, is_exact_completion
from completion_core.modeling import load_model_and_config
from completion_core.prep import encode_chars
from completion_core.training import SequenceDataset
from completion_core.validation import is_finite_number
from completion_core.vocabulary import Vocabulary
from phonebook.replay_protocol import (Membership, canonical_digest, digest_bytes, membership,
                                      read_json, validate_training_input)

SCHEMA = "phonebook-replay-evaluation-v1"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _count(value, maximum):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum


def _vocab(selected):
    stoi = dict(selected.vocabulary)
    return Vocabulary(len(stoi), stoi, {i: c for c, i in stoi.items()})


def _check_vocab(vocab, selected):
    expected = _vocab(selected)
    _require(not vocab.is_2char and vocab.vocab_size == expected.vocab_size
             and canonical_digest(vocab.stoi) == canonical_digest(expected.stoi)
             and vocab.itos == expected.itos and all(type(i) is int for i in vocab.itos)
             and (vocab.pad_id, vocab.equal_id, vocab.eot_id) == (expected.pad_id, expected.equal_id, expected.eot_id),
             "Vocabulary identity differs from the pinned common vocabulary")


def _summary(ids, rows):
    chosen = [rows[entry] for entry in ids]
    tokens = sum(r["teacher"]["token_total"] for r in chosen)
    token_correct = sum(r["teacher"]["token_correct"] for r in chosen)
    sequence_correct = sum(r["teacher"]["sequence_correct"] for r in chosen)
    correct = sum(r["generation"]["correct"] for r in chosen)
    loss_sum = sum(r["teacher"]["loss_sum"] for r in chosen)
    return {"status": "ok" if chosen else "not_applicable", "entry_ids": list(ids),
            "generation": {"correct": correct, "total": len(chosen),
                           "accuracy": correct / len(chosen) if chosen else None,
                           "stop_counts": {stop: sum(r["generation"]["stop_reason"] == stop for r in chosen)
                                           for stop in ("newline", "budget", "context")}},
            "teacher_forced": {"loss": loss_sum / tokens if tokens else None, "loss_sum": loss_sum,
                               "token_correct": token_correct, "token_total": tokens,
                               "token_accuracy": token_correct / tokens if tokens else None,
                               "sequence_correct": sequence_correct, "sequence_total": len(chosen),
                               "sequence_accuracy": sequence_correct / len(chosen) if chosen else None}}


def audit_partition_report(report, selected: Membership, training_input):
    """Recompute counts/rates from raw rows; teacher CE needs fresh inference to authenticate."""
    receipt = validate_training_input(selected, training_input)
    headers = {"schema": SCHEMA, "protocol_sha256": selected.protocol_sha256,
               "condition": selected.condition_id, "kind": selected.kind,
               "context_length": selected.context_length,
               "vocabulary_sha256": canonical_digest(dict(selected.vocabulary)),
               "generation_budget": "automatic_context_limit", "training_execution_verified": False,
               "probe_exposed": bool(set(selected.probe_ids) & set(selected.training_ids))}
    _require(isinstance(report, dict) and set(report) == set(headers) | {"training_input", "records", "partitions"},
             "Evaluation report fields differ from the schema")
    _require(canonical_digest({k: report[k] for k in headers}) == canonical_digest(headers),
             "Evaluation binding, condition or probe exposure differs from the protocol")
    saved = report["training_input"]
    _require(isinstance(saved, dict) and set(saved) == set(receipt), "Training receipt fields differ")
    _require(canonical_digest({k: saved[k] for k in saved if k != "sha256"}) ==
             canonical_digest({k: receipt[k] for k in receipt if k != "sha256"}), "Training receipt membership differs")
    text = selected.training_text()
    variants = {receipt["sha256"], digest_bytes(text.encode()), digest_bytes(text.replace("\n", "\r\n").encode())}
    _require(isinstance(saved["sha256"], str) and saved["sha256"] in variants, "Training receipt hash differs")
    records = selected.a + selected.b
    _require(isinstance(report["records"], list) and len(report["records"]) == len(records), "Record coverage differs")
    vocab = _vocab(selected)
    rows = {}
    for expected, row in zip(records, report["records"], strict=True):
        _require(isinstance(row, dict) and set(row) == {"id", "prompt", "expected", "teacher", "generation"},
                 "Record fields differ")
        _require((row["id"], row["prompt"], row["expected"]) == (expected.id, expected.prompt, expected.target),
                 "Record identity or expected target differs")
        teacher, generation = row["teacher"], row["generation"]
        _require(isinstance(teacher, dict) and set(teacher) == {"loss_sum", "token_total", "token_correct", "sequence_correct"},
                 "Teacher fields differ")
        total = len(expected.target) + 1
        _require(type(teacher["token_total"]) is int and teacher["token_total"] == total,
                 "Supervised-token denominator differs")
        _require(_count(teacher["token_correct"], total) and _count(teacher["sequence_correct"], 1)
                 and teacher["sequence_correct"] == int(teacher["token_correct"] == total), "Teacher counts disagree")
        _require(is_finite_number(teacher["loss_sum"]) and teacher["loss_sum"] >= 0, "Teacher loss must be finite and nonnegative")
        _require(isinstance(generation, dict) and set(generation) == {"prediction", "token_ids", "stop_reason", "correct"},
                 "Generation fields differ")
        ids, stop = generation["token_ids"], generation["stop_reason"]
        available = selected.context_length - len(expected.prompt) - 1
        _require(isinstance(ids, list) and 1 <= len(ids) <= available
                 and all(_count(i, vocab.vocab_size - 1) for i in ids), "Generated token IDs or length differ")
        _require(stop in ("newline", "context"), "Stop reason is incompatible with automatic context budgeting")
        if stop == "newline":
            _require(ids[-1] == vocab.eot_id and vocab.eot_id not in ids[:-1], "Newline stop disagrees with token IDs")
        else:
            _require(len(ids) == available and vocab.eot_id not in ids, "Context stop disagrees with token IDs")
        decoded = "".join(vocab.itos[i] for i in (ids[:-1] if stop == "newline" else ids))
        _require(generation["prediction"] == decoded, "Raw prediction disagrees with generated token IDs")
        exact = is_exact_completion(CompletionResult(decoded, tuple(ids), stop), expected.target, vocab)
        _require(type(generation["correct"]) is bool and generation["correct"] == exact,
                 "Strict generation score disagrees with raw tokens")
        _require(teacher["sequence_correct"] == int(exact), "Teacher and greedy exact sequence counts disagree")
        rows[expected.id] = row
    partitions = {name: _summary(ids, rows) for name, ids in selected.partitions().items()}
    _require(canonical_digest(partitions) == canonical_digest(report["partitions"]),
             "Partition membership, counts, rates or loss differ from the raw records")
    return {"status": "passed", "condition": selected.condition_id, "records": len(records),
            "probe_exposed": headers["probe_exposed"], "training_execution_verified": False}


def evaluate_partitions(model, vocab, selected: Membership, training_input, *, context_length=None):
    receipt = validate_training_input(selected, training_input)  # Before any model call.
    _check_vocab(vocab, selected)
    if context_length is None:
        context_length = selected.context_length
    _require(type(context_length) is int and context_length == selected.context_length, "Checkpoint context differs from the protocol")
    records = selected.a + selected.b
    modes = [(module, module.training) for module in model.modules()]
    rows = []
    def checked_forward(tokens):
        output = model(tokens)
        _require(isinstance(output, torch.Tensor) and output.is_floating_point()
                 and tuple(output.shape) == (*tokens.shape, vocab.vocab_size) and torch.isfinite(output).all(),
                 "Invalid or nonfinite model logits")
        return output
    try:
        model.eval()
        with torch.random.fork_rng(devices=[]), torch.no_grad(), tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "all.bin"
            encode_chars("".join(r.line for r in records), vocab.stoi).tofile(path)
            dataset = SequenceDataset(path, vocab)
            x, y = zip(*(dataset[i] for i in range(len(dataset))))
            x, y = torch.stack(x), torch.stack(y)
            logits = checked_forward(x)
            losses = F.cross_entropy(logits.reshape(-1, vocab.vocab_size), y.reshape(-1), reduction="none").reshape_as(y)
            mask = y != -100
            correct = (logits.argmax(dim=-1) == y) & mask
            for i, record in enumerate(records):
                generated = generate_completion(record.prompt, checked_forward, vocab, None, context_length, "cpu")
                rows.append({"id": record.id, "prompt": record.prompt, "expected": record.target,
                             "teacher": {"loss_sum": losses[i].double().sum().item(), "token_total": mask[i].sum().item(),
                                         "token_correct": correct[i].sum().item(),
                                         "sequence_correct": int(correct[i].sum() == mask[i].sum())},
                             "generation": {"prediction": generated.text, "token_ids": list(generated.token_ids),
                                            "stop_reason": generated.stop_reason,
                                            "correct": is_exact_completion(generated, record.target, vocab)}})
    finally:
        for module, mode in modes:
            module.training = mode
    by_id = {row["id"]: row for row in rows}
    report = {"schema": SCHEMA, "protocol_sha256": selected.protocol_sha256, "condition": selected.condition_id,
              "kind": selected.kind, "context_length": context_length,
              "vocabulary_sha256": canonical_digest(dict(selected.vocabulary)),
              "generation_budget": "automatic_context_limit", "training_execution_verified": False,
              "probe_exposed": bool(set(selected.probe_ids) & set(selected.training_ids)),
              "training_input": receipt, "records": rows,
              "partitions": {name: _summary(ids, by_id) for name, ids in selected.partitions().items()}}
    audit_partition_report(report, selected, training_input)  # Catch input changes during inference too.
    return report


def evaluate_checkpoint_partitions(checkpoint, meta, protocol_path, protocol_pin, condition, training_input, *, repo_root=ROOT):
    protocol = read_json(protocol_path)
    selected = membership(protocol, condition, protocol_pin, repo_root=repo_root)
    validate_training_input(selected, training_input)
    checkpoint, meta = Path(checkpoint).resolve(), Path(meta).resolve()
    before = {"checkpoint_sha256": digest_bytes(checkpoint.read_bytes()), "metadata_sha256": digest_bytes(meta.read_bytes())}
    vocab = Vocabulary.from_pickle(meta)
    _check_vocab(vocab, selected)
    with torch.random.fork_rng(devices=[]):
        model, config = load_model_and_config(checkpoint, vocab, "cpu")
        result = evaluate_partitions(model, vocab, selected, training_input, context_length=config.seq_len)
    membership(read_json(protocol_path), condition, protocol_pin, repo_root=repo_root)
    _require(before == {"checkpoint_sha256": digest_bytes(checkpoint.read_bytes()), "metadata_sha256": digest_bytes(meta.read_bytes())},
             "Checkpoint or metadata changed during evaluation")
    return {"schema": "phonebook-replay-checkpoint-v1", "checkpoint": str(checkpoint), "metadata": str(meta),
            **before, "model_config": asdict(config), "device": "cpu", "evaluation": result}


def _same_fresh(actual, saved):
    if isinstance(actual, float) and isinstance(saved, float):
        return math.isclose(actual, saved, rel_tol=1e-6, abs_tol=1e-9)
    if type(actual) is not type(saved):
        return False
    if isinstance(actual, dict):
        return actual.keys() == saved.keys() and all(_same_fresh(actual[k], saved[k]) for k in actual)
    if isinstance(actual, list):
        return len(actual) == len(saved) and all(_same_fresh(a, b) for a, b in zip(actual, saved, strict=True))
    return actual == saved


def audit_saved_evaluation(result, protocol_path, protocol_pin, condition, training_input, *,
                           require_local_artifacts=False, repo_root=ROOT):
    selected = membership(read_json(protocol_path), condition, protocol_pin, repo_root=repo_root)
    _require(isinstance(result, dict) and set(result) == {"schema", "checkpoint", "metadata", "checkpoint_sha256",
             "metadata_sha256", "model_config", "device", "evaluation"}, "Checkpoint report fields differ")
    _require(result["schema"] == "phonebook-replay-checkpoint-v1" and result["device"] == "cpu", "Unsupported checkpoint report")
    for key in ("checkpoint", "metadata"):
        _require(isinstance(result[key], str) and result[key], "Artifact path must be nonempty")
        digest = result[key + "_sha256"]
        _require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                 "Malformed artifact hash")
    config = result["model_config"]
    _require(isinstance(config, dict) and set(config) == {"seq_len", "d_model", "n_heads", "n_layers"}
             and all(type(v) is int and v > 0 for v in config.values()), "Malformed model configuration")
    _require(config["seq_len"] == selected.context_length and config["d_model"] % config["n_heads"] == 0,
             "Model configuration differs from the common context or head dimensions")
    audit_partition_report(result["evaluation"], selected, training_input)
    if require_local_artifacts:
        for key in ("checkpoint", "metadata"):
            _require(digest_bytes(Path(result[key]).read_bytes()) == result[key + "_sha256"], "Artifact hash differs")
        fresh = evaluate_checkpoint_partitions(result["checkpoint"], result["metadata"], protocol_path, protocol_pin,
                                               condition, training_input, repo_root=repo_root)
        saved_body = copy.deepcopy(result["evaluation"])
        # The readable check has already authenticated normalized text and its
        # allowed LF/CRLF byte variants; compare scientific values independently.
        saved_body["training_input"]["sha256"] = fresh["evaluation"]["training_input"]["sha256"]
        _require(fresh["model_config"] == config and _same_fresh(fresh["evaluation"], saved_body),
                 "Fresh inference differs from the saved evaluation")
    return {"status": "passed", "condition": condition, "records": len(selected.a + selected.b),
            "local_artifacts_checked": require_local_artifacts, "training_execution_verified": False}


def write_evaluation(result, out_root):
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    (out_root / "evaluation.json").write_text(payload, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--checkpoint", type=Path)
    action.add_argument("--audit-report", type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--require-local-artifacts", action="store_true")
    args = parser.parse_args()
    if args.audit_report:
        if args.meta is not None or args.out_root is not None:
            parser.error("--meta and --out-root do not apply to --audit-report")
        result = audit_saved_evaluation(read_json(args.audit_report), args.protocol, args.protocol_sha256,
                                         args.condition, args.train_input, require_local_artifacts=args.require_local_artifacts)
        print(json.dumps(result, allow_nan=False))
        return
    if args.meta is None or args.out_root is None:
        parser.error("--meta and --out-root are required when evaluating a checkpoint")
    if args.require_local_artifacts:
        parser.error("--require-local-artifacts applies to --audit-report")
    if args.out_root.exists():
        parser.error("Output root already exists; choose an unused directory")
    result = evaluate_checkpoint_partitions(args.checkpoint, args.meta, args.protocol, args.protocol_sha256,
                                             args.condition, args.train_input)
    write_evaluation(result, args.out_root)
    print(json.dumps({"status": "passed", "protocol_sha256": args.protocol_sha256, "condition": args.condition,
                      "training_execution_verified": False}, allow_nan=False))


if __name__ == "__main__":
    main()
