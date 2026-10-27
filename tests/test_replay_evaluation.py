from __future__ import annotations

import copy
import math
import pickle
import sys

import pytest
import torch
from torch import nn

from completion_core.modeling import CompletionTransformer, ModelConfig, make_checkpoint_payload
from completion_core.vocabulary import Vocabulary
from phonebook import evaluate_replay as er
from phonebook import replay_protocol as rp


class AnswerModel(nn.Module):
    """Deterministic per-position predictions; useful for independently known counts."""
    def __init__(self, vocab, records):
        super().__init__()
        self.vocab = vocab
        self.answers = {r.prompt: r.target if r.prompt in {"a0", "a2", "a5", "b1"} else "9" for r in records}
        self.offset = nn.Parameter(torch.zeros(vocab.vocab_size))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        logits = torch.full((*x.shape, self.vocab.vocab_size), -4.)
        for row, tokens in enumerate(x):
            eq = (tokens == self.vocab.equal_id).nonzero()[0].item()
            prompt = "".join(self.vocab.itos[t.item()] for t in tokens[:eq])
            answer = self.answers[prompt] + "\n"
            for position in range(x.shape[1]):
                index = position - eq
                char = answer[index] if 0 <= index < len(answer) else "\n"
                logits[row, position, self.vocab.stoi[char]] = 4.
        return self.dropout(logits + self.offset)


@pytest.fixture
def setup(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("".join(f"a{i}={'123'[:i % 3 + 1]}\n" for i in range(6)), encoding="utf-8")
    b.write_text("b0=1\nb1=12\nb2=123\n", encoding="utf-8")
    protocol = rp.build_protocol(a, b, probe_size=2, memory_sizes=[1, 2])
    pin = rp.canonical_digest(protocol)
    prepared = tmp_path / "prepared"
    rp.write_protocol(protocol, prepared)
    selected = rp.membership(protocol, "partial_m1_seed1701", pin)
    stoi = dict(selected.vocabulary)
    vocab = Vocabulary(len(stoi), stoi, {i: c for c, i in stoi.items()})
    model = AnswerModel(vocab, selected.a + selected.b)
    training = prepared / selected.condition_id / "train.txt"
    return protocol, pin, prepared, selected, vocab, model, training


def test_partition_metrics_match_hand_counts_and_token_weighted_ce(setup):
    _, _, _, selected, vocab, model, training = setup
    report = er.evaluate_partitions(model, vocab, selected, training)
    a, b = (report["partitions"][name] for name in ("A_all", "B"))
    assert (a["generation"]["correct"], a["generation"]["total"]) == (3, 6)
    assert (b["generation"]["correct"], b["generation"]["total"]) == (1, 3)
    assert (a["teacher_forced"]["token_correct"], a["teacher_forced"]["token_total"]) == (13, 18)
    assert (b["teacher_forced"]["token_correct"], b["teacher_forced"]["token_total"]) == (5, 9)
    correct_ce = math.log1p((vocab.vocab_size - 1) * math.exp(-8))
    assert a["teacher_forced"]["loss"] == pytest.approx(correct_ce + 40/18, abs=2e-6)
    assert b["teacher_forced"]["loss"] == pytest.approx(correct_ce + 32/9, abs=2e-6)
    replayed, unreplayed = (report["partitions"][name] for name in ("A_replayed", "A_unreplayed"))
    for key in ("correct", "total"):
        assert replayed["generation"][key] + unreplayed["generation"][key] == a["generation"][key]
    for key in ("token_correct", "token_total", "sequence_correct", "sequence_total", "loss_sum"):
        assert replayed["teacher_forced"][key] + unreplayed["teacher_forced"][key] == pytest.approx(a["teacher_forced"][key])
    assert not report["probe_exposed"] and report["training_execution_verified"] is False
    assert er.audit_partition_report(report, selected, training)["status"] == "passed"


@pytest.mark.parametrize("condition", ["plain_b", "partial_m1_seed1701", "partial_m2_seed1701", "full_replay", "fresh_joint"])
def test_fixed_probe_and_empty_partitions_are_explicit_for_every_control(setup, condition):
    protocol, pin, prepared, _, vocab, model, _ = setup
    selected = rp.membership(protocol, condition, pin)
    report = er.evaluate_partitions(model, vocab, selected, prepared / condition / "train.txt")
    assert report["partitions"]["A_probe"]["entry_ids"] == protocol["probe_ids"]
    assert report["probe_exposed"] == (condition in ("full_replay", "fresh_joint"))
    empty = "A_replayed" if condition == "plain_b" else "A_unreplayed" if condition in ("full_replay", "fresh_joint") else None
    if empty:
        partition = report["partitions"][empty]
        assert partition["status"] == "not_applicable"
        assert partition["generation"]["total"] == 0 and partition["generation"]["accuracy"] is None
        assert partition["teacher_forced"]["token_total"] == 0 and partition["teacher_forced"]["loss"] is None


def test_evaluation_preserves_weights_gradients_mixed_module_modes_and_rng(setup):
    _, _, _, selected, vocab, model, training = setup
    model.train()
    model.dropout.eval()
    model.offset.grad = torch.ones_like(model.offset)
    modes = [module.training for module in model.modules()]
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    er.evaluate_partitions(model, vocab, selected, training)
    assert modes == [module.training for module in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(model.offset.grad, torch.ones_like(model.offset))
    assert all(torch.equal(tensor, state[name]) for name, tensor in model.state_dict().items())


@pytest.mark.parametrize("case", ["training", "vocabulary", "context"])
def test_bad_preflight_inputs_fail_before_any_model_call(setup, case, monkeypatch):
    _, _, _, selected, vocab, model, training = setup
    def forbidden(*args):
        pytest.fail("Invalid inputs must be rejected before inference")
    monkeypatch.setattr(model, "forward", forbidden)
    context = selected.context_length
    if case == "training":
        training.write_text(selected.training_text() + selected.a[0].line, encoding="utf-8")
    elif case == "vocabulary":
        first, second = list(vocab.stoi)[:2]
        vocab.stoi[first], vocab.stoi[second] = vocab.stoi[second], vocab.stoi[first]
    else:
        context += 1
    with pytest.raises(ValueError):
        er.evaluate_partitions(model, vocab, selected, training, context_length=context)


@pytest.mark.parametrize("case", ["teacher", "generation", "input_change"])
def test_failed_inference_restores_modes_and_cannot_return_a_success_report(setup, monkeypatch, case):
    _, _, _, selected, vocab, model, training = setup
    original = model.forward
    calls = 0
    def broken(tokens):
        nonlocal calls
        calls += 1
        result = original(tokens)
        if case == "teacher" or (case == "generation" and calls > 1):
            result.fill_(float("nan"))
        if case == "input_change":
            training.write_text("tampered=1\n", encoding="utf-8")
        return result
    monkeypatch.setattr(model, "forward", broken)
    model.train()
    with pytest.raises(ValueError):
        er.evaluate_partitions(model, vocab, selected, training)
    assert model.training


@pytest.mark.parametrize("mutation", ["score", "prediction", "ids", "stop", "denominator", "teacher_bool",
                                       "teacher_sequence", "loss_nan", "duplicate", "probe", "partition_count",
                                       "exposure", "pin", "receipt", "extra"])
def test_readable_audit_rejects_mutated_records_and_partition_claims(setup, mutation):
    _, _, _, selected, vocab, model, training = setup
    report = er.evaluate_partitions(model, vocab, selected, training)
    row = report["records"][0]
    if mutation == "score":
        row["generation"]["correct"] = not row["generation"]["correct"]
    elif mutation == "prediction":
        row["generation"]["prediction"] += "extra"
    elif mutation == "ids":
        row["generation"]["token_ids"][0] = vocab.pad_id
    elif mutation == "stop":
        row["generation"]["stop_reason"] = "budget"
    elif mutation == "denominator":
        row["teacher"]["token_total"] += 1
    elif mutation == "teacher_bool":
        row["teacher"]["token_correct"] = True
    elif mutation == "teacher_sequence":
        row["teacher"]["sequence_correct"] = 1 - row["teacher"]["sequence_correct"]
    elif mutation == "loss_nan":
        row["teacher"]["loss_sum"] = float("nan")
    elif mutation == "duplicate":
        report["records"][1] = copy.deepcopy(row)
    elif mutation == "probe":
        report["partitions"]["A_probe"]["entry_ids"] = list(selected.replay_ids)
    elif mutation == "partition_count":
        report["partitions"]["A_unreplayed"]["generation"]["correct"] += 1
    elif mutation == "exposure":
        report["probe_exposed"] = True
    elif mutation == "pin":
        report["protocol_sha256"] = "0" * 64
    elif mutation == "receipt":
        report["training_input"]["entry_ids"].append(selected.probe_ids[0])
    else:
        report["undeclared"] = True
    with pytest.raises(ValueError):
        er.audit_partition_report(report, selected, training)


def checkpoint_files(setup, tmp_path):
    _, _, _, selected, vocab, _, _ = setup
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = CompletionTransformer(vocab.vocab_size, selected.context_length, 8, 2, 1)
    config = ModelConfig(selected.context_length, 8, 2, 1)
    checkpoint, meta = tmp_path / "model.pth", tmp_path / "meta.pkl"
    torch.save(make_checkpoint_payload(model, config), checkpoint)
    meta.write_bytes(pickle.dumps({"vocab_size": vocab.vocab_size, "stoi": vocab.stoi, "itos": vocab.itos}))
    return checkpoint, meta


@pytest.mark.integration
def test_checkpoint_evaluation_and_fresh_audit_reconcile_without_training(setup, tmp_path):
    _, pin, prepared, selected, _, _, training = setup
    checkpoint, meta = checkpoint_files(setup, tmp_path)
    rng = torch.get_rng_state().clone()
    result = er.evaluate_checkpoint_partitions(checkpoint, meta, prepared / "protocol.json", pin, selected.condition_id, training)
    assert er.audit_saved_evaluation(result, prepared / "protocol.json", pin, selected.condition_id, training,
                                     require_local_artifacts=True)["status"] == "passed"
    assert torch.equal(torch.get_rng_state(), rng)
    assert result["evaluation"]["training_execution_verified"] is False
    path = tmp_path / "evaluation"
    er.write_evaluation(result, path)
    saved = (path / "evaluation.json").read_bytes()
    with pytest.raises(FileExistsError):
        er.write_evaluation(result, path)
    assert (path / "evaluation.json").read_bytes() == saved


@pytest.mark.integration
def test_coedited_teacher_losses_pass_consistency_only_and_fail_fresh_inference(setup, tmp_path):
    _, pin, prepared, selected, _, _, training = setup
    checkpoint, meta = checkpoint_files(setup, tmp_path)
    report = er.evaluate_checkpoint_partitions(checkpoint, meta, prepared / "protocol.json", pin, selected.condition_id, training)
    body = report["evaluation"]
    body["records"][0]["teacher"]["loss_sum"] += 1
    by_id = {r["id"]: r for r in body["records"]}
    body["partitions"] = {name: er._summary(ids, by_id) for name, ids in selected.partitions().items()}
    assert er.audit_saved_evaluation(report, prepared / "protocol.json", pin, selected.condition_id, training)["status"] == "passed"
    with pytest.raises(ValueError, match="Fresh"):
        er.audit_saved_evaluation(report, prepared / "protocol.json", pin, selected.condition_id, training,
                                  require_local_artifacts=True)


def test_nonfinite_evaluation_cannot_create_an_output(tmp_path):
    out = tmp_path / "new"
    with pytest.raises(ValueError):
        er.write_evaluation({"loss": float("nan")}, out)
    assert not out.exists()


def test_existing_cli_output_is_rejected_before_loading_any_inputs(setup, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Existing output must fail before checkpoint/input loading")
    monkeypatch.setattr(er, "evaluate_checkpoint_partitions", forbidden)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--protocol", "missing.json", "--protocol-sha256", "0"*64,
                        "--condition", "plain_b", "--train-input", "missing.txt", "--checkpoint", "missing.pth",
                        "--meta", "missing.pkl", "--out-root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        er.main()
    assert exc.value.code == 2


@pytest.mark.integration
def test_evaluation_and_full_audit_clis_execute_the_pinned_workflow(setup, tmp_path, run_script):
    _, pin, prepared, selected, _, _, training = setup
    checkpoint, meta = checkpoint_files(setup, tmp_path)
    common = ["--protocol", str(prepared / "protocol.json"), "--protocol-sha256", pin,
              "--condition", selected.condition_id, "--train-input", str(training)]
    out = tmp_path / "cli-evaluation"
    result = run_script("phonebook/evaluate_replay.py", [*common, "--checkpoint", str(checkpoint),
                        "--meta", str(meta), "--out-root", str(out)])
    assert result.returncode == 0, result.stderr
    assert rp.read_json(out / "evaluation.json")["evaluation"]["training_execution_verified"] is False
    audited = run_script("phonebook/evaluate_replay.py", [*common, "--audit-report", str(out / "evaluation.json"),
                         "--require-local-artifacts"])
    assert audited.returncode == 0, audited.stderr
    assert '"local_artifacts_checked": true' in audited.stdout
