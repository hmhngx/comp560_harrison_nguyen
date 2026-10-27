from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from completion_core.modeling import CompletionTransformer
from completion_core.prep import build_char_vocab
from completion_core.training import SequenceDataset
from completion_core.vocabulary import Vocabulary
from phonebook import verify_weighted_objective as verify


def toy():
    logits = torch.tensor([[[4., 0.]] * 3] * 2, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0, -100, -100], [1, 1, 1]])
    return logits, targets, torch.tensor([1., 3.], dtype=torch.float64)


def test_weighted_token_loss_and_gradient_match_a_hand_calculation():
    logits, targets, weights = toy()
    loss = verify.weighted_token_cross_entropy(logits, targets, weights)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-4)) + 36 / 10, abs=1e-12)
    loss.backward()
    expected = logits.detach().softmax(dim=-1)
    expected[0, 0, 0] -= 1
    expected[1, :, 1] -= 1
    expected *= (targets != -100)[..., None] * weights[:, None, None] / 10
    torch.testing.assert_close(logits.grad, expected, atol=1e-12, rtol=1e-12)


def test_weighted_loss_passes_finite_difference_gradient_check():
    logits, targets, weights = toy()
    assert torch.autograd.gradcheck(lambda x: verify.weighted_token_cross_entropy(x, targets, weights),
                                   (logits,), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_book_balancing_requires_token_total_correction():
    logits, targets, _ = toy()
    weights = torch.tensor([0.5 / 1, 0.5 / 3], dtype=torch.float64)
    balanced = verify.weighted_token_cross_entropy(logits, targets, weights)
    assert balanced.item() == pytest.approx(math.log1p(math.exp(-4)) + 2, abs=1e-12)
    naive = verify.weighted_token_cross_entropy(logits, targets, torch.tensor([0.5, 0.5]))
    assert not math.isclose(balanced.item(), naive.item(), abs_tol=0.5)


def test_weight_scaling_and_aligned_row_permutation_preserve_loss():
    logits, targets, weights = toy()
    original = verify.weighted_token_cross_entropy(logits, targets, weights)
    scaled = verify.weighted_token_cross_entropy(logits, targets, weights * 7)
    permuted = verify.weighted_token_cross_entropy(logits.flip(0), targets.flip(0), weights.flip(0))
    torch.testing.assert_close(original, scaled, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(original, permuted, atol=1e-12, rtol=1e-12)


def test_zero_weight_and_masked_targets_have_zero_logit_gradient():
    logits, targets, _ = toy()
    loss = verify.weighted_token_cross_entropy(logits, targets, torch.tensor([1, 0]))
    assert loss.item() == pytest.approx(math.log1p(math.exp(-4)), abs=1e-12)
    loss.backward()
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0


@pytest.mark.parametrize("weights", [
    torch.tensor([1., float("nan")]), torch.tensor([1., float("inf")]),
    torch.tensor([1., -float("inf")]), torch.tensor([1., -1.]), torch.tensor([0., 0.]),
    torch.tensor([True, False]), torch.tensor([1+0j, 1+0j]), torch.ones(1), torch.ones(2, 1),
    torch.tensor([1., 1e308], dtype=torch.float64),
])
def test_invalid_weights_are_rejected(weights):
    logits, targets, _ = toy()
    with pytest.raises(ValueError):
        verify.weighted_token_cross_entropy(logits, targets, weights)


@pytest.mark.parametrize("value", [-1e-300, float("inf"), float("nan"), 1e300])
def test_conversion_to_float32_does_not_hide_invalid_weights(value):
    logits, targets, _ = toy()
    with pytest.raises(ValueError):
        verify.weighted_token_cross_entropy(logits.float(), targets, torch.tensor([1., value], dtype=torch.float64))


@pytest.mark.parametrize("case", ["all_masked", "wrong_target_shape", "float_targets", "integer_logits", "device"])
def test_invalid_tensor_contract_and_undefined_denominator_fail(case):
    logits, targets, weights = toy()
    if case == "all_masked":
        targets.fill_(-100)
    elif case == "wrong_target_shape":
        targets = targets[:, :1]
    elif case == "float_targets":
        targets = targets.float()
    elif case == "integer_logits":
        logits = logits.long()
    else:
        weights = torch.ones(2, device="meta")
    with pytest.raises(ValueError):
        verify.weighted_token_cross_entropy(logits, targets, weights)


def test_all_masked_row_is_allowed_when_other_rows_have_positive_mass():
    logits, targets, weights = toy()
    targets[1].fill_(-100)
    loss = verify.weighted_token_cross_entropy(logits, targets, weights)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-4)), abs=1e-12)


def test_fixture_encoder_matches_the_actual_dataset_mask_and_padding(tmp_path):
    lines = ["a=1\n", "ab=123\n", "b=12\n"]
    stoi, itos = build_char_vocab("".join(lines), "_")
    vocab = Vocabulary(len(stoi), stoi, itos)
    binary = tmp_path / "fixture.bin"
    np.array([stoi[c] for c in "".join(lines)], dtype=np.uint16).tofile(binary)
    dataset = SequenceDataset(binary, vocab)
    expected_x, expected_y = zip(*(dataset[i] for i in range(len(dataset))))
    # Common context can be longer than this particular subset's longest line.
    x, y = verify.completion_batch(lines, vocab, dataset.max_len + 2)
    assert torch.equal(x, F.pad(torch.stack(expected_x), (0, 2), value=vocab.pad_id))
    assert torch.equal(y, F.pad(torch.stack(expected_y), (0, 2), value=-100))
    assert (y != -100).sum(dim=1).tolist() == [2, 4, 3]


@pytest.fixture
def transformer():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = CompletionTransformer(7, 8, 64, 2, 2).double()
    x = torch.tensor([[1, 2, 0, 0], [2, 3, 4, 0], [3, 4, 5, 6]])
    y = torch.tensor([[2, -100, -100, -100], [3, 4, -100, -100], [4, 5, 6, -100]])
    return model, x, y


@pytest.mark.parametrize("choice", [
    {"repetitions": 0}, {"repetitions": 1}, {"repetitions": 3}, {"repetitions": 9},
    {"fraction": 0.}, {"fraction": 0.25}, {"fraction": 0.5}, {"fraction": 1.},
])
def test_actual_transformer_checks_every_parameter_for_both_objectives(transformer, choice):
    model, x, y = transformer
    result = verify.compare_objective(model, x, y, 1, **choice)
    assert result["passed"]
    names = {name for name, _ in model.named_parameters()}
    assert len(names) == 29
    for objective in ("task", "with_l2sp", "l2sp_derivative"):
        rows = [p for p in result["parameters"] if p["objective"] == objective]
        assert {p["parameter"] for p in rows} == names
        assert sum(p["elements"] for p in rows) == sum(p.numel() for p in model.parameters())


def test_verification_does_not_change_input_weights_gradients_mode_or_rng(transformer):
    model, x, y = transformer
    model.eval()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    verify.compare_objective(model, x, y, 1, repetitions=3)
    assert not model.training
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(torch.equal(p, before[name]) for name, p in model.state_dict().items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())


@pytest.mark.parametrize("mutant", ["ignore_weights", "row_denominator", "book_means"])
def test_gate_rejects_wrong_objectives_even_with_a_shared_l2sp_penalty(monkeypatch, transformer, mutant):
    model, x, y = transformer
    def wrong(logits, targets, weights):
        losses = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1), reduction="none").reshape_as(targets)
        if mutant == "ignore_weights":
            return losses.sum() / (targets != -100).sum()
        if mutant == "row_denominator":
            return (losses * weights[:, None]).sum() / weights.sum()
        # Incorrect for literal duplication when A/B token totals differ.
        a = losses[:1].sum() / (targets[:1] != -100).sum()
        b = losses[1:].sum() / (targets[1:] != -100).sum()
        return (a + 3 * b) / 4
    monkeypatch.setattr(verify, "weighted_token_cross_entropy", wrong)
    result = verify.compare_objective(model, x, y, 1, repetitions=3)
    assert not result["passed"]
    assert not result["scalars"]["task"]["passed"]
    assert any(not p["passed"] for p in result["parameters"] if p["objective"] == "task")


def test_failed_nonfinite_comparison_can_be_saved_as_strict_json(monkeypatch, transformer, tmp_path):
    model, x, y = transformer
    original = verify.weighted_token_cross_entropy
    monkeypatch.setattr(verify, "weighted_token_cross_entropy", lambda *args: original(*args) * float("nan"))
    result = verify.compare_objective(model, x, y, 1, repetitions=3)
    assert not result["passed"]
    verify.write_verification({"cases": [{"state": "unit", "fixture": "unit", **result}]}, tmp_path / "failure")


def test_evidence_writer_preserves_existing_directory_and_rejects_nonfinite_metadata(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "sentinel"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        verify.write_verification({"cases": []}, out)
    assert sentinel.read_bytes() == b"preserve"
    invalid_out = tmp_path / "new"
    with pytest.raises(ValueError):
        verify.write_verification({"bad": float("nan"), "cases": []}, invalid_out)
    assert not invalid_out.exists()


def test_checker_rejects_one_bad_coordinate_and_shape_broadcasting():
    reference = torch.zeros(10000, dtype=torch.float64)
    actual = reference.clone()
    actual[-1] = 2 * verify.ATOL
    result = verify.comparison(actual, reference)
    assert not result["passed"] and result["max_tolerance_fraction"] == 2
    with pytest.raises(ValueError, match="shapes"):
        verify.comparison(torch.zeros(1), reference)


def test_nonzero_attention_dropout_is_outside_the_equivalence_contract(transformer):
    model, x, y = transformer
    model.transformer.layers[0].self_attn.dropout = 0.2
    with pytest.raises(ValueError, match="dropout"):
        verify.compare_objective(model, x, y, 1, repetitions=3)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_book_fractions_fail_before_computing_an_objective(transformer, value):
    model, x, y = transformer
    with pytest.raises(ValueError, match="fraction"):
        verify.compare_objective(model, x, y, 1, fraction=value)


def test_failed_cli_gate_records_failure_and_exits_nonzero(monkeypatch, tmp_path):
    out = tmp_path / "failed-gate"
    monkeypatch.setattr(sys, "argv", ["verify", "--out-root", str(out)])
    monkeypatch.setattr(verify, "run_verification", lambda: {"status": "failed", "cases": []})
    threads = torch.get_num_threads()
    try:
        with pytest.raises(SystemExit) as exc:
            verify.main()
    finally:
        torch.set_num_threads(threads)
    assert exc.value.code == 1
    assert json.loads((out / "verification.json").read_text())["status"] == "failed"


def test_existing_cli_output_is_rejected_before_verification(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["verify", "--out-root", str(tmp_path)])
    def forbidden():
        pytest.fail("Existing output must be rejected before verification")
    monkeypatch.setattr(verify, "run_verification", forbidden)
    with pytest.raises(SystemExit) as exc:
        verify.main()
    assert exc.value.code == 2


@pytest.mark.integration
def test_complete_matrix_uses_pinned_artifacts_preserves_rng_and_checks_all_coordinates():
    root = Path(__file__).resolve().parents[1]
    if not (root / "phonebook/out_phaseA/model.pth").exists():
        pytest.skip("Complete matrix requires the local historical Phase-A checkpoint")
    rng = torch.get_rng_state().clone()
    report = verify.run_verification(root)
    assert torch.equal(torch.get_rng_state(), rng)
    assert report["status"] == "passed" and len(report["cases"]) == 81
    assert report["model_config"]["d_model"] == 64
    assert report["model_config"]["n_heads"] == report["model_config"]["n_layers"] == 2
    assert report["environment"]["threads"] == torch.get_num_threads()
    assert report["atol"] == 1e-10 and report["rtol"] == 1e-9
    for name, digest in (report["source_sha256"] | report["input_sha256"]).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    expected = {(state, fixture, objective) for state in
                ("phase_a_float32_promoted_to_float64", "random_seed42", "random_seed123")
                for fixture in ("original_phonebooks", "unequal_target_lengths", "unequal_book_sizes")
                for objective in [*(f"duplicate_B_{k}" for k in (0, 1, 2, 3, 5, 9)),
                                  *(f"book_fraction_A_{f}" for f in (0.25, 0.5, 0.75))]}
    assert {(c["state"], c["fixture"], c["objective"]) for c in report["cases"]} == expected
    for case in report["cases"]:
        assert case["passed"] and len(case["parameters"]) == 29 * 3
        assert {p["objective"] for p in case["parameters"]} == {"task", "with_l2sp", "l2sp_derivative"}


def local_root():
    root = Path(__file__).resolve().parents[1]
    if not (root / "phonebook/out_phaseA/model.pth").exists():
        pytest.skip("Provenance check requires the local historical Phase-A checkpoint")
    return root


@pytest.mark.integration
def test_substituted_checkpoint_digest_is_rejected_before_forward_pass(monkeypatch):
    root = local_root()
    original = verify._sha
    monkeypatch.setattr(verify, "_sha", lambda path: "0" * 64 if path.name == "model.pth" else original(path))
    with pytest.raises(ValueError, match="initialization"):
        verify.run_verification(root)


@pytest.mark.integration
def test_initialization_spec_must_match_the_recorded_commit(monkeypatch):
    root = local_root()
    original = verify.subprocess.run
    def changed_spec(command, **kwargs):
        result = original(command, **kwargs)
        if command[:2] == ["git", "show"]:
            changed = json.loads(result.stdout)
            changed["seed"] += 1
            return SimpleNamespace(stdout=json.dumps(changed))
        return result
    monkeypatch.setattr(verify.subprocess, "run", changed_spec)
    with pytest.raises(ValueError, match="spec in HEAD"):
        verify.run_verification(root)


@pytest.mark.integration
def test_source_change_during_verification_prevents_a_success_report(monkeypatch):
    root = local_root()
    original = verify._sha
    changed = False
    def fake_comparison(*args, **kwargs):
        nonlocal changed
        changed = True
        return {"passed": True}
    def virtual_change(path):
        return "0" * 64 if changed and path.name == "verify_weighted_objective.py" else original(path)
    monkeypatch.setattr(verify, "compare_objective", fake_comparison)
    monkeypatch.setattr(verify, "_sha", virtual_change)
    with pytest.raises(ValueError, match="bytes changed"):
        verify.run_verification(root)
