import copy
import csv
import pickle
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from completion_core.modeling import ModelConfig, make_checkpoint_payload
from phonebook import run_partial_replay as pilot
from phonebook.replay_protocol import build_protocol, canonical_digest, membership, read_json


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("pilot-fixture")
    a, b = root / "a.txt", root / "b.txt"
    a.write_text("a=1\nb=2\nc=3\n", encoding="utf-8")
    b.write_text("d=4\ne=5\n", encoding="utf-8")
    protocol = build_protocol(a, b, probe_size=1, memory_sizes=(1,))
    selected = membership(protocol, "full_replay", canonical_digest(protocol))
    vocab = pilot._vocab(selected)
    cfg = {"seq_len": 4, "d_model": 8, "n_heads": 2, "n_layers": 1}
    settings = dict(pilot.DEFAULT_SETTINGS, primary_updates=3, secondary_updates=1,
                    sequence_budget=30, evaluation_interval=2, diagnostic_updates=[1, 3])
    meta = root / "meta.pkl"
    with meta.open("wb") as stream:
        pickle.dump({"vocab_size": vocab.vocab_size, "stoi": vocab.stoi, "itos": vocab.itos}, stream)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = pilot.new_model(vocab, cfg)
        x, y = pilot.tensor_batch(selected.a, vocab, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0)
        for _ in range(200):
            pilot.step(model, opt, x, y, torch.ones(3), 1.0)
    checkpoint = root / "initial.pth"
    torch.save(make_checkpoint_payload(model, ModelConfig(**cfg)), checkpoint)
    return root, protocol, selected, vocab, cfg, settings, meta, checkpoint


@pytest.fixture(scope="module")
def completed(fixture):
    root, protocol, _, _, cfg, settings, meta, checkpoint = fixture
    output = root / "output"
    rng = torch.random.get_rng_state().clone()
    pin = pilot.run_pilot(protocol, canonical_digest(protocol), checkpoint, pilot.sha(checkpoint), meta, pilot.sha(meta),
                          output, settings=settings, model_config=cfg)
    assert torch.equal(rng, torch.random.get_rng_state())
    return output, pin


def test_self_contained_end_to_end_and_full_optimizer_replay(completed):
    output, pin = completed
    assert pilot.audit_pilot(output, pin)["status"] == "passed"
    report = pilot.audit_pilot(output, pin, replay_training=True)
    assert report["training_recipe_reproduced"]
    assert len(report["conditions"]) == 4


@pytest.mark.parametrize("field,value", [("lr", float("nan")), ("lr", 0), ("seed", True),
    ("sequence_budget", 1), ("primary_updates", True), ("diagnostic_updates", [1, 1])])
def test_invalid_settings_and_budget_leave_no_output(fixture, tmp_path, field, value):
    _, protocol, _, _, cfg, settings, meta, checkpoint = fixture
    settings = dict(settings, **{field: value})
    output = tmp_path / "unused"
    with pytest.raises(ValueError):
        pilot.run_pilot(protocol, canonical_digest(protocol), checkpoint, pilot.sha(checkpoint), meta, pilot.sha(meta),
                        output, settings=settings, model_config=cfg)
    assert not output.exists()


@pytest.mark.parametrize("bad", ["checkpoint", "meta", "protocol", "heads"])
def test_pins_and_semantic_model_configuration_fail_before_output(fixture, tmp_path, bad):
    _, protocol, _, _, cfg, settings, meta, checkpoint = fixture
    pins = {"checkpoint": pilot.sha(checkpoint), "meta": pilot.sha(meta), "protocol": canonical_digest(protocol)}
    cfg = dict(cfg)
    if bad == "heads":
        cfg["n_heads"] = 1  # State shapes alone cannot detect a head-count change.
    else:
        pins[bad] = "0" * 64
    output = tmp_path / "unused"
    with pytest.raises(ValueError):
        pilot.run_pilot(protocol, pins["protocol"], checkpoint, pins["checkpoint"], meta, pins["meta"],
                        output, settings=settings, model_config=cfg)
    assert not output.exists()


def test_existing_output_is_untouched(fixture, completed):
    _, protocol, _, _, cfg, settings, meta, checkpoint = fixture
    output, _ = completed
    before = pilot.sha(output / "complete.json")
    with pytest.raises(ValueError, match="unused"):
        pilot.run_pilot(protocol, canonical_digest(protocol), checkpoint, pilot.sha(checkpoint), meta, pilot.sha(meta),
                        output, settings=settings, model_config=cfg)
    assert pilot.sha(output / "complete.json") == before


@pytest.mark.parametrize("mutant", ["prediction", "subset", "initialization", "metric", "summary", "final_epoch",
                                     "binary", "checkpoint", "optimizer", "diagnostic", "source", "spec"])
def test_auditor_rejects_disposable_mutations(completed, tmp_path, mutant):
    import shutil
    output, pin = completed
    target = tmp_path / "copy"
    shutil.copytree(output, target)
    folder = target / "prepared" / "full_replay"
    if mutant == "prediction":
        path = folder / "evaluation-0000.json"
        obj = read_json(path)
        obj["evaluation"]["records"][0]["generation"]["prediction"] += "x"
        pilot.save_json(path, obj)
    elif mutant == "subset":
        path = target / "prepared" / "protocol.json"
        obj = read_json(path)
        obj["probe_ids"][0] = "0" * 64
        pilot.save_json(path, obj)
    elif mutant in ("initialization", "summary", "diagnostic"):
        path = folder / "run.json"
        obj = read_json(path)
        if mutant == "initialization":
            obj["initial_state_sha256"] = "0" * 64
        elif mutant == "summary":
            obj["work"]["training_sequences"] += 1
        else:
            obj["diagnostics"][0]["linear_term"] += 1
        pilot.save_json(path, obj)
    elif mutant in ("metric", "final_epoch"):
        path = folder / "updates.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if mutant == "final_epoch":
            rows.pop()
        else:
            rows[0]["task_loss_before_update"] = str(float(rows[0]["task_loss_before_update"]) + 1)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif mutant == "binary":
        path = folder / "train.bin"
        path.write_bytes(path.read_bytes()[:-2])
    elif mutant in ("checkpoint", "optimizer"):
        path = folder / "checkpoint-0006.pth"
        obj = torch.load(path, weights_only=True)
        if mutant == "checkpoint":
            obj["state_dict"]["lm_head.weight"][0, 0] += 1
        else:
            obj["optimizer_state_dict"]["state"][0]["step"] += 1
        torch.save(obj, path)
    elif mutant == "source":
        path = target / "source" / "completion_core" / "objectives.py"
        path.write_text(path.read_text() + "\n# changed\n")
    else:
        path = target / "specification.json"
        obj = read_json(path)
        obj["settings"]["lr"] *= 2
        pilot.save_json(path, obj)
    with pytest.raises(ValueError):
        pilot.audit_pilot(target, pin, replay_training=True)


def test_binary_mask_and_order_validation(fixture, tmp_path):
    _, _, selected, vocab, _, _, _, _ = fixture
    text, binary = tmp_path / "train.txt", tmp_path / "train.bin"
    text.write_text(selected.training_text(), newline="\n")
    binary.write_bytes(pilot.encode_chars(selected.training_text(), vocab.stoi).tobytes())
    x, y, w = pilot.training_batch(selected, text, vocab, binary)
    assert (y != -100).sum().item() == 10
    assert torch.allclose((w * (y != -100).sum(1)).sum(), torch.tensor(1.0))
    binary.write_bytes(binary.read_bytes()[2:])
    with pytest.raises(ValueError, match="binary"):
        pilot.training_batch(selected, text, vocab, binary)


@pytest.mark.parametrize("condition", ["plain_b", "partial_m1_seed1701", "full_replay"])
def test_float32_loss_every_gradient_and_adamw_update_against_separate_book_means(fixture, condition):
    _, protocol, _, vocab, cfg, settings, _, _ = fixture
    selected = membership(protocol, condition, canonical_digest(protocol))
    rows = {r.id: r for r in selected.a + selected.b}
    x, y = pilot.tensor_batch([rows[i] for i in selected.training_ids], vocab, selected.context_length)
    n_a = len(selected.replay_ids)
    counts = (y != -100).sum(1)
    weights = torch.ones(len(x)) / counts[n_a:].sum() * (0.5 if n_a else 1)
    if n_a:
        weights[:n_a] = 0.5 / counts[:n_a].sum()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        actual = pilot.new_model(vocab, cfg).train()
        reference = copy.deepcopy(actual)
        actual_loss = pilot.weighted_token_cross_entropy(actual(x), y, weights)
        b_logits = reference(x[n_a:])
        reference_loss = F.cross_entropy(b_logits.reshape(-1, vocab.vocab_size), y[n_a:].reshape(-1))
        if n_a:
            a_logits = reference(x[:n_a])
            reference_loss = 0.5 * reference_loss + 0.5 * F.cross_entropy(a_logits.reshape(-1, vocab.vocab_size), y[:n_a].reshape(-1))
        # Declared float32 tolerances; this does not assert long-trajectory identity.
        torch.testing.assert_close(actual_loss, reference_loss, atol=1e-6, rtol=1e-5)
        actual_loss.backward()
        reference_loss.backward()
        for a, b in zip(actual.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
        initial = copy.deepcopy(actual.state_dict())
        clipped = []
        for model in (actual, reference):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            clipped.append({name: p.grad.clone() for name, p in model.named_parameters()})
            pilot.optimizer_for(model, settings).step()
        # Tiny reduction differences in mathematically zero attention-key bias
        # gradients can be amplified by AdamW's epsilon. Do not loosen a claimed
        # trajectory-equivalence tolerance to conceal this observed divergence.
        # Verify each implementation against the independent first-step formula.
        for model, gradients in zip((actual, reference), clipped, strict=True):
            for name, parameter in model.named_parameters():
                gradient = gradients[name]
                expected = initial[name] * (1 - settings["lr"] * settings["weight_decay"])
                expected -= settings["lr"] * gradient / (gradient.abs() + 1e-8)
                torch.testing.assert_close(parameter, expected, atol=2e-7, rtol=1e-5)


def test_diagnostics_and_evaluation_preserve_every_training_state(fixture, tmp_path):
    _, _, selected, vocab, cfg, settings, _, _ = fixture
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        model = pilot.new_model(vocab, cfg)
        x, y = pilot.tensor_batch(selected.a + selected.b, vocab, cfg["seq_len"])
        optimizer = pilot.optimizer_for(model, settings)
        before = copy.deepcopy(model)
        pilot.step(model, optimizer, x, y, torch.ones(len(x)), 1.0)
        weights = copy.deepcopy(model.state_dict())
        grads = [p.grad.clone() for p in model.parameters()]
        opt = copy.deepcopy(optimizer.state_dict())
        rng = torch.random.get_rng_state().clone()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            diagnostic = pilot.local_diagnostic(before, model, vocab, selected)
        path = tmp_path / "train.txt"
        path.write_text(selected.training_text(), newline="\n")
        pilot.evaluate_partitions(model, vocab, selected, path)
        pilot.compare_nested(model.state_dict(), weights)
        pilot.compare_nested(optimizer.state_dict(), opt)
        assert all(torch.equal(a, p.grad) for a, p in zip(grads, model.parameters(), strict=True))
        assert torch.equal(rng, torch.random.get_rng_state())
        assert model.training
        assert diagnostic["analyst_only"]
        assert abs(diagnostic["actual_loss_change"] - diagnostic["linear_term"] - diagnostic["quadratic_term"]
                   - diagnostic["second_order_residual"]) < 1e-12


def test_report_rebuilds_every_endpoint_from_raw_counts(completed):
    from phonebook.report_partial_replay import derive
    output, pin = completed
    first = derive(output, pin)
    assert first == derive(output, pin)
    assert len(first["endpoints"]) == 12
    assert first["confirmation_ready"] is False
    assert first["independent_datasets"] == 1
    for row in first["endpoints"]:
        path = output / "prepared" / row["condition"] / f"evaluation-{row['updates']:04d}.json"
        raw = read_json(path)
        for name, partition in raw["evaluation"]["partitions"].items():
            assert row[name + "_total"] == partition["generation"]["total"]
            assert row[name + "_correct"] == (partition["generation"]["correct"] if partition["status"] == "ok" else None)
        assert row["training_sequences"] == row["updates"] * (2 + row["replay_records"])
    for row in first["sampled_observations"]:
        assert row["sampled_trapezoidal_A_probe_error_update_units"] >= 0


def test_report_rejects_a_changed_external_specification_pin(completed):
    from phonebook.report_partial_replay import derive
    output, _ = completed
    with pytest.raises(ValueError, match="pin"):
        derive(output, "0" * 64)


def test_artifact_seal_covers_checkpoint_bytes_and_detects_changes(completed, tmp_path):
    import shutil
    from phonebook.report_partial_replay import artifact_manifest, verify_artifacts
    output, _ = completed
    target = tmp_path / "relocated"
    shutil.copytree(output, target)
    manifest = artifact_manifest(target)
    pin = canonical_digest(manifest)
    assert verify_artifacts(target, manifest, pin)["status"] == "passed"
    assert any(p.endswith(".pth") for p in manifest["sha256"])
    path = target / "prepared" / "full_replay" / "checkpoint-0006.pth"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes"):
        verify_artifacts(target, manifest, pin)


def test_report_plot_uses_declared_panel_sizes_and_common_horizon(completed, tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from phonebook.report_partial_replay import derive, write_report
    output, pin = completed
    captured = []
    original = Figure.savefig
    def capture(figure, *args, **kwargs):
        captured.append(figure)
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture)
    write_report(derive(output, pin), tmp_path / "report")
    axes = captured[0].axes
    assert axes[0].get_title() == "Fixed A probe (1 facts)"
    assert axes[1].get_title() == "Book B (2 facts)"
    assert max(max(line.get_xdata()) for axis in axes for line in axis.lines) == 3
    assert axes[0].get_ylim() == (-0.2, 1.2)
    assert axes[1].get_ylim() == (-0.2, 2.2)
    assert captured[0]._suptitle.get_text() == "Development data / seed42 / one memory draw — sampled every 2 updates"
