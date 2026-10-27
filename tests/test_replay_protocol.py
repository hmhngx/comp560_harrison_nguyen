from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import random

import pytest

from phonebook import replay_protocol as rp


@pytest.fixture
def prepared(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("".join(f"a{i}={'123'[:i % 3 + 1]}\n" for i in range(6)), encoding="utf-8")
    b.write_text("b0=1\nb1=12\nb2=123\n", encoding="utf-8")
    protocol = rp.build_protocol(a, b, probe_size=2, memory_sizes=(1, 2), subset_seeds=(1701, 1702))
    return protocol, rp.canonical_digest(protocol), a, b


def test_selection_is_deterministic_nested_and_independent_of_global_rng(prepared):
    protocol, _, a, b = prepared
    before = random.getstate()
    same = rp.build_protocol(a, b, probe_size=2, memory_sizes=(1, 2), subset_seeds=(1701, 1702))
    assert same == protocol and random.getstate() == before
    probe = set(protocol["probe_ids"])
    assert len(probe) == 2 and len(protocol["replay_pool_ids"]) == 4
    for draw in protocol["draws"]:
        assert set(draw["permutation"]) == set(protocol["replay_pool_ids"])
        conditions = [c for c in protocol["conditions"] if c["id"].endswith(f"seed{draw['seed']}")]
        assert conditions[0]["replay_ids"] == draw["permutation"][:1]
        assert conditions[1]["replay_ids"] == draw["permutation"][:2]
        assert not probe & set(conditions[1]["training_ids"])


def test_every_condition_has_exact_unique_training_rows_and_reconciling_partitions(prepared):
    protocol, pin, _, _ = prepared
    for condition in protocol["conditions"]:
        selected = rp.membership(protocol, condition["id"], pin)
        partitions = selected.partitions()
        assert set(partitions["A_replayed"]).isdisjoint(partitions["A_unreplayed"])
        assert set(partitions["A_replayed"]) | set(partitions["A_unreplayed"]) == set(partitions["A_all"])
        assert len(set(selected.training_ids)) == len(selected.training_ids) == len(selected.replay_ids) + 3
        assert partitions["A_probe"] == tuple(protocol["probe_ids"])
        if selected.kind in ("partial", "b_only"):
            assert set(selected.probe_ids).isdisjoint(selected.training_ids)
        else:
            assert set(selected.probe_ids) <= set(selected.training_ids)
            assert partitions["A_unreplayed"] == ()


def test_validated_membership_is_deeply_immutable_and_detached_from_input(prepared):
    protocol, pin, _, _ = prepared
    selected = rp.membership(protocol, "partial_m1_seed1701", pin)
    saved = selected.training_text()
    with pytest.raises(FrozenInstanceError):
        selected.kind = "full_replay"
    with pytest.raises(FrozenInstanceError):
        selected.a[0].target = "changed"
    protocol["books"]["A"]["records"][0]["target"] = "changed"
    protocol["conditions"][1]["training_ids"].clear()
    assert selected.training_text() == saved


@pytest.mark.parametrize("mutation", ["probe", "condition", "target", "vocabulary_type", "extra", "context"])
def test_rehashing_an_edited_manifest_cannot_bypass_source_and_selection_reconstruction(prepared, mutation):
    protocol, pin, _, _ = prepared
    edited = copy.deepcopy(protocol)
    if mutation == "probe":
        edited["probe_ids"][0] = edited["replay_pool_ids"][0]
    elif mutation == "condition":
        c = edited["conditions"][1]
        c["replay_ids"] = edited["probe_ids"][:1]
        c["training_ids"] = c["replay_ids"] + [r["id"] for r in edited["books"]["B"]["records"]]
        lookup = {r["id"]: f"{r['prompt']}={r['target']}\n" for book in edited["books"].values() for r in book["records"]}
        c["training_sha256"] = rp.digest_bytes("".join(lookup[i] for i in c["training_ids"]).encode())
    elif mutation == "target":
        edited["books"]["A"]["records"][0]["target"] = "999"
    elif mutation == "vocabulary_type":
        key = next(k for k, v in edited["vocabulary"].items() if v == 0)
        edited["vocabulary"][key] = False
    elif mutation == "extra":
        edited["undeclared"] = True
    else:
        edited["context_length"] += 1
    with pytest.raises(ValueError, match="pin"):
        rp.validate_protocol(edited, pin)
    with pytest.raises(ValueError):
        rp.validate_protocol(edited, rp.canonical_digest(edited))


@pytest.mark.parametrize("field,value", [
    ("probe_size", 0), ("probe_size", True), ("probe_size", 6), ("probe_seed", True),
    ("probe_seed", 2**64), ("subset_seeds", []), ("subset_seeds", [1, 1]), ("subset_seeds", [True]),
    ("memory_sizes", []), ("memory_sizes", [0]), ("memory_sizes", [1, 1]),
    ("memory_sizes", [True]), ("memory_sizes", [5]),
])
def test_invalid_selection_parameters_fail(prepared, field, value):
    _, _, a, b = prepared
    settings = {"probe_size": 2, "memory_sizes": [1, 2], "subset_seeds": [1701]}
    settings[field] = value
    with pytest.raises(ValueError):
        rp.build_protocol(a, b, **settings)


@pytest.mark.parametrize("text", ["", "a=1\na=1\n", "a=1\na=2\n", "a=\n", "=1\n",
                                  "a=1=2\n", " a=1\n", "a=1 \n", "a_=1\n", "a=1\n\n", "a\t=1\n"])
def test_malformed_or_duplicate_records_are_rejected_before_writing(prepared, tmp_path, text):
    _, _, a, b = prepared
    a.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        rp.build_protocol(a, b, probe_size=1, memory_sizes=[1])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("target", ["1", "different"])
def test_cross_book_prompt_overlap_is_rejected_even_if_targets_disagree(prepared, target):
    _, _, a, b = prepared
    b.write_text(f"a0={target}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        rp.build_protocol(a, b, probe_size=2, memory_sizes=[1])


def test_record_ids_and_protocol_identity_are_stable_across_newlines(prepared):
    protocol, pin, a, b = prepared
    for path in (a, b):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    rebuilt = rp.build_protocol(a, b, probe_size=2, memory_sizes=(1, 2), subset_seeds=(1701, 1702))
    assert rebuilt == protocol and rp.canonical_digest(rebuilt) == pin
    rp.validate_protocol(protocol, pin)


@pytest.mark.parametrize("mutation", ["probe_insertion", "duplicate", "missing", "order", "target", "no_newline"])
def test_training_text_must_match_the_pinned_rows_and_order(prepared, tmp_path, mutation):
    protocol, pin, _, _ = prepared
    selected = rp.membership(protocol, "partial_m1_seed1701", pin)
    lines = selected.training_text().splitlines(keepends=True)
    if mutation == "probe_insertion":
        lines.append(next(r.line for r in selected.a if r.id in selected.probe_ids))
    elif mutation == "duplicate":
        lines.append(lines[0])
    elif mutation == "missing":
        lines.pop()
    elif mutation == "order":
        lines.reverse()
    elif mutation == "target":
        lines[0] = lines[0].split("=")[0] + "=changed\n"
    else:
        lines[-1] = lines[-1].rstrip("\n")
    path = tmp_path / "train.txt"
    path.write_bytes("".join(lines).encode())
    with pytest.raises(ValueError, match="training text"):
        rp.validate_training_input(selected, path)


def test_writer_materializes_all_conditions_and_never_overwrites(prepared, tmp_path):
    protocol, pin, _, _ = prepared
    out = tmp_path / "prepared"
    assert rp.write_protocol(protocol, out) == pin
    assert rp.read_json(out / "protocol.json") == protocol
    for c in protocol["conditions"]:
        selected = rp.membership(protocol, c["id"], pin)
        receipt = rp.validate_training_input(selected, out / c["id"] / "train.txt")
        assert receipt["entry_ids"] == c["training_ids"]
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        rp.write_protocol(protocol, out)
    assert before == {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_relative_sources_can_be_validated_in_a_relocated_directory(prepared, tmp_path):
    _, _, a, b = prepared
    protocol = rp.build_protocol("a.txt", "b.txt", probe_size=2, memory_sizes=[1], repo_root=tmp_path)
    other = tmp_path / "relocated"
    other.mkdir()
    (other / "a.txt").write_bytes(a.read_bytes())
    (other / "b.txt").write_bytes(b.read_bytes())
    rp.validate_protocol(protocol, rp.canonical_digest(protocol), repo_root=other)


@pytest.mark.parametrize("text", ['{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'])
def test_manifest_json_rejects_nonfinite_numbers_and_duplicate_keys(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        rp.read_json(path)


def test_unsupported_unequal_contexts_fail_before_preparing_conditions(prepared):
    _, _, a, b = prepared
    a.write_text(a.read_text() + "longername=123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="equal longest"):
        rp.build_protocol(a, b, probe_size=2, memory_sizes=[1])


def test_protocol_cli_writes_a_pinned_prepared_bundle(prepared, tmp_path, run_script):
    _, _, a, b = prepared
    out = tmp_path / "cli"
    result = run_script("phonebook/replay_protocol.py", ["--phonebook-a", str(a), "--phonebook-b", str(b),
                        "--probe-size", "2", "--memory-sizes", "1", "2", "--out-root", str(out)])
    assert result.returncode == 0, result.stderr
    protocol = rp.read_json(out / "protocol.json")
    assert result.stdout.strip() == rp.canonical_digest(protocol)
    assert len(protocol["conditions"]) == 5
