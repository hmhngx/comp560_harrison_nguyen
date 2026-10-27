"""Initialization declarations must remain tied to the committed protocol spec."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from phonebook import audit_experiment as auditor
from phonebook.run_experiment import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "phonebook/experiments/week_2026-10-07"


def copy_readable_evidence(tmp_path: Path, name="e1_prefix_check") -> Path:
    destination = tmp_path / name
    shutil.copytree(EVIDENCE / name, destination,
                    ignore=shutil.ignore_patterns("*.pth", "*.pkl", "*.bin", "*.png"))
    return destination


@pytest.fixture
def readable_copy(tmp_path):
    return copy_readable_evidence(tmp_path)


def audit(root, *, check_git=True):
    return auditor.audit(root, repo_root=ROOT, check_git=check_git)


def change_protocol(root, mutate):
    path = root / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    scenario = next(iter(protocol["initializations"]))
    mutate(protocol, scenario)
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def test_untouched_pinned_evidence_passes_without_ignored_artifacts(readable_copy, monkeypatch):
    original_read = Path.read_bytes

    def forbid_binary_reads(path):
        assert path.suffix not in (".pth", ".pkl", ".bin"), "readable audit opened a binary artifact"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_binary_reads)
    result = audit(readable_copy)
    assert result["status"] == "passed" and result["git_checked"] is True
    assert result["local_artifacts_checked"] is False


@pytest.mark.parametrize("field", ["checkpoint_sha256", "meta_sha256"])
def test_recorded_initialization_hash_must_match_committed_pin(readable_copy, field):
    change_protocol(readable_copy, lambda p, s: p["initializations"][s].update({field: "0" * 64}))
    with pytest.raises(ValueError, match="(?i)initialization.*pin"):
        audit(readable_copy)


@pytest.mark.parametrize("field", ["checkpoint", "meta"])
def test_declared_initialization_path_cannot_be_substituted(readable_copy, field):
    protocol = json.loads((readable_copy / "protocol.json").read_text(encoding="utf-8"))
    scenario = next(iter(protocol["initializations"]))
    replacement = protocol["initializations"][scenario][field] + ".other"
    change_protocol(readable_copy, lambda p, s: p["initializations"][s].update({field: replacement}))
    if field == "checkpoint":
        # Co-edit the command too: command-to-protocol agreement alone must not vouch for the wrong anchor.
        path = readable_copy / "manifest.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            command = record["command"]
            command[command.index("--adapt-from") + 1] = str(ROOT / replacement)
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)initialization.*path"):
        audit(readable_copy)


@pytest.mark.parametrize("mutation", ["missing_scenario", "extra_scenario", "extra_field"])
def test_initialization_coverage_and_record_fields_are_exact(readable_copy, mutation):
    def mutate(protocol, scenario):
        initializations = protocol["initializations"]
        if mutation == "missing_scenario":
            initializations.pop(scenario)
        elif mutation == "extra_scenario":
            initializations["undeclared"] = copy.deepcopy(initializations[scenario])
        else:
            initializations[scenario]["unrecognized"] = "value"
    change_protocol(readable_copy, mutate)
    with pytest.raises(ValueError, match="(?i)initialization"):
        audit(readable_copy)


@pytest.mark.parametrize("field", ["checkpoint", "meta", "checkpoint_sha256", "meta_sha256", "context_length"])
def test_missing_initialization_fields_fail_clearly(readable_copy, field):
    change_protocol(readable_copy, lambda p, s: p["initializations"][s].pop(field))
    with pytest.raises(ValueError, match="(?i)initialization"):
        audit(readable_copy)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "meta_sha256"])
@pytest.mark.parametrize("value", [None, 42, "", "a" * 63, "A" * 64])
def test_malformed_initialization_hashes_are_rejected(readable_copy, field, value):
    change_protocol(readable_copy, lambda p, s: p["initializations"][s].update({field: value}))
    with pytest.raises(ValueError, match="(?i)initialization"):
        audit(readable_copy)


@pytest.mark.parametrize("value", [0, -1, True, "29", None])
def test_context_length_is_a_positive_integer(readable_copy, value):
    change_protocol(readable_copy, lambda p, s: p["initializations"][s].update({"context_length": value}))
    with pytest.raises(ValueError, match="(?i)initialization.*context"):
        audit(readable_copy)


@pytest.mark.parametrize("value", [None, []])
@pytest.mark.parametrize("location", ["mapping", "record"])
def test_initializations_and_records_must_be_objects(readable_copy, value, location):
    def mutate(protocol, scenario):
        if location == "mapping":
            protocol["initializations"] = value
        else:
            protocol["initializations"][scenario] = value
    change_protocol(readable_copy, mutate)
    with pytest.raises(ValueError, match="(?i)initialization"):
        audit(readable_copy)


def test_fresh_training_has_no_initialization_records(tmp_path):
    root = copy_readable_evidence(tmp_path, "e2_phase_a")
    assert audit(root)["status"] == "passed"
    path = root / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["initializations"]["invented"] = {"checkpoint": "invented.pth"}
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)initialization.*scenario"):
        audit(root)


def test_unpinned_specs_remain_supported_but_cannot_verify_an_unpinned_digest(readable_copy):
    def remove_pins(protocol, scenario):
        for declared in protocol["spec"]["scenarios"]:
            declared["init"].pop("checkpoint_sha256")
            declared["init"].pop("meta_sha256")
        protocol["spec_sha256"] = canonical_sha256(protocol["spec"])
        protocol["initializations"][scenario]["checkpoint_sha256"] = "0" * 64
    change_protocol(readable_copy, remove_pins)
    # Synthetic unpinned-spec compatibility only. Real committed evidence is tested with Git checking above.
    assert audit(readable_copy, check_git=False)["status"] == "passed"


def test_rehashing_an_edited_spec_does_not_bypass_the_committed_spec(readable_copy):
    def mutate(protocol, scenario):
        protocol["spec"]["scenarios"][0]["init"]["checkpoint_sha256"] = "0" * 64
        protocol["spec_sha256"] = canonical_sha256(protocol["spec"])
    change_protocol(readable_copy, mutate)
    with pytest.raises(ValueError, match="committed"):
        audit(readable_copy)


def test_initialization_context_must_agree_with_adapted_runs(readable_copy):
    change_protocol(readable_copy, lambda p, s: p["initializations"][s].update({"context_length": 30}))
    with pytest.raises(ValueError, match="(?i)initialization.*context"):
        audit(readable_copy)


def test_coedited_context_and_run_lengths_are_checked_against_training_text(readable_copy):
    def mutate(protocol, scenario):
        protocol["initializations"][scenario]["context_length"] = 30
        for run in protocol["runs"]:
            run["seq_len"] = 30
    change_protocol(readable_copy, mutate)
    manifest = readable_copy / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["run"]["seq_len"] = 30
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)initialization.*context"):
        audit(readable_copy)


@pytest.mark.parametrize("mutation", ["missing", "null", "empty"])
def test_missing_spec_path_cannot_count_as_a_git_verified_audit(readable_copy, mutation):
    def mutate(protocol, scenario):
        if mutation == "missing":
            protocol.pop("spec_path")
        else:
            protocol["spec_path"] = None if mutation == "null" else ""
    change_protocol(readable_copy, mutate)
    with pytest.raises(ValueError, match="spec path"):
        audit(readable_copy)


def test_explicit_git_opt_out_does_not_claim_git_verification(readable_copy):
    change_protocol(readable_copy, lambda p, s: p.pop("spec_path"))
    result = audit(readable_copy, check_git=False)
    assert result["status"] == "passed" and result["git_checked"] is False


@pytest.mark.integration
def test_readable_audit_still_works_in_a_clean_relocated_clone(tmp_path):
    clone = tmp_path / "relocated-repo"
    subprocess.run(["git", "clone", "--shared", str(ROOT), str(clone)],
                   check=True, capture_output=True, text=True)
    assert not list(clone.rglob("*.pth"))
    result = auditor.audit(clone / "phonebook/experiments/week_2026-10-07/e1_prefix_check",
                           repo_root=clone)
    assert result["status"] == "passed" and result["git_checked"] is True
    assert result["local_artifacts_checked"] is False
