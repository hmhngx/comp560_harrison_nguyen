"""Reconcile final tests, source preservation, sealed artifacts, and rebuilt reports."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from phonebook import run_partial_replay as pilot
from phonebook.report_partial_replay import derive, write_report, verify_artifacts

PIN = "d54a941e5cd794e590eadfc197395e55d4954988fdf5b030d8787f2af911d2f3"
ARTIFACT_PIN = "7552d5fd34ca9e52497b52f310d19c683720b1a8fb7a78a8f6ddc0f424256720"


def main():
    review = Path(__file__).parent
    root = ROOT / "phonebook/experiments/partial_replay_development_v1"
    results = review / "partial-replay-results"
    seal = verify_artifacts(root, pilot.read_json(results / "artifact-manifest.json"), ARTIFACT_PIN)
    full_audit_text = (review / "partial-pilot-replay-audit.txt").read_text(encoding="utf-8-sig")
    audit = json.loads(full_audit_text[full_audit_text.index("{"):])
    assert audit["status"] == "passed" and audit["specification_sha256"] == PIN and audit["training_recipe_reproduced"] is True
    assert sum(r["updates"] for r in audit["conditions"]) == 2645
    assert len(audit["conditions"]) == 6 and all(r["training_replayed"] is True for r in audit["conditions"])
    tests = ET.parse(review / "partial-tests-final.xml").getroot()
    suites = list(tests) if tests.tag == "testsuites" else [tests]
    assert sum(int(s.attrib["tests"]) for s in suites) == 613
    assert all(int(s.attrib[field]) == 0 for s in suites for field in ("failures", "errors", "skipped"))
    old_audits = pilot.read_json(review / "partial-historical-audits.json")
    assert len(old_audits) == 7 and sum(r["local"]["runs"] for r in old_audits) == 159
    assert sum(r["local"]["completions"] for r in old_audits) == 3180
    assert all(r[mode]["status"] == "passed" for r in old_audits for mode in ("readable", "local"))
    suffixes = {".pth", ".pkl", ".bin", ".json", ".jsonl", ".csv", ".png", ".log", ".txt", ".xml"}
    with (review / "file-inventory.csv").open(newline="", encoding="utf-8-sig") as stream:
        original = [r for r in csv.DictReader(stream) if Path(r["path"]).suffix in suffixes and ".pytest_cache" not in r["path"]]
    assert len(original) == 1337
    for row in original:
        assert pilot.sha(ROOT / row["path"]) == row["sha256"], row["path"]
    prior = pilot.read_json(review / "membership-preservation.json")["preexisting_python_sha256"]
    changed = {name: {"before": pin, "after": pilot.sha(ROOT / name)} for name, pin in prior.items() if pilot.sha(ROOT / name) != pin}
    assert set(changed) == {"phonebook/verify_weighted_objective.py"}
    assert all(pilot.sha(ROOT / name) == prior[name] for name in ("phonebook/run_replay_pilot.py", "tests/test_replay_pilot.py"))
    report = derive(root, PIN)
    assert report["controls_pass_at_primary"] is True and report["directional_hypothesis_observed_in_development"] is False
    assert sum(r["training_sequences"] for r in report["endpoints"] if r["endpoint"] == "sequence_budget") == 35995
    saved = pilot.read_json(results / "summary.json")
    saved.pop("generated_at_utc")
    assert saved == report
    rebuilt = []
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "rebuilt"
        write_report(report, output)
        for name in ("endpoints.csv", "sampled-observations.csv", "curves.csv", "recall.png"):
            assert (results / name).read_bytes() == (output / name).read_bytes(), name
            rebuilt.append(name)
    docs = [review / "partial-replay-execution.md", ROOT / "phonebook/docs/professor-discussion-packet.md",
            ROOT / "phonebook/docs/replay-membership-protocol.md", ROOT / "phonebook/docs/week-2026-10-12-to-20.md",
            ROOT / "phonebook/docs/weighted-objective-analysis.md"]
    links = 0
    for path in docs:
        source = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", source):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#", 1)[0]).exists(), (path, target)
            links += 1
    new_sources = ["completion_core/objectives.py", "phonebook/run_partial_replay.py", "phonebook/report_partial_replay.py",
                   "tests/test_partial_replay.py", "phonebook/verify_weighted_objective.py",
                   "phonebook/docs/review-2026-10-20/verify_partial_optimizer.py",
                   "phonebook/docs/review-2026-10-20/verify_partial_delivery.py"]
    validation = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_full_suite": {"tests": 613, "failures": 0, "errors": 0, "skipped": 0},
        "artifact_seal": seal, "full_training_replay": audit,
        "historical_artifacts_unchanged": len(original), "prior_python_files_unchanged": len(prior)-len(changed),
        "intentional_prior_python_change": changed,
        "historical_local_audit_runs": 159, "historical_fresh_answers": 3180,
        "reports_rebuilt_identically": rebuilt, "local_markdown_links_checked": links,
        "source_sha256": {name: pilot.sha(ROOT / name) for name in new_sources},
        "evidence_sha256": {p.relative_to(ROOT).as_posix(): pilot.sha(p) for p in [
            review / "partial-tests-final.xml", review / "partial-tests-final.txt",
            review / "partial-pilot-replay-audit.txt", review / "partial-historical-audits.json",
            review / "partial-optimizer-verification.json", results / "artifact-manifest.json"]},
        "limitations": ["Development only; one original dataset, initialization seed and subset draw",
                        "No confirmation data/training, professor meeting or exhaustive novelty review",
                        "Source hashes support reproducibility; no guarantee of 100% correctness"]}
    target = review / "partial-delivery-verification.json"
    assert not target.exists()
    pilot.save_json(target, validation)
    print(json.dumps({k: validation[k] for k in ("status", "verified_at_utc", "final_full_suite", "artifact_seal",
        "historical_artifacts_unchanged", "prior_python_files_unchanged", "reports_rebuilt_identically", "local_markdown_links_checked")}, indent=2))


if __name__ == "__main__":
    main()
