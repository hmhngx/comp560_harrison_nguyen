"""Engineering-only check; run from repository root with one UNUSED output path."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import torch
from phonebook import replay_protocol as rp
from phonebook import evaluate_replay as er
from phonebook.evaluate_checkpoint import evaluate_checkpoint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        raise ValueError("Output directory must be unused")
    torch.set_num_threads(1)
    paths = ["phonebook/replay_protocol.py", "phonebook/evaluate_replay.py", "phonebook/evaluate_checkpoint.py",
             "completion_core/inference.py", "completion_core/modeling.py", "completion_core/training.py",
             "completion_core/prep.py", "completion_core/vocabulary.py", "completion_core/validation.py",
             "phonebook/prepare_phonebook.py", "phonebook/docs/review-2026-10-20/verify_membership_gate.py"]
    sources = {p: sha(ROOT/p) for p in paths}
    checkpoint, meta = ROOT/"phonebook/out_phaseA/model.pth", ROOT/"phonebook/data_phaseA/meta.pkl"
    spec = rp.read_json(ROOT/"phonebook/specs/e3_replay_ratio.json")
    init = next(s["init"] for s in spec["scenarios"] if s["id"] == "d0_seed42")
    assert sha(checkpoint) == init["checkpoint_sha256"] and sha(meta) == init["meta_sha256"]
    books = {"A": ROOT/"phonebook/inputs/phonebookA.txt", "B": ROOT/"phonebook/inputs/phonebookB.txt"}
    protocol = rp.build_protocol("phonebook/inputs/phonebookA.txt", "phonebook/inputs/phonebookB.txt")
    pin = rp.write_protocol(protocol, out/"prepared")
    with torch.random.fork_rng(devices=[]):
        baseline = evaluate_checkpoint(checkpoint, meta, books)
    (out/"legacy-baseline.json").write_text(json.dumps(baseline,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    cases=[]
    for condition in protocol["conditions"]:
        name=condition["id"]
        selected=rp.membership(protocol,name,pin)
        training=out/"prepared"/name/"train.txt"
        result=er.evaluate_checkpoint_partitions(checkpoint,meta,out/"prepared/protocol.json",pin,name,training)
        destination=out/"evaluations"/name
        er.write_evaluation(result,destination)
        audits={"readable": er.audit_saved_evaluation(result,out/"prepared/protocol.json",pin,name,training),
                "full_local": er.audit_saved_evaluation(result,out/"prepared/protocol.json",pin,name,training,
                                                        require_local_artifacts=True)}
        body=result["evaluation"]
        for label,book in (("A_all","A"),("B","B")):
            observed=body["partitions"][label]
            reference=baseline["books"][book]
            assert observed["generation"]["correct"] == reference["generation"]["correct"]
            assert observed["generation"]["total"] == reference["generation"]["total"]
            assert math.isclose(observed["teacher_forced"]["loss"],reference["teacher_forced"]["loss"],rel_tol=1e-6,abs_tol=1e-9)
            predictions={r["prompt"]:r["generation"] for r in body["records"]}
            for old in reference["generation"]["predictions"]:
                current=predictions[old["prompt"]]
                assert all(current[key]==old[key] for key in ("prediction","stop_reason","correct"))
        record_by_id={r.id:r for r in selected.a+selected.b}
        cases.append({"condition":name,"kind":selected.kind,"prepared_sequences":len(selected.training_ids),
                      "prepared_supervised_tokens":sum(len(record_by_id[i].target)+1 for i in selected.training_ids),
                      "prepared_padded_input_tokens":len(selected.training_ids)*(selected.context_length-1),
                      "probe_exposed":body["probe_exposed"],
                      "partitions":body["partitions"],"audits":audits})
        print(name,"prepared membership, both audits, and legacy-score comparison passed",flush=True)
    assert sources=={p:sha(ROOT/p) for p in paths}
    summary={"status":"passed","purpose":"Engineering partition views of one reference Phase-A checkpoint; no condition training or initialization executed",
             "protocol_sha256":pin,"checkpoint_sha256":sha(checkpoint),"metadata_sha256":sha(meta),
             "source_sha256":sources,"conditions":cases,"optimizer_steps":0,"research_training_runs":0,
             "initial_evaluation_answers":len(cases)*20,"fresh_audit_answers":len(cases)*20,
             "legacy_reference_answers":20}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (out/"partition-counts.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.writer(file,lineterminator="\n")
        writer.writerow(["condition","partition","status","correct","total","token_correct","token_total","loss","probe_exposed"])
        for case in cases:
            for name,p in case["partitions"].items():
                writer.writerow([case["condition"],name,p["status"],p["generation"]["correct"],p["generation"]["total"],
                                 p["teacher_forced"]["token_correct"],p["teacher_forced"]["token_total"],p["teacher_forced"]["loss"],case["probe_exposed"]])
    print("passed",len(cases),"partition views; zero optimizer steps",flush=True)


if __name__ == "__main__":
    main()
