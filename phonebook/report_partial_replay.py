"""Rebuild all-cell development summaries from audited raw partition reports."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phonebook.replay_protocol import canonical_digest, read_json, membership
from phonebook.run_partial_replay import require, save_json, evaluation_updates, endpoints, audit_pilot, sha


def artifact_manifest(root):
    root = Path(root)
    return {"schema": "phonebook-development-artifacts-v1", "sha256": {
        path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts}}


def verify_artifacts(root, manifest, expected_pin):
    require(canonical_digest(manifest) == expected_pin, "Artifact manifest differs from external pin")
    require(canonical_digest(artifact_manifest(root)) == expected_pin, "Pilot artifact bytes or coverage differ")
    return {"status": "passed", "files": len(manifest["sha256"]), "manifest_sha256": expected_pin}


def derive(root, pin):
    root = Path(root)
    spec = read_json(root / "specification.json")
    require(canonical_digest(spec) == pin, "Specification differs from external pin")
    protocol = read_json(root / "prepared" / "protocol.json")
    settings = spec["settings"]
    rows, curves, observations = [], [], []
    for condition in protocol["conditions"]:
        item = membership(protocol, condition["id"], spec["protocol_sha256"])
        folder = root / "prepared" / item.condition_id
        reports = {u: read_json(folder / f"evaluation-{u:04d}.json") for u in evaluation_updates(item, settings)}
        run = read_json(folder / "run.json")
        for update, report in reports.items():
            counts = {p: v["generation"]["correct"] for p, v in report["evaluation"]["partitions"].items()}
            curves.append({"condition": item.condition_id, "update": update, **counts})
        common = [(u, r) for u, r in reports.items() if u <= settings["primary_updates"]]
        def count(report, name):
            return report["evaluation"]["partitions"][name]["generation"]["correct"]
        def joint(report):
            return count(report, "A_probe") == len(item.probe_ids) and count(report, "B") >= len(item.b) - 1
        first = next((u for u, report in common if joint(report)), None)
        sustained = next((u for i, (u, report) in enumerate(common) if all(joint(r) for _, r in common[i:])), None)
        area = sum((b-a) * (1 - (count(ra, "A_probe") + count(rb, "A_probe")) / (2 * len(item.probe_ids)))
                   for (a, ra), (b, rb) in zip(common, common[1:]))
        observations.append({"condition": item.condition_id, "first_sampled_probe_B_joint_update": first,
                             "sampled_joint_sustained_through_primary_from": sustained,
                             "minimum_sampled_A_probe_correct": min(count(r, "A_probe") for _, r in common),
                             "sampled_trapezoidal_A_probe_error_update_units": area,
                             "training_seconds": run["training_seconds"], "evaluation_seconds": run["evaluation_seconds"],
                             "diagnostic_seconds": run["diagnostic_seconds"], "parameter_count": run["parameter_count"],
                             "parameter_bytes": run["parameter_bytes"], "optimizer_tensor_bytes": run["optimizer_tensor_bytes"],
                             "retained_replay_records": run["retained_replay_records"],
                             "retained_replay_utf8_bytes": run["retained_replay_utf8_bytes"],
                             "anchor_tensor_bytes": run["anchor_tensor_bytes"]})
        for label, update in (("secondary", settings["secondary_updates"]), ("primary", settings["primary_updates"]),
                               ("sequence_budget", endpoints(item, settings)[-1])):
            report = reports[update]
            partitions = report["evaluation"]["partitions"]
            row = {"condition": item.condition_id, "endpoint": label, **report["work"],
                   "replay_records": len(item.replay_ids), "probe_exposed": report["evaluation"]["probe_exposed"]}
            for name, values in partitions.items():
                row[name + "_correct"] = values["generation"]["correct"] if values["status"] == "ok" else None
                row[name + "_total"] = values["generation"]["total"]
                row[name + "_CE"] = values["teacher_forced"]["loss"]
            rows.append(row)
    controls = [r for r in rows if r["endpoint"] == "primary" and r["condition"] in ("full_replay", "fresh_joint")]
    controls_pass = len(controls) == 2 and all(r["A_all_correct"] == r["A_all_total"] and r["B_correct"] == r["B_total"] for r in controls)
    primary = {r["condition"]: r for r in rows if r["endpoint"] == "primary"}
    m1, m5 = primary.get("partial_m1_seed1701"), primary.get("partial_m5_seed1701")
    directional = (m5["A_probe_correct"] > m1["A_probe_correct"] and m5["B_correct"] >= 9) if m1 and m5 else None
    return {"specification_sha256": pin, "stage": "development_only", "controls_pass_at_primary": controls_pass,
            "directional_hypothesis_observed_in_development": directional,
            "confirmation_ready": False, "endpoints": rows, "sampled_observations": observations,
            "curves": curves, "independent_datasets": 1, "subset_draws": len(protocol["draws"]),
            "sampling": {"seed": settings["seed"], "evaluation_interval": settings["evaluation_interval"],
                         "primary_updates": settings["primary_updates"]},
            "limitations": ["Original data and seed42 only; reused development data",
                            "Three fixed probe facts give 33-percentage-point resolution",
                            "First/minimum/sustained recall and integrals refer only to sampled updates",
                            "Sequence-budget analysis changes optimizer-step and decay counts",
                            "Teacher losses authenticate through fresh inference/training replay, not readable counts alone",
                            "No population inference, capacity limit or novelty claim"]}


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "summary.json", {"generated_at_utc": datetime.now(timezone.utc).isoformat(), **report})
    for name, rows in (("endpoints", report["endpoints"]), ("sampled-observations", report["sampled_observations"]),
                       ("curves", report["curves"])):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    primary = next(r for r in report["endpoints"] if r["endpoint"] == "primary")
    horizon = primary["updates"]
    probe_count, b_count = primary["A_probe_total"], primary["B_total"]
    for condition in dict.fromkeys(r["condition"] for r in report["curves"]):
        curve = [r for r in report["curves"] if r["condition"] == condition and r["update"] <= horizon]
        style = "--" if condition in ("full_replay", "fresh_joint") else "-"
        for axis, partition, maximum in zip(axes, ("A_probe", "B"), (probe_count, b_count), strict=True):
            axis.plot([r["update"] for r in curve], [r[partition] for r in curve], style, label=condition, linewidth=1.6)
            axis.set_ylim(-0.2, maximum + 0.2)
            axis.set_yticks(range(maximum + 1))
            axis.set_xlabel("Optimizer updates")
            axis.set_ylabel("Strict recall count")
            axis.set_title(f"Fixed A probe ({probe_count} facts)" if partition == "A_probe" else f"Book B ({b_count} facts)")
            axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    sampling = report["sampling"]
    draws = "one memory draw" if report["subset_draws"] == 1 else f"{report['subset_draws']} memory draws"
    figure.suptitle(f"Development data / seed{sampling['seed']} / {draws} — sampled every {sampling['evaluation_interval']} updates")
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    figure.savefig(output / "recall.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--specification-sha256", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    audit_pilot(args.root, args.specification_sha256)
    report = derive(args.root, args.specification_sha256)
    write_report(report, args.out_root)
    manifest = artifact_manifest(args.root)
    save_json(args.out_root / "artifact-manifest.json", manifest)
    print(f"Artifact manifest pin: {canonical_digest(manifest)}")
    print(f"Rebuilt {len(report['endpoints'])} endpoint rows; controls pass: {report['controls_pass_at_primary']}")


if __name__ == "__main__":
    main()
