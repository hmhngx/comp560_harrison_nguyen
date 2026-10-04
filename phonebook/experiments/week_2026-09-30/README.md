# September 30–October 6: evaluation corrections and bounded replay pilot

Protocol written October 4 before running the new training conditions.

## Question

Does balanced replay's zero exact B recall at lr=0.0001, 100 updates persist
with 500 updates or lr=0.001? Can the same architecture memorize A+B from
fresh initialization under the tested budget?

The previous log compared L2-SP B token accuracy to replay B exact recall.
That comparison cannot establish a recall advantage. Existing seed42
checkpoints have 10/10 A and 0/10 B generation recall for both methods at
the cited full-retention settings. B token accuracy is 26.92% for L2-SP λ=2,
61.54% for replay k=1, and 53.08% for their tested combination.

## Protocol

- Fixed A and B: ten unique, disjoint names per book. Balanced replay is
  exactly A once followed by B once. Sources are version-controlled.
- Model: 64 dimensions, 2 heads, 2 layers. Batch size 64, CPU, seed42,
  default AdamW settings from the committed trainer, no L2-SP.
- Twenty training records give exactly one optimizer update per epoch.
- Four primary runs: replay lr=0.0001 at 100 and 500 updates; replay
  lr=0.001 at 100 updates; fresh joint A+B training lr=0.001 at 100 updates.
- Predeclared follow-ups: extend lr=0.001 replay and fresh joint training
  to 500 updates if either book lacks perfect generation at 100 updates.
  Every adaptation starts anew from the original Phase-A checkpoint.
- Evaluation: A and B separately, strict target-token match including
  newline; generation uses the context limit without target-length hints.
  Teacher-forced loss, token accuracy, sequence accuracy, counts, and raw
  generation predictions are retained independently for each book.
- Code must be committed and the tree clean before training. The runner
  refuses an existing output root, preserving all historical evidence.
- `protocol.json` records input/checkpoint hashes, code commit, environment,
  primary conditions, and extension triggers before training. Each run has
  an exact command, exit status, runtime, log, and hashed final evaluation.
- Only endpoint checkpoints are saved. The pilot cannot measure the first
  update reaching joint perfection, or validate generation at intermediate epochs.

## Reproduce

Run from the repository root in the recorded environment. Rebuild the
original Phase-A artifacts using `phonebook/README.md` if missing, and
verify their hashes against `protocol.json` before expecting identical results.

```powershell
python phonebook/run_replay_pilot.py --dry-run
python phonebook/run_replay_pilot.py --out-root phonebook/experiments/week_2026-09-30-rerun
```

After inspecting the completed seed42 pilot, two fixed 100-update replay
conditions (lr=0.0001 and lr=0.001) were selected for Phase-A checkpoint
replication. This follow-up was selected after the primary results, rather
than being part of the original predeclared pilot. Phase-B training seed
remains 42; only the source Phase-A checkpoint changes. No conditional
extensions or fresh-joint runs are included in this follow-up.

```powershell
python phonebook/run_replay_pilot.py --replication-phase-a-seed 123
python phonebook/run_replay_pilot.py --replication-phase-a-seed 999
```

The runner prepares new binary data directly from the tracked sources.
It additionally evaluates the original seed42 Phase-A, L2-SP λ=2, replay k=1/3/5,
and combination checkpoints, plus Phase-A seeds123/999 and their λ=5 L2-SP
checkpoints. These historical local checkpoint files must
exist for that audit. Their hashes identify the exact evaluated artifacts.

Checkpoints and prepared binaries are intentionally ignored. Small result
files are explicitly committed after the run, including the raw predictions,
summary, protocol, preparation record, logs, and run manifest.

## Results

The seed42 pilot completed successfully from clean code commit `54dde8e`.
All counts below are strict, newline-terminated generation, evaluated on
A and B independently. Each book has ten entries.

| Initialization and data | Learning rate | Updates | A correct | B correct |
|---|---|---|---|---|
| Phase-A seed42, balanced replay | 0.0001 | 100 | 10/10 | 0/10 |
| Phase-A seed42, balanced replay | 0.0001 | 500 | 10/10 | 10/10 |
| Phase-A seed42, balanced replay | 0.001 | 100 | 10/10 | 10/10 |
| Fresh initialization, joint A+B | 0.001 | 100 | 10/10 | 10/10 |

The two conditional 500-update extensions were not triggered: both
lr=0.001 conditions already had perfect A and B generation at 100 updates.
This establishes joint memorization at the tested successful endpoints;
the old zero-B result is sensitive to optimization settings, rather than
evidence that balanced replay necessarily prevents learning B.

The selected follow-up completed from clean code commit `54ff994`:

| Phase-A checkpoint | Replay lr | Updates | A correct | B correct | B token accuracy |
|---|---|---|---|---|---|
| seed42 | 0.0001 | 100 | 10/10 | 0/10 | 61.54% |
| seed123 | 0.0001 | 100 | 10/10 | 1/10 | 76.15% |
| seed999 | 0.0001 | 100 | 10/10 | 0/10 | 79.23% |
| seed42 | 0.001 | 100 | 10/10 | 10/10 | 100% |
| seed123 | 0.001 | 100 | 10/10 | 10/10 | 100% |
| seed999 | 0.001 | 100 | 10/10 | 10/10 | 100% |

Increasing lr from 0.0001 to 0.001 at the same 100-update budget reaches
joint memorization for all three available Phase-A checkpoints. Seed123's
1/10 B recall at the original setting also shows that zero exact B recall
was not universal across checkpoints. These are three initial checkpoints
on the same A/B data, with Phase-B seed42 held fixed; they are not three
independent datasets or a study of Phase-B seed variance.

Machine-readable seed42 evidence: [summary](run_2026-10-04/summary.csv),
[protocol](run_2026-10-04/protocol.json),
[manifest](run_2026-10-04/manifest.jsonl).
Follow-up summaries: [seed123](replication_phaseA_seed123/summary.csv),
[seed999](replication_phaseA_seed999/summary.csv).

## Independent verification

`audit_results.py` recomputes exact recall directly from the saved raw
strings and stop reasons, checks prompt/target coverage against the tracked
sources, checks token/sequence denominators, reconciles summaries with
evaluations and rounded final training CSVs, checks all logged metrics for
NaN/Inf and missing epochs, and reconciles executed runs with the protocol.
Its full local mode also checks checkpoint, metadata, and prepared-bin hashes.

```powershell
python phonebook/experiments/week_2026-09-30/audit_results.py --require-local-artifacts
```

The [audit report](audit.json) covers eight successful training runs,
20 checkpoint evaluations, and 400 raw completions. Without local model
and prepared-bin artifacts, omit `--require-local-artifacts` to reconcile
the committed readable evidence. This cannot independently rerun model
inference when the weights are absent.
Fault checks on temporary copies confirmed that the auditor rejects an
altered raw prediction and a missing final logged epoch; original evidence
was preserved.

Regression coverage includes budgets below/equal to the answer length,
missing newline, excess generated characters/padding, context exhaustion,
two-character tokenization, invalid/empty splits, stale-output protection,
unterminated binary records, and separate B evaluation despite A duplication.
The complete suite's final result is recorded in
[validation_tests.log](validation_tests.log) and
[validation_tests.xml](validation_tests.xml): **102 passed in 171.09 seconds,
zero skips and warnings**. The real-phonebook integration test ran using
the local ignored checkpoint/data artifacts; a fresh clone may skip that
test until those artifacts are rebuilt. Direct `generate.py` runs with no
budget override also reported 10/10 for untouched Phase-A A and the
successful seed42 replay B.

## Scope

Three Phase-A checkpoints on one fixed twenty-entry task, two learning rates,
and two update budgets (500 updates tested only from seed42). This establishes only the observed behavior at
those settings. No claim about optimal replay weighting, general method
superiority, independent-dataset replication, or maximum model capacity follows.

Code, audit, and writing used Codex assistance; the measurements must be
checked against the committed machine-readable evidence.
