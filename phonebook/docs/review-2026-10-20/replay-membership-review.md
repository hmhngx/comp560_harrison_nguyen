# Approved replay-membership and measurement gate — October 20, 2026

## Outcome and execution boundary

The new versioned protocol and evaluator implement pinned replay/probe membership and separate A-replayed, A-unreplayed, A-all, fixed-probe, and B metrics. The existing trainer, whole-book runner, evaluator, and historical auditor remain unchanged in this gate. This follows the [weighted-objective check](weighted-objective-review.md).

The new [protocol guide](../replay-membership-protocol.md) describes the schema, identity rules, CLI commands, pin trust boundary, empty-partition semantics, and audit modes. The code is in [replay_protocol.py](../../replay_protocol.py) and [evaluate_replay.py](../../evaluate_replay.py), with [protocol tests](../../../tests/test_replay_protocol.py) and [evaluation tests](../../../tests/test_replay_evaluation.py).

Every evaluation explicitly records `training_execution_verified: false`. This gate binds a **prepared text file** to declared membership and checks measurements. It does not establish which data produced a checkpoint. Weighted training, binary preparation with the common vocabulary, command/initialization linkage, and a tiny integrated workflow remain the next implementation gate before the development pilot.

## Protocol and review decisions

- Record IDs hash normalized UTF-8 completion records. Exact prompt duplicates/conflicts fail within and across books. Padding/whitespace/control-character ambiguities and empty/malformed records fail.
- A fixed three-record probe is excluded from the seven-record replay pool. One declared seed1701 draw produces nested m=1/2/5 prefixes; the default bundle has six layouts. Additional declared draws are supported and tested. The selection algorithm is explicitly `sha256-rank-v1`, using probe seed1733 and isolated seed-based hashes; no global RNG stream is consumed. This fixes the algorithm before observing pilot results.
- Full replay and fresh joint expose the probe and are marked accordingly. The unreplayed complement is empty/N/A. B-only has an empty/N/A replayed partition, with all A in the unreplayed partition.
- Loading requires an external canonical digest and reconstructs the entire protocol from the source books and selection. A co-edited subset/text hash, probe, target, context, or vocabulary does not pass merely by rehashing its JSON. The resulting Python object has frozen records/tuples.
- Physical training text is checked before inference, including order, targets, duplicates, omissions, and probe exclusion. Newline variants preserve normalized identity; missing final training newlines fail. Source-relative relocation is tested.
- The current trainer infers context from training text. The preparer therefore rejects unequal A/B maximum record lengths; all declared conditions include B, preserving the common context. Evaluation also rejects an incompatible checkpoint context or vocabulary before inference.
- All partition metrics derive from the same per-record outputs. Raw generated token IDs, stop reasons, and target identities support independent strict-recall counting. Teacher losses use supervised-token sums, with prompt/pad masks excluded. Empty partition accuracy/loss is null, not a fabricated zero.
- Readable auditing reconciles raw IDs/strings/counts/partitions. Full-local auditing additionally verifies the model/meta hashes and reruns inference. A test deliberately co-edits a teacher loss and all aggregate losses: readable consistency passes, while fresh inference rejects it. The distinction is intentional and explicitly reported.
- Every model forward is checked for finite, correctly shaped logits. Evaluation restores mixed module modes and CPU RNG even on failure, and leaves tested weights/gradients unchanged. Protocol and evaluation writers reject existing outputs and nonfinite JSON before output creation.

## Tests and evidence

The [initial protocol test run](membership-tests-initial.txt) exposed one defect: Python dictionary equality accepted a forged boolean vocabulary index `False` as integer `0`. The test rehashed the edited protocol to exercise internal reconstruction separately from the external pin. Canonical JSON comparison now preserves strict JSON types and rejects it. The initial run had **1 failure and 47 passes**.

The final [focused output](membership-tests-focused.txt) and [JUnit report](membership-tests-focused.xml) pass **83 cases in 19.22 seconds**. Tests cover deterministic/nested selection; no RNG consumption; all control layouts; frozen membership; external pins; rehashed edits; malformed/duplicate/conflicting prompts; invalid sizes/seeds; context/vocabulary checks; newline/relocation behavior; exact prepared rows; hand-computed count/CE checks with unequal target lengths; empty partitions; raw-token/partition mutations; failure-state restoration; strict JSON; both CLIs; and fresh checkpoint auditing. All checkpoints needed by these tests are created temporarily; none requires historical ignored weights.

The full [test output](membership-tests-full.txt) and [JUnit report](membership-tests-full.xml) pass **581 cases in 409.32 seconds**, with zero failures, errors, or skips and no reported warnings. This is the preceding 498-test suite plus 83 new cases. [Preservation checks](membership-preservation.json) confirm that all **59 pre-existing Python files** and **1,337 original data/result/artifact files** remain byte-identical. The [final validation summary](membership-change-validation.json) records exact source/evidence hashes. New source, tests, and the engineering script were reviewed and syntax-parsed; `git diff --check` passes.

The last seven-root, 159-record / 3,180-answer historical audit remains [the preceding training/auditor validation](finite-loss-historical-validation.json). Its dependencies are unchanged. The six new reports additionally pass their own readable and fresh-inference audits as recorded below.

## Reference-checkpoint engineering check

[The reproducible check script](verify_membership_gate.py) prepares the original books with the default six layouts, evaluates the **same historical Phase-A reference checkpoint** in each layout, audits every report in readable and full-local modes, and compares its whole-book predictions/CE with the existing evaluator. It executes zero optimizer steps and no condition training or initialization. In particular, the `fresh_joint` entry is a test of its **partition layout**, not a trained fresh-joint control.

All six layouts pass both audits and match the existing evaluator. The reference checkpoint recalls A=10/10, B=0/10, and the fixed A probe=3/3 in each view. These are pre-adaptation measurement checks, not partial-replay retention outcomes.

| Declared layout | Prepared unique sequences | Prepared supervised tokens | Probe exposed |
|---|---:|---:|---|
| B-only | 10 | 130 | No |
| m=1, seed1701 | 11 | 143 | No |
| m=2, seed1701 | 12 | 156 | No |
| m=5, seed1701 | 15 | 195 | No |
| Full replay | 20 | 260 | Yes |
| Fresh joint layout | 20 | 260 | Yes |

The canonical protocol pin is `8bd47398f7ea33564e646dbcf04e6e9c84aa6344bf778a642b2b9f4b8093bd30`. These counts describe the prepared data, not executed training work. Full-local audits regenerated 120 answers in addition to the 120 initial partition evaluations; a separate existing-evaluator baseline generated 20 answers. No original historical report or model file was overwritten.

Inspect [the prepared protocol](replay-membership-evidence/prepared/protocol.json), [engineering summary](replay-membership-evidence/summary.json), [partition-count table](replay-membership-evidence/partition-counts.csv), [legacy baseline](replay-membership-evidence/legacy-baseline.json), and [execution log](membership-engineering-output.txt). The summary records each condition's audit results; per-condition raw token reports are under `replay-membership-evidence/evaluations/CONDITION/evaluation.json`.

To rerun with the required local historical artifacts, from repository root:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
python phonebook/docs/review-2026-10-20/verify_membership_gate.py NEW_OUTPUT_DIRECTORY
```

## Next step

Integrate the pinned membership and already verified weighted objective with training and the versioned run evidence. Validate prepared binary rows and common vocabulary, initialization/context, float32 behavior, objective weights, actual work, and full-local score regeneration. Preserve the existing unweighted workflow. The six-condition development pilot remains behind those checks; the provisional confirmation grid remains unexecuted.

No commit, push, professor message, meeting booking, activity-log submission, or research training was performed in this gate. These new files remain local until published.
