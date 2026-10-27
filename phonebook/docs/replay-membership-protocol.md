# Replay membership and partitioned evaluation, version 1

This gate prepares immutable **declared adaptation inputs** and checks their evaluation partitions. Weighted training and training-command linkage are subsequent work. Every report in this version contains `training_execution_verified: false`: a matching text file does not establish that a checkpoint was trained using it.

The implementation is in [replay_protocol.py](../replay_protocol.py) and [evaluate_replay.py](../evaluate_replay.py). The [October 20 review](review-2026-10-20/replay-membership-review.md) records tests, reference-checkpoint checks, limitations, and the next implementation gate.

## Identity and selection

The schema is `phonebook-replay-membership-v1`. Each record ID is SHA-256 of its UTF-8 `prompt=target\n` record after newline normalization. Prompt identity is exact and case-sensitive. Source LF/CRLF differences and a missing final source newline preserve record identity; record order, prompts, or targets remain part of the normalized book hash. No Unicode or case normalization is silently applied.

Books must be nonempty, have one delimiter per record, and contain nonempty prompt/target text. Duplicate or conflicting prompts within a book or across A/B fail. Surrounding whitespace, reserved `_` padding characters, blank records, and embedded control characters fail. The current inferred-context trainer additionally requires A and B to have equal longest record lengths. Every prepared condition contains all B records, so this restriction preserves a common context even for B-only and smaller replay subsets.

The protocol declares one common character vocabulary, using the existing forced alphanumeric alphabet and vocabulary ordering. Evaluation rejects incompatible metadata, including a same-sized permutation of token IDs, before any model inference.

Selection uses `sha256-rank-v1`: sort IDs by the SHA-256 of compact JSON `[algorithm, seed, record_id]`, with record ID as a tie-breaker. This makes the ordering explicit and independent of Python/NumPy/Torch RNG state. It is a deterministic seeded permutation, not selection based on learning outcomes. It is not a claim that repeated subset draws are independent datasets.

Defaults implement the first development layout: three A probe records ranked with seed 1733; the remaining seven form the replay pool; seed 1701 orders that pool; nested prefixes select m=1,2,5. Another declared draw such as 1702 is supported, but the default bundle does not prepare a confirmation run matrix. The implementation fixes the previously unspecified permutation algorithm before any research pilot.

The six default declarations are B-only, partial replay m=1/2/5, full replay, and fresh joint. The last two deliberately include the probe in their training membership. Their exposed-probe measurements are positive-control layouts, not evidence of preservation without rehearsal. “Unreplayed” means excluded from the declared **adaptation input**; all A facts may already have been seen during Phase A.

## Pin and materialize

From the repository root:

```powershell
python phonebook/replay_protocol.py --out-root NEW_PREPARED_DIRECTORY
```

This writes `protocol.json` and a unique `train.txt` per condition, then prints the canonical protocol digest. The JSON records source paths/hashes, full record identities, probe/pool IDs, complete draw permutations, condition membership/order, normalized training-text hashes, vocabulary, context, and initialization intent. Creation rejects existing output directories. Validation finishes before creating output files.

Freeze the digest in the experiment declaration before training. An evaluator requires that **external pin**; a self-reported or freshly recomputed digest does not demonstrate that a declaration preceded the results. On loading, the code verifies the pin and reconstructs the complete protocol from the source books and declared selection. Changing a subset, probe, target, vocabulary, or context and merely rehashing the edited JSON is insufficient to pass reconstruction.

The validated Python membership object contains only frozen records and tuples, detached from the mutable JSON input. The full source catalog remains analyst/evaluation evidence. This does not claim that total system storage is limited to m records; future training must consume only the declared prepared rows when computing adaptation gradients.

Prepared training text must match the exact declared rows, targets, multiplicity, and order, including a final newline. LF/CRLF variants are accepted and their normalized hash must match. Probe insertion, duplicate rows, omissions, reordered rows, and changed answers fail before inference. Relative source paths can be resolved against a relocated repository root.

## Evaluate and audit

Use the actual printed digest in place of `PIN_FROM_PREPARATION`:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
python phonebook/evaluate_replay.py --protocol NEW_PREPARED_DIRECTORY/protocol.json --protocol-sha256 PIN_FROM_PREPARATION --condition partial_m1_seed1701 --train-input NEW_PREPARED_DIRECTORY/partial_m1_seed1701/train.txt --checkpoint CHECKPOINT_PATH --meta METADATA_PATH --out-root NEW_EVALUATION_DIRECTORY
```

The checkpoint context must equal the declared common context, and metadata must match the common vocabulary. Source membership and the protocol pin are checked again after evaluation, as are the exact checkpoint/metadata bytes. Invalid or nonfinite teacher/generation logits fail. Existing evaluation directories are rejected, and nonfinite output cannot be serialized.

Evaluation computes per-record teacher statistics in one common batch and generates each A/B answer once using the automatic context budget. Prompt and padding positions are excluded from teacher supervision. Saved generated token IDs include the terminal newline, allowing independent strict recall checks. Both teacher and generation are then partitioned from the same raw records:

- `A_all`: every old record.
- `A_replayed`: the old records present in the declared adaptation input.
- `A_unreplayed`: its complement in A.
- `A_probe`: the same fixed A panel for every condition.
- `B`: every new record.

Partition counts sum exactly at the record/token level. Loss is summed task CE divided by supervised-token count; rates are not averaged across unequal partitions. Empty partitions have zero integer counts, `status: not_applicable`, and null accuracy/loss. Thus full-replay/joint complements and the B-only replay partition do not display a misleading 0% score. `probe_exposed` states whether that condition's prepared rows include probe facts.

Readable auditing needs source books and prepared training text, but does not load weights:

```powershell
python phonebook/evaluate_replay.py --protocol NEW_PREPARED_DIRECTORY/protocol.json --protocol-sha256 PIN_FROM_PREPARATION --condition partial_m1_seed1701 --train-input NEW_PREPARED_DIRECTORY/partial_m1_seed1701/train.txt --audit-report NEW_EVALUATION_DIRECTORY/evaluation.json
```

Add `--require-local-artifacts` to hash the recorded checkpoint/metadata and regenerate all answers and teacher statistics. This requires those files at their recorded locations. Readable mode establishes raw-string/count/partition consistency; it cannot authenticate a co-edited teacher loss and all its aggregates. Fresh-inference mode detects that substitution. Neither mode authenticates an adaptation training history in this version.

Evaluation restores the caller's CPU RNG, model module modes, and existing gradients under the tested transformer workflow. It performs no optimizer steps. This is not a guarantee for arbitrary state-mutating custom model implementations or GPU workflows.

## Reproduce the correctness gate

All 83 new tests use temporary source data/checkpoints; they require no historical ignored model artifacts:

```powershell
python -m pytest tests/test_replay_protocol.py tests/test_replay_evaluation.py -q
```

The recorded six-layout engineering check additionally uses the historical Phase-A checkpoint. It evaluates the **same reference checkpoint** under six partition layouts and compares whole-book results with the existing evaluator. Its source and outputs are linked from [the review record](review-2026-10-20/replay-membership-review.md). It performs no adaptation or fresh-joint initialization/training.

The subsequent [partial-replay runner and development report](review-2026-10-20/partial-replay-execution.md) now integrate pinned binaries/common vocabulary, initialization/source/command receipts, the checked book-balanced objective, work logs, endpoint checkpoints, and a deterministic training replay audit. The six-condition development pilot is complete. Standalone evaluation retains its original limited flag; execution reproducibility is established separately by the runner audit. Confirmation still requires the scientific-review gate in the [research plan](week-2026-10-12-to-20.md).
