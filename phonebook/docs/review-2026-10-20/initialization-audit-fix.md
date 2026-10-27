# Approved provenance gate: initialization auditing — October 20, 2026

## Scope and decision

After the RNG repair, the user approved the next step: reconcile initialization records against the committed experiment spec. This follow-up changes the auditor and adds regression tests. It does not change training, datasets, model weights, or historical experiment records, and does not run the proposed partial-replay study.

The changes are in [the auditor](../../audit_experiment.py) and [the initialization regression tests](../../../tests/test_audit_initializations.py). The implementation covers three related ways an apparently consistent report could evade its declared initialization:

1. A protocol's checkpoint or vocabulary hash could contradict the committed pin and still pass readable auditing.
2. Co-editing the checkpoint path and the manifest's training command could preserve their agreement while substituting an undeclared anchor. Vocabulary paths were also not reconciled with the declaration.
3. Removing `spec_path` skipped the Git comparison while the report still returned `git_checked=True`.

Review also found that comparing context metadata only with other recorded context metadata was insufficient: both could be co-edited. The auditor now reconstructs the expected adaptation context from the declared training text and mixture, consistent with the current trainer's context-sizing rule.

## Implementation

- Initialization scenarios must exactly match scenarios with an `init` declaration. Fresh-training specs continue to require an empty initialization mapping.
- Each initialization must have the five fields emitted by the current runner: checkpoint path/hash, metadata path/hash, and context length. Malformed records, missing/extra fields, and invalid hashes or context values fail with an initialization-specific error.
- Declaration paths are compared verbatim because the runner copies them verbatim. This prevents substitution while preserving relocation of unchanged repository-relative declarations. Training-command relocation checks remain separate.
- Every hash pin present in the spec must match the recorded initialization hash in **both** audit modes. Default readable auditing does not open checkpoint or vocabulary binaries.
- When `check_git=True`, a nonempty recorded spec path is mandatory, and the embedded spec must match the spec at the recorded Git commit. Missing paths can no longer produce a successful report claiming Git verification.
- `check_git=False` remains available for synthetic/programmatic fixtures and explicitly reports `git_checked=False`. It does not provide a committed-spec guarantee.
- Adaptation context metadata and run lengths must match the length reconstructed from the training records. This is tied to the current trainer's inferred-context behavior; a future fixed-context protocol must update the runner and auditor together.

An unpinned, well-formed digest cannot be authenticated without the actual file. Such specs remain supported, but checking their file bytes requires full-local mode. This gate checks internal/committed-declaration consistency; it is not independent attestation that training actually occurred at a claimed time.

## Tests and review evidence

- Before the fix, four pin/path mutations were accepted: both hash substitutions, a co-edited checkpoint/command substitution, and a metadata-path substitution. The baseline passed. See [pin and path failures](pin-tests-before.txt).
- Three missing-spec-path cases were also accepted while claiming Git verification. The explicit Git opt-out control passed. See [Git-binding failures](pin-spec-path-before.txt).
- During review, a co-edited initialization/run context still passed an intermediate implementation that checked only mutual agreement. That regression was reproduced, then closed by checking the training text. See [context regression](pin-context-before.txt).
- The final focused initialization file passed all 41 cases present before the clone test was added. See [focused output](pin-tests-focused.txt) and [JUnit report](pin-tests-focused.xml).
- The current auditor also passed on a clean local Git clone in a different directory with no checkpoints, with Git verification enabled. This tests checkout relocation on the current platform, not every cross-platform path convention. See [relocation output](pin-tests-relocation.txt).
- All seven historical experiment folders pass in **both readable and full-local modes**. Full-local auditing regenerated **3,180 answers across 159 training records**, and all **1,337 original data/result/artifact files** checked against the first inventory remain byte-identical. See [historical validation](pin-historical-validation.json).
- The full suite passed **350 tests in 446.95 seconds**, with zero failures, errors, or skips and no reported warnings. See [full test output](pin-tests-full.txt), [JUnit results](pin-tests-full.xml), and [the source/validation summary](pin-change-validation.json). The 42 new cases comprise the 41 focused initialization cases and the relocated-clone integration case.

Mutations were made only on temporary copies. Tests include untouched evidence, fresh training, optional pins, malformed/missing records, committed-spec verification even after rehashing a changed embedded spec, and compatibility with the existing command/path checks.

## Remaining gates

The next engineering work is finite-value validation for CLI/config/spec inputs, followed by correcting unequal-batch loss reporting. Weighted-loss/gradient equivalence and the fixed-panel pilot remain later stages. The large confirmation grid is still provisional.

No commit, push, professor message, meeting booking, or activity-log submission was performed. The earlier [RNG execution record](second-review.md) and [initial audit](README.md) remain point-in-time evidence; this document records the approved follow-up.
