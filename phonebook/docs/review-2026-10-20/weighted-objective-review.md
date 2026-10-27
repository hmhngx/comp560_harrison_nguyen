# Approved weighted-objective gate — October 20, 2026

## Decision and scope

**The float64 objective/gradient gate passes.** Literal full-batch duplication and weighted unique rows agree on the actual phonebook transformer within the tolerances declared before execution. Fixed book-mass objectives also agree with independently computed book means when weights include the supervised-token correction.

This gate adds [an analysis tool](../../verify_weighted_objective.py), [regression tests](../../../tests/test_weighted_objective.py), and [the mathematical derivation](../weighted-objective-analysis.md). It does **not** add a weighted training CLI, subset protocol, or partial-replay experiment. Existing training, evaluation, and auditing implementations remain unchanged from the preceding gate.

The next justified implementation is immutable subset membership and partitioned evaluation, followed by integrating the checked objective into a versioned training protocol. The six-condition development pilot remains behind those checks. The provisional confirmation grid remains unexecuted.

## What was checked

The declared tolerances are coordinatewise: `abs(weighted-reference) <= 1e-10 + 1e-9*abs(reference)`. All computations use CPU float64. The original float32 checkpoint and random initializations are promoted to float64; this does not imply that they were trained or originally initialized in float64.

| Axis | Coverage |
|---|---|
| Model | The repository's `CompletionTransformer`: 64 embedding dimensions, 2 heads, 2 layers, context 29, vocabulary 67; zero dropout |
| Starting state | Historical Phase-A seed42 checkpoint, random seed42, random seed123 |
| Data fixtures | Original books; unequal target lengths and book sizes; two A records versus all ten B records |
| Literal duplication | One A copy and k B copies, k=0,1,2,3,5,9 |
| Fixed book mass | A fractions 0.25, 0.5, 0.75, using row weights alpha/T_A and (1-alpha)/T_B |
| Regularization | Task loss separately, task plus lambda=0.3 L2-SP, and the analytic penalty derivative for every parameter |

The three fixture token totals `(T_A,T_B)` are `(130,130)`, `(29,23)`, and `(26,130)`. These deliberately prevent an equal-token-only test from accepting the wrong book-mean formula. The fixture encoder is separately checked against the actual `SequenceDataset` mask and padding behavior.

Three starting states × three fixtures × nine objectives produce **81 comparisons**. Each compares task and regularized scalar losses and all coordinates of **29 named parameter tensors**. The saved evidence contains **4,698 gradient-tensor comparisons** plus **2,349 analytic L2-SP derivative checks**. These are numerical checks, not independent research samples or training runs. No optimizer is constructed or stepped by the verifier.

## Numerical result

| Quantity | Largest observed absolute difference |
|---|---:|
| Task/regularized scalar loss | 1.7763568394002505e-15 |
| Task parameter gradient | 2.7755575615628914e-16 |
| Parameter gradient including L2-SP | 2.7755575615628914e-16 |
| Penalty derivative versus `2*lambda*(theta-anchor)` | 1.6696713456276768e-17 |

Every coordinate passes. The largest gradient error uses about `1.79e-6` of its allowed coordinatewise tolerance. Raw case definitions, input/source hashes, environment, and all tensor comparison statistics are preserved in [verification.json](weighted-objective-evidence/verification.json), [the parameter CSV](weighted-objective-evidence/parameter-comparisons.csv), and [the compact summary](weighted-objective-evidence/summary.json). [The standalone execution log](weighted-matrix-output.txt) records all nine state/fixture groups.

The full-batch identity depends on deterministic, batch-independent row computation and fixed weights/masks. It does not equate multiple minibatch optimizer steps, guarantee long float32/AdamW trajectories, establish retention, or establish research novelty. The [derivation](../weighted-objective-analysis.md) explains the assumptions and the separate book-balanced objective.

## Review and regression evidence

- [Initial tests](weighted-tests-initial.txt): 30 passed and 10 failed. Eight failures identified missing required analytic-penalty coverage in the first version of the verifier. Two exposed actual verifier defects: a tiny negative float64 weight became negative zero during float32 conversion, and NaN failure scalars could not be saved with strict JSON. Original-dtype validation and explicit null failure diagnostics now close these cases. These were defects in the new analysis tool, not newly discovered historical training failures.
- [Focused tests](weighted-tests-focused.txt) and [JUnit results](weighted-tests-focused.xml): **50 passed in 60.58 seconds**, including the complete matrix. Three additional provenance regressions then passed, with 50 deselected; see [their output](weighted-tests-provenance.txt).
- Three deliberately wrong objectives—ignoring weights, dividing by row mass, and substituting book means for unequal token totals—are rejected in both scalar and gradient checks, despite the shared L2-SP penalty. A single wrong coordinate also fails; shape broadcasting is rejected.
- Hand-calculated CE and logit gradients, finite-difference gradient checking, zero-weight exclusion, masked-gradient zeros, global weight scaling, aligned row permutation, and dataset encoder parity pass. Invalid/nonfinite weights, underflow-hidden negative weights, denominator overflow, unsupported dtypes/shapes/devices, and undefined supervised mass fail explicitly.
- Tests verify unchanged input model tensors, existing gradients, model mode, and global CPU RNG. The complete matrix also preserves the caller's CPU RNG. Nonzero attention dropout is rejected.
- The analysis checks initialization pins against the unchanged spec at the recorded Git commit, captures source/input hashes before computation, rechecks those bytes and Git HEAD afterward, and records the dirty working tree explicitly. Tests reject a substituted checkpoint digest, a different committed declaration, and a virtual source change during execution. These are consistency checks, not independent attestation of historical training.
- The writer rejects existing output directories, rejects nonfinite metadata before output creation, preserves failed comparisons in strict JSON, and exits nonzero on a failed gate. Mutations use temporary outputs or mocked reads; original files are not edited.
- [Source-only checks](weighted-tests-source-only.txt): **49 passed, 4 deselected in 7.26 seconds**, with `torch.load` replaced by a function that fails if called. These checks do not load historical checkpoints.
- [Full-suite output](weighted-tests-full.txt) and [JUnit results](weighted-tests-full.xml): **498 passed in 566.28 seconds**, with zero failures, errors, or skips and no reported warnings. This is the preceding 445-test suite plus 53 new cases. [The final validation summary](weighted-change-validation.json) records exact source/evidence hashes and parsed suite counts. All new source and tests were reviewed; syntax parsing and `git diff --check` pass.

## Preservation and reproducibility

[Preservation checks](weighted-objective-preservation.json) confirm that all **10 Python files** recorded by the preceding gate remain byte-identical, as do **1,337 original data/result/artifact files** from the initial inventory. The preceding [historical audit](finite-loss-historical-validation.json) remains the latest fresh 159-record / 3,180-answer audit. It was not rerun solely for this separate analysis tool; its training, evaluation, and auditor dependencies are unchanged.

Run source-only checks after installing the repository's dependencies:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
python -m pytest tests/test_weighted_objective.py -q -m "not integration"
```

The full 81-case verifier additionally requires the historical ignored Phase-A checkpoint and vocabulary whose pins are recorded in the spec. Run it only into an unused output folder:

```powershell
python phonebook/verify_weighted_objective.py --out-root NEW_OUTPUT_DIRECTORY
```

The provided verification JSON/CSV can be inspected without those binaries. A fresh checkout's source-only checks use the actual transformer architecture with synthetic targets, rather than claiming to reproduce the historical checkpoint state. These new files and the revised plan are still local additions until published.

## Next execution boundary

Implement a versioned subset/weight declaration that records exact replay IDs and a disjoint fixed probe panel before training. Reject missing IDs, duplicate/conflicting prompts, overlaps, incompatible vocabulary/context, and incorrect partition counts. Report A-replayed, A-unreplayed, A-all, fixed A probe, and B separately; whole-book controls have no unreplayed complement. Then integrate the checked token-weighted objective, preserving the old unweighted path, and validate float32 behavior and a tiny end-to-end workflow before the bounded research pilot.

No partial-replay pilot, confirmation run, commit, push, professor message, meeting booking, or submission was performed in this gate.
