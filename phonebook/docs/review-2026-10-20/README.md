# October 20 repository review: evidence and limits

This is the initial audit snapshot. See [the second review and execution record](second-review.md) for the later plan corrections, implemented RNG fix, and its separate 308-test validation.

The subsequent approved [initialization-audit fix](initialization-audit-fix.md) closes the pin/path and missing-spec-path gaps, adds training-text context checks, and passes the expanded 350-test suite. These follow-up records supersede the corresponding open findings in this initial snapshot.

The [numeric-validation and loss-reporting gate](finite-loss-fix.md) rejects nonfinite inputs and corrects unequal-batch loss reporting while preserving the optimizer updates. Its expanded suite passed **445 tests**; historical audits passed and all 1,337 original artifacts checked remained unchanged.

The [weighted-objective gate](weighted-objective-review.md) passed **81 float64 objective comparisons**, including every parameter gradient and analytic L2-SP derivative checks. Its expanded suite passed **498 tests**, and 49 source-only checks ran without historical checkpoint loading. [The derivation](../weighted-objective-analysis.md) explains token weighting versus fixed book mass. Existing trainer/evaluator/auditor sources and all 1,337 original artifacts remained unchanged in that gate.

The latest [membership/measurement gate](replay-membership-review.md) adds externally pinned replay/probe selection, prepared-text validation, five evaluation partitions, and readable/fresh-inference auditing. **581 tests pass**, including 83 new self-contained cases. Six partition views of the existing Phase-A checkpoint match legacy whole-book scores; these execute no training and explicitly leave training execution unverified. All 59 pre-existing Python files and 1,337 original artifacts remain unchanged. Weighted trainer/runner integration and binary/command/initialization linkage are next, before the pilot.


The subsequent [partial-replay execution and review](partial-replay-execution.md) completes the weighted development runner and six-condition pilot, with raw partition records, exact work, source/initialization receipts, isolated Taylor diagnostics, and deterministic optimizer replay. At 300 updates, partial replay retains only its replayed old entries in this development instance; both positive controls learn A and B completely. The [professor packet](../professor-discussion-packet.md) prepares the next scientific decision. This follow-up continued after midnight on October21; no confirmation data or runs were generated.

Reviewed revision: `8b4ccf34ca6351f7914596e8138da3a358cce1db`. The original working tree was clean. Review outputs were prepared in a temporary directory before adding this documentation. Production code, input books, checkpoints, and historical results were not changed.

The [October 12–20 plan](../week-2026-10-12-to-20.md) contains the research recommendation, retrospective schedule, prioritized findings, theory, experiment design, and professor-meeting draft.

## Coverage

[file-inventory.csv](file-inventory.csv) records every working-directory file at scan time except Git internals: 1,446 files, 133,363,377 bytes, including all 643 tracked files. It includes ignored data, checkpoints, logs, plots, and caches. [inventory-summary.json](inventory-summary.json) records totals and check errors; there were no parsing/integrity errors.

[source-review.csv](source-review.csv) lists every one of the 52 Python files, with its review role and relevant follow-up. All 7,657 source/test lines were included in the source review. The ten existing Markdown files were read for protocols, claims, corrections, and course context. Structured evidence was parsed and reconciled through the audit/report pipelines. Text files and logs were read by the inventory process; a full text read or hash is not a semantic inspection of every log line. PNGs were checked for file integrity, not all individually viewed. Bytecode was hashed only. This review makes no claim of proving that every file is bug-free.

## Executed checks

| Artifact | Scope and result |
|---|---|
| [test-results.txt](test-results.txt), [test-results.xml](test-results.xml) | `python -m pytest -q`: **302 passed in 256.87s**, no skipped cases or reported warnings |
| [experiment-audits.json](experiment-audits.json) | Seven experiment folders, readable mode and full-local mode: **all passed**; 159 training records, 3,180 freshly regenerated answers |
| [historical-audits.json](historical-audits.json) | Three October 4 folders: eight training records, twenty evaluations, 400 saved answers; readable reconciliation plus local artifact hashes passed; the old auditor does not perform fresh inference |
| [derived-results-check.json](derived-results-check.json) | All twelve committed derived CSV/JSON reports match outputs rebuilt into a temporary directory through the report functions |
| [review-probes.json](review-probes.json) | Additional targeted checks reproduced shared evaluation/training RNG, nonfinite parameter acceptance, unequal-batch loss averaging, and the initialization-pin gap in readable auditing |
| [e3-resource-accounting.csv](e3-resource-accounting.csv) | Twenty existing E3 ratio/lr cells, with sequence and supervised-token work to median sustained joint recall |
| [e3-resource-accounting.png](e3-resource-accounting.png) | Visually inspected plot comparing update ratios and sequence-processing ratios against k=1 |

Environment: Python 3.12.10, PyTorch 2.12.0+cpu, pytest 9.1.0; `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MPLBACKEND=Agg` for the full suite. GPU availability was false. Audit and report reconstruction ran with one CPU compute thread as well. Original experiment environment records remain in their original protocols.

Type checks covered 225 JSON files, 11 JSONL files, 250 CSV files, 212 loadable finite checkpoints (6,148 tensors), 46 bijective vocabulary dictionaries, 92 uint16 token binaries, 241 PNGs, and one XML file. These counts describe the initial repository and generated caches at inventory time; this review package is not recursively included in its own inventory.

## Additional probes: what failed and why it matters

1. **RNG isolation:** iterating a non-shuffled evaluation DataLoader changed the next shuffled training permutation, with seed and first batch held fixed. The saved arrays make the difference explicit. Historical d1 already demonstrates trajectory effects. Dedicated training/evaluation generators and a final-tensor regression are required for future timing claims.
2. **Nonfinite values:** spec validation accepted NaN and Infinity for lr and lambda. `train.py --l2sp-lambda nan --adapt-from unused.pth` also passed argument parsing; `lambda > 0` then evaluated false. This probe did not train on an invalid spec or alter any historical run.
3. **Loss denominator:** a three-token fixture split into batches of two and one reports loss 5.0000456, whereas evaluating the concatenated tokens gives 3.3333790. Token accuracy remains correctly counted at 2/3. This matters when extending beyond the current one-batch comparisons.
4. **Initialization pins:** on a disposable copy of `e1_prefix_check`, replacing the protocol initialization checkpoint hash with 64 zeros did not fail the readable audit, even with Git checking enabled and the unchanged committed spec containing the original pin. Full-local mode checks the actual hash, and the original full-local evidence passes. Default mode should also reconcile the protocol's recorded pins with the committed spec.

The issues were reproduced outside the repository's historical evidence directories. A passing suite does not cover them. No production fixes are claimed; the main plan specifies acceptance tests for implementing them.

## Reproduction

From the repository root, run the full suite and each October 7 audit:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
python -m pytest -q
$experimentRoots = Get-ChildItem -LiteralPath phonebook/experiments/week_2026-10-07 -Directory
foreach ($experimentRoot in $experimentRoots) {
    if (Test-Path -LiteralPath (Join-Path $experimentRoot.FullName 'protocol.json')) {
        python phonebook/audit_experiment.py --root $experimentRoot.FullName --require-local-artifacts
        if ($LASTEXITCODE -ne 0) { throw "Audit failed: $($experimentRoot.Name)" }
    }
}
python phonebook/experiments/week_2026-09-30/audit_results.py --require-local-artifacts
```

For report reconstruction, use the commands in the original experiment README with a **new output directory**, then compare CSV rows and parsed JSON to `phonebook/results/`. Plots are not expected to be byte-identical across plotting-library versions.

Full-local mode requires the ignored weights and prepared artifacts and uses recorded evaluation paths. The readable mode can run without those artifacts but provides weaker assurances. The original history is pinned to the reviewed commit; the documentation additions in this package do not constitute a new research-code revision or a pre-registration of executed experiments.

## Dates and provenance

The local Git history available during review has no commits from October 12 through October 20. The latest of the 167 manifest records under `phonebook/experiments/` is `2026-10-11T17:49:32.661374+00:00`, or October 11 at 1:49:32 p.m. America/New_York. Those experiments must not be relabeled as work newly performed October 12–20.

The user may have relevant work or meetings elsewhere; this review cannot infer them. The proposed experiments, professor message, meeting, and activity-log submission have not been carried out by this review. Only this review's checks and artifacts are newly completed.

The repository was confirmed public through the GitHub connector during this review. GitHub's `main` resolves to `7bef2eb4fb2abe026af98da8d7d716381d6c9770`; local `main` is one documentation commit ahead at the reviewed `8b4ccf3`. Professor-facing links in the plan use the confirmed published revision. No push was performed. Primary literature references were checked on the publisher/arXiv pages linked in the plan. Course-site deadlines and requirements were not confirmed: the attempted public-page reads were unavailable, so no current deadline is asserted here.
