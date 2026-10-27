# Second plan review and first execution step — October 20, 2026

This records the first execution decision and its checks. The subsequently approved [initialization-audit follow-up](initialization-audit-fix.md) has completed step 2 below and passed the expanded 350-test suite. The remaining-order discussion here records the earlier decision point.

## Decision

Execute the training/evaluation RNG separation first. A change in measurement frequency must not change a treatment's batch order or learned weights. This defect affects every subsequent timing comparison and can be fixed and tested without choosing a new research direction.

The 90-run confirmation matrix is **not ready to execute**. The next scientific stage is a development pilot of at most six conditions, after the remaining provenance and measurement gates. Professor feedback should help choose the final research question; routine correctness repairs do not require waiting for that meeting.

## Problems found in the first plan

1. **Changing evaluation population.** In the original design, increasing memory from one to five records changed which A records counted as unreplayed. Differences could reflect record difficulty as well as protection. The revised plan reserves a fixed three-entry probe panel for every partial-replay/B-only comparison. All ten A records remain available for descriptive partitions. Full replay and joint training intentionally include the probe records and are positive controls, not evidence about never-replayed facts.
2. **Unequal optimization exposure.** A sequence budget gives smaller memories more optimizer updates, more cumulative learning-rate exposure, and more AdamW decay applications. The revised primary comparison uses 300 updates with fixed optimizer settings. The sequence-budget comparison remains a separate operational question, with actual work and remainder reported.
3. **Premature scale.** Nine starting points and two subset draws do not justify implementing and running the entire grid before validating the measurement. The revised first research pilot has six conditions on the development data: B-only, partial replay of one/two/five records, full replay, and fresh joint training. Its purpose is feasibility and debugging. It does not establish novelty or independent confirmation.
4. **Weak hypothesis and resolution.** High token accuracy does not define a falsifiable memory hypothesis. The replacement is a directional comparison on the same probe records at a fixed update count, with an explicit B-learning criterion. Three probe entries provide only 33-percentage-point resolution. No precise critical memory size can follow from this design.
5. **Overstated theoretical language.** L2-SP penalizes distance; it is not a hard constraint. The local stationary-point calculation and gradient diagnostics remain approximations, with no nonlinear convergence or causal-mechanism guarantee.

These corrections are incorporated in [the revised plan](../week-2026-10-12-to-20.md). The October 12–20 date table remains a retrospective proposal, not a claim that those activities happened on those dates.

## Implemented engineering change

- `train.py` now constructs three separate seeded DataLoader generators by default: training, training-set evaluation, and validation. Evaluation iterator creation advances neither the training generator nor the global model RNG in this deterministic dataset/model setup.
- `--loader-rng isolated` is the default. `--loader-rng legacy` restores the original shared-global-RNG loader construction for historical reproduction. Python config files can select the policy, and CLI values override them. Invalid policies are rejected.
- New checkpoints store the effective policy and seed in `training_config`; the training log prints the policy.
- The root README documents the new default and historical reproduction. New default runs can produce different numbers from the old training pipeline. Reproduction still requires the original accuracy interval, environment, and other settings.
- Legacy mode restores training behavior, not historical checkpoint serialization bytes. Newly saved policy metadata changes checkpoint files; use the original recorded code revision for historical byte/hash reproduction.
- Historical datasets, model files, metrics, figures, manifests, protocols, and evaluations are preserved. No partial-replay study or confirmation matrix was run.

The guarantee covers this loader use with deterministic datasets and the current dropout-free model. It does not promise identical results across PyTorch versions, devices, or arbitrary stochastic dataset/model code. Future stochastic transforms need their own RNG-isolation tests.

## Validation

- Before the fix, the new end-to-end tests failed for both full-batch training and a partial last batch: **2 failed, 1 passed**. The failure was a changed training order when accuracy frequency changed. See [the failing regression output](rng-tests-before.txt).
- After the fix, the focused regression file passed **all 6 tests**. Intervals 1, 5, and 10 have identical batch ordering and every epoch's parameter tensors on the tiny model, for both batch regimes. Short runs match long-run prefixes despite an extra final evaluation. Explicit legacy mode matches the original loader constructors, including every epoch's tensors. Generator isolation, configuration, CLI overrides, and checkpoint policy metadata are also checked. See [focused results](rng-tests-focused.txt).
- The actual 64-dimensional, two-head, two-layer phonebook adaptation at lr=0.001 and L2-SP lambda=0.3 was run for 100 updates at each of intervals 1, 5, and 10. Parameter-state hashes agree at **all 100 updates**. These are engineering checks using the working-tree code, not pre-registered research runs. The [verification record](phonebook-rng-validation.json) includes per-update hashes, environment, and the SHA-256 of `train.py`.
- All seven historical full-local experiment audits still pass: **159 training records and 3,180 freshly regenerated answers**. Every one of **1,337 original data/result/artifact files** checked against the first inventory is byte-identical. See [historical preservation](historical-preservation.json).
- The full suite passed **308 tests in 353.62 seconds**, with zero failures, errors, or skips and no reported warnings. See [full test output](rng-tests-full.txt) and [JUnit results](rng-tests-full.xml). [The validation summary](rng-change-validation.json) records the source hashes and reconciles these results with the real-phonebook and preservation checks.

## Remaining execution order

1. **Completed:** this RNG change's full-suite and real-phonebook verification, plus historical-artifact preservation checks.
2. **Next:** repair readable-audit reconciliation of initialization pins against the committed spec. Add mutation tests before relying on new provenance reports.
3. Reject nonfinite experiment/config values and correct unequal-batch loss reporting before adding weighted/subset training support.
4. Validate literal-duplication versus weighted-unique losses and gradients, then implement the fixed-panel subset pilot with explicit provenance. Lock its protocol before confirmation.
5. Discuss novelty and interpretation with the professor; choose the confirmation size from the scientific question and pilot evidence, not the earlier run-count estimate.

Only step 1 was selected for implementation in this follow-up. The other defects remain open and should not be described as repaired. Changes remain local; no commit, push, message, meeting booking, or submission was performed.
