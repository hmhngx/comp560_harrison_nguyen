# Approved numeric-validation and loss-reporting gates — October 20, 2026

## Scope and decision

The approved next step was to reject invalid numeric training inputs, then repair epoch loss reporting. Both changes are implemented locally. This record follows the [initialization-audit repair](initialization-audit-fix.md) and [RNG repair](second-review.md); it does not change the dates or conclusions of the saved research experiments.

The next justified engineering step is the full-batch weighted-loss and parameter-gradient equivalence check described below. The partial-replay development pilot and large confirmation grid remain unexecuted.

## Numeric inputs

- Training validates the effective configuration **after** Python config loading and CLI overrides, before creating its output directory. Learning rate must be finite and positive; L2-SP lambda, weight decay, and gradient clipping must be finite and nonnegative. Zero optional penalties/clipping remain accepted.
- Dimensions, layer/head counts, batch size, epochs, and logging/accuracy intervals must be positive integers, excluding booleans. Embedding width must be divisible by head count. Seeds must be integers in PyTorch's supported inclusive range, `[-2**63, 2**64 - 1]`.
- Declarative experiment specs validate finite learning rates/lambdas and supported seeds. They also reject nonfinite values in arbitrary nested metadata before planning reads inputs or creates experiment outputs.
- Spec loading rejects the nonstandard constants `NaN`, `Infinity`, and `-Infinity`, plus valid numeric syntax that overflows a Python float, such as `1e999`. Canonical hashing, JSON report writing, and runner manifest serialization use `allow_nan=False`. Serialization finishes before the report writer opens an existing output file, so rejected objects do not truncate it. Finite JSON encoding remains unchanged.
- The legacy learning-rate sweep validates learning rates, lambdas, epochs, timeout, and seed range before creating its output root. Its manifest serialization also rejects nonfinite values.
- [Shared checks](../../../completion_core/validation.py) keep these numeric rules consistent without importing PyTorch into the legacy sweep.

These checks validate declared input values; finite hyperparameters do not guarantee stable optimization. The runner still executes the model config through the child trainer, where effective config validation occurs. It does not preflight arbitrary Python config execution before creating its own protocol/prepared-data directory. Reconciliation of all effective config fields with protocol metadata remains a separate runner improvement.

## Loss definition and compatibility

Both [epoch helpers](../../../completion_core/training.py) now report:

`sum(batch_mean_task_CE * supervised_tokens_in_batch) / sum(supervised_tokens_in_batch)`

The denominator excludes targets masked with `-100`. Validation and teacher-forced evaluation use a fixed model. Training loss is an **online** mean measured before each batch update, with a different model at successive batches; it is not a post-epoch evaluation of the final model. L2-SP is excluded from the reported task loss.

The optimizer still backpropagates the original per-batch mean task loss plus the original L2-SP penalty. Reporting weights do not rescale gradients, clipping, AdamW decay, or optimizer steps. The tests compare exact final tensors, gradients, and optimizer state with an independent reference loop under SGD and AdamW, with and without L2-SP.

The helpers explicitly require unweighted mean `CrossEntropyLoss(ignore_index=-100)`, matching every current caller. Other reductions, class weights, or ignore indices need a defined denominator and fail explicitly. Empty loaders and batches with no supervised targets raise `ValueError` instead of returning undefined means or applying a NaN update. This check applies as each batch is encountered; it does not roll back successful earlier updates in a streaming epoch.

Historical phonebook conditions use one training batch per epoch, and historical checkpoint evaluations fit each book into one batch. Their saved artifacts remain unchanged and re-audit successfully. Older multi-batch curves can differ under the corrected reporting rule even when weights are identical. The [root README](../../../README.md) now explains this distinction and requires the recorded code revision for historical metric reproduction.

## Tests and evidence

- [Initial pre-fix output](finite-loss-initial-before.txt): 71 failures and 24 passes, including 60 failing numeric-validation cases. This first loss fixture had equal token totals across its two training batches; review corrected that accidental equality before implementation.
- [Corrected pre-fix loss tests](loss-tests-before.txt): **16 failed, 1 passed**. The fixture has three records with 2, 2, and 3 supervised targets. Batch sizes 1 and 2 expose the wrong epoch mean; batch size 3 is the one-batch control. The hand-calculated dataset loss is `log(1 + exp(-10)) + 20/7`.
- [Focused output](finite-loss-tests-focused.txt) and [JUnit results](finite-loss-tests-focused.xml): **138 passed in 6.76 seconds**, covering the 95 new cases and the existing L2-SP/sweep regressions. Tests exercise direct CLI and Python config validation, preservation of existing CSV/JSON files, strict nested JSON, large integers, valid boundary seeds, variable target lengths, a short final batch, unchanged accuracy counts, optimizer equivalence, and undefined-loss handling.
- [Historical validation](finite-loss-historical-validation.json): all seven experiment roots pass both readable and full-local audits, with **159 training records**, **3,180 regenerated answers**, and **1,337 original artifacts byte-identical** to the initial inventory. These are validation reruns of historical evidence, not new research results.
- [Full-suite output](finite-loss-tests-full.txt) and [JUnit results](finite-loss-tests-full.xml): **445 passed in 214.86 seconds**, with zero failures, errors, or skips and no reported warnings. This is the earlier 350-test suite plus 95 new cases. See [the source and validation summary](finite-loss-change-validation.json) for the exact reviewed hashes. The initialization auditor and its regression tests remain unchanged from the preceding gate.

Tests run on Python 3.12.10 and PyTorch 2.12.0+cpu with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `MPLBACKEND=Agg`. CPU checks do not establish equivalence across devices or library versions.

## Next execution decision

Before adding weighted replay to training, verify the mathematical full-batch control in float64 on the actual transformer: literal duplication of B versus weighted unique A+B records must agree in scalar task loss and **every parameter gradient**. Include unequal supervised lengths, masked targets, and several repetition ratios. Use token sums and `T_A + k*T_B`, not equal weights on book means when token totals differ. Keep the model initialization, context, and L2-SP objective identical. Failure of any comparison blocks the weighted implementation.

Passing this check would justify implementation and a small development pilot; it would not prove identical long AdamW trajectories, establish novelty, or justify starting the provisional confirmation grid. The fixed three-entry unreplayed A panel and separate update/work-budget analyses remain in the [revised plan](../week-2026-10-12-to-20.md).

No research pilot, commit, push, professor message, meeting booking, or activity-log submission was performed in this gate.
