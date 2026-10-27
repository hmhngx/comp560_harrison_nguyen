# Replay weighting: objective, gradients, and limits

Prepared October 20, 2026. This is analysis of the repository's existing completion objective, not a new research result or a claim of novelty. The accompanying numerical gate is recorded in [the October 20 review](review-2026-10-20/weighted-objective-review.md).

## 1. The denominator determines the objective

For record `i`, let `m_it` be 1 on a supervised target and 0 on a prompt or padded target. Let `ell_it(theta)` be its next-token cross-entropy. Define:

`s_i(theta) = sum_t m_it * ell_it(theta)`

`t_i = sum_t m_it`

For fixed, nonnegative row weights `w_i`, with positive total supervised weight, the token-weighted objective is:

`L_w(theta) = sum_i w_i * s_i(theta) / Z`, where `Z = sum_i w_i * t_i`.

The weights multiply **token sums**, not equally weighted record means. `Z` excludes masked prompt/padding positions. Normalizing by the number of rows, the sum of row weights, or all padded positions generally changes the objective. Multiplying every weight by the same positive constant leaves `L_w` unchanged in exact arithmetic.

## 2. Literal duplication is a special case

When `w_i` is a nonnegative integer, it represents that many literal copies of the record. Every copy contributes the same token losses and parameter derivatives, provided the model computes each row independently with the same context and deterministic behavior. Thus literal full-batch duplication and weighted unique rows have the same scalar objective in exact arithmetic.

Let `S_A = sum_(i in A) s_i`, `T_A = sum_(i in A) t_i`, and `L_A = S_A/T_A`; define B similarly. A once plus B repeated `k` times has:

`L_dup = (S_A + k*S_B)/(T_A + k*T_B)`

`      = (T_A*L_A + k*T_B*L_B)/(T_A + k*T_B)`.

Only when `T_A = T_B` does this reduce to `(L_A + k*L_B)/(1+k)`. In the original ten-entry phonebooks, every answer has 13 supervised targets including newline, so both totals are 130. Unequal answer lengths or a smaller replay subset break that simplification.

Because weights and masks are fixed and independent of model parameters:

`grad L_w = sum_i w_i * grad s_i / Z`.

This proves the gradient identity in exact arithmetic under the stated assumptions. The numerical check separately compares **every coordinate of every parameter gradient**, rather than relying on an aggregate norm or cosine that can hide errors.

## 3. Fixed book mass is a different control

The planned memory-size comparison holds the old-book objective mass fixed:

`L_balanced = alpha*L_A_subset + (1-alpha)*L_B`.

For nonempty books with positive supervised totals, assign each A-subset row weight `alpha/T_A_subset` and each B row weight `(1-alpha)/T_B`. The weighted-token denominator becomes `alpha + (1-alpha) = 1`, giving the required objective. Endpoint alpha=0 or alpha=1 excludes one book's contribution.

At alpha=0.5, with `m` old records and ten B records, all having 13 targets, the row-weight ratio A:B is `10/m : 1`. For m=1,2,5 this is 10:1, 5:1, and 2:1. Giving every distinct row weight 1 would instead assign A only `m/(m+10)` of the token objective. That would confound memory size with old-data objective mass.

The fixed book-mass objective can be calculated once on each unique record. Its objective denominator is a mathematical normalization, **not** a count of physical training work. Log actual processed sequences, supervised tokens, padded input tokens, and optimizer updates separately. Repeated or heavily weighted rows are not new independent facts or independent experimental units.

## 4. L2-SP and optimizer steps

The repository uses:

`J(theta) = L_task(theta) + lambda * sum_p ||theta_p - anchor_p||^2`.

The penalty is added **once**, outside the normalized task loss. Its derivative for each parameter is `2*lambda*(theta_p-anchor_p)`. Dividing the penalty by the number of records or weighted tokens would change its effective strength. The gate compares task gradients independently, then compares the combined objective, and checks the penalty derivative analytically. A shared penalty must not conceal an incorrect task loss.

If full-batch gradients, current parameters, optimizer state, clipping, learning rate, and decay are identical, a deterministic optimizer computes the same next update in exact arithmetic. This statement does not equate a full-batch update with multiple sequential minibatch steps: parameters and Adam moments change between those steps. It also does not equate different update budgets, because AdamW applies decay on every optimizer step.

## 5. Numerical and scientific limits

- The current transformer has zero dropout and no batch-normalization coupling. With independently sampled dropout masks, repeated copies need not have identical realized losses/gradients. Models that couple rows require a separate argument.
- Use the same vocabulary, targets, supervised mask, context, and causal attention behavior in both computations. Re-inferring a shorter context for a replay subset can invalidate checkpoint compatibility.
- Floating-point summation order and batch-shape kernels can differ. The gate uses CPU float64 with declared coordinatewise tolerances `abs(error) <= 1e-10 + 1e-9*abs(reference)`. It does not promise bit-identical float32, GPU, or long AdamW trajectories. Existing float32 checkpoint/initialization tensors are promoted to float64 for the numerical check; they were not originally trained in float64.
- Finite test cases support implementation correctness; they are not a proof about every possible input. The algebra above supplies the exact-arithmetic argument under its assumptions.
- Passing establishes neither retention of unreplayed facts nor novelty. Those require the fixed-panel pilot, explicit data exclusion, controls, replication, and scientific review described in [the retrospective plan](week-2026-10-12-to-20.md).

## 6. Subsequent production integration and float32 observation

The verified helper now lives in `completion_core/objectives.py` and is imported by both the numerical checker and the separate `phonebook/run_partial_replay.py` development runner. Legacy unweighted training keeps its existing path. The [execution report](review-2026-10-20/partial-replay-execution.md) records the six-condition pilot, independent float32 book-mean/AdamW checks, and actual-update Taylor diagnostics. Small float32 gradient differences produced a maximum one-step parameter gap of 0.000283432 between separate-book and weighted-batch calculations in the checked production cases; the maximum checked training-logit gap was 0.000024319. This confirms that the exact-arithmetic identity must not be promoted to bit-identical optimizer trajectories. All 174 production-model parameter-gradient comparisons and both analytic first-step optimizer checks pass within their declared tolerances.
