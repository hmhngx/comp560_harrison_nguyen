# Professor discussion packet — prepared October 21, 2026

Purpose: use the professor's invitation to choose a stronger research question before scaling the experiments. This is meeting preparation; no meeting or message delivery is claimed.

## Evidence to bring

Repository: [comp560_harrison_nguyen](https://github.com/hmhngx/comp560_harrison_nguyen). Historical results are described in [RESULTS.md](../RESULTS.md). The new development work is local and uncommitted.

The [October 12–20 retrospective](week-2026-10-12-to-20.md) separates prior October10–11 experiments from the October20 review. The new [development report](review-2026-10-20/partial-replay-execution.md) records the six-condition study and subsequent October21 verification without backdating it.

The controlled question was whether more distinct replayed old facts protect **old facts excluded from adaptation**, with old/new CE mass held at 50/50. A fixed three-fact A probe was excluded from every partial memory; nested memories used m=1,2,5. The original books and seed42 were development data.

At 300 updates, all partial memories learn B10/10 and retain their replayed A facts, but recall zero of the unreplayed A facts, including probe0/3. Full replay and fresh joint both reach A10/10 and B10/10. The separate sequence-budget endpoints are preserved too. The proposed m=5-over-m=1 probe advantage was not observed. One dataset, one draw, and three probe facts do not establish a population result or a capacity limit.

The derivation establishes full-batch duplication as token weighting under stated assumptions. Float64 loss/every-gradient checks and float32 independent optimizer checks support the implementation. Isolated actual-update diagnostics show that a first-order account can miss much of an initial probe-CE increase; a second-order approximation also has a substantial residual. This supplies a precise follow-up question, not an established mechanism.

## Decisions to seek

1. Is the negative unreplayed-fact result scientifically interesting enough to replicate on independent books, or should the main question change before investing in the tentative 90-run matrix?
2. Would a comparison of arbitrary and structured associations, or curvature-aware protection measured under an explicit storage/work budget, be a stronger direction? What closely related work must be ruled out before claiming novelty?
3. What scale and independent dataset/probe counts would make the question convincing? The current three-probe design resolves only 33-percentage-point changes. Agree on outcomes, uncertainty, and stop rules before confirmation.

Existing replay and importance-regularization papers are background in [the plan](week-2026-10-12-to-20.md), not an exhaustive novelty review. Implementing another established regularizer alone would not justify a contribution claim.

## Draft meeting request

> Thank you for your feedback. My repository is https://github.com/hmhngx/comp560_harrison_nguyen. I have now separated objective weighting, distinct replay memory, and training work in a small phonebook development study. Partial replay retained the rehearsed entries but did not preserve the fixed unreplayed probe in this instance; full replay and fresh joint controls learned both books. I also checked a local gradient/curvature explanation against the actual AdamW updates, including where its approximation falls short. Could we meet to decide whether to replicate this controlled negative result or pursue a more novel direction involving structured associations or parameter sensitivity? I would particularly value your advice on related work and what theoretical claim would be worth testing.

## After the discussion

Record actual feedback and the chosen question. Revise the design if necessary, then lock a clean committed implementation/spec, independently generated books, panels, subset draws, seeds, budgets, metrics, and failure handling before confirmation. Preserve all development outcomes. Do not count checkpoints or subset draws as independent datasets, tune on the new confirmation books, or claim that the professor endorsed a direction before the conversation occurs.
