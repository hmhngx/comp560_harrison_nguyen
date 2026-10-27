# Next-step decision — October 21, 2026

**Decision:** prepare a shareable, reviewed version of the current evidence and bring the professor a more precise question about **failed free recall versus surviving name–number discrimination**. Resolve the scientific question and related-work overlap before locking a new training matrix. The existing six-condition development result remains the primary evidence; the 90-run confirmation design remains provisional.

## What the recheck changed

The accepted 614-test/source/evidence record still matches all 13 recorded source/evidence hashes. The prior exact-recall statement is correct: at 300 updates, every partial replay condition gets 0/3 fixed probe completions. It does **not** establish that every trace of the old associations is absent.

The same saved evaluations show different teacher-forced behavior on the identical three probe entries:

| Condition | Free exact recall | Probe CE, lower is better | Teacher tokens correct |
|---|---:|---:|---:|
| B only | 0/3 | 2.577618 | 15/39 |
| Partial m=1 | 0/3 | 1.933922 | 21/39 |
| Partial m=2 | 0/3 | 1.735541 | 23/39 |
| Partial m=5 | 0/3 | 1.575871 | 27/39 |

These averages are descriptive. The third probe, Giayn Cranesen, has **worse** CE under m=5 than m=1 (2.651125 versus 1.776426), despite the improved mean. Teacher scoring supplies the correct previous digits, and its 39-token denominator includes punctuation/newlines; 27/39 is not a count of successfully remembered phone numbers. The predeclared exact-recall hypothesis still fails in development. These are post-hoc observations, not a replacement primary endpoint.

## Bounded read-only diagnostic completed

[The diagnostic script](review-2026-10-21/analyze_probe_specificity.py) scores the same three names against all **20 unique A+B phone numbers**, including terminal newline. It uses existing checkpoints only: Phase A at update0 and all six development conditions at update300. It performs **zero training updates**. Both the candidate set and the wrong-name control were specified in the script before this new scoring call, after observing the teacher-score pattern; this is an exploratory analysis, not a preregistered confirmation.

| Model/checkpoint | Correct-target rank for Anra | Camra | Giayn | Rank first, out of 3 |
|---|---:|---:|---:|---:|
| Phase A reference | 1 | 1 | 1 | 3/3 |
| B only | 3 | 4 | 4 | 0/3 |
| Partial m=1 | 2 | 3 | 4 | 0/3 |
| Partial m=2 | 2 | 2 | 4 | 0/3 |
| Partial m=5 | 1 | 1 | 4 | 2/3 |
| Full replay | 1 | 1 | 1 | 3/3 |
| Fresh joint | 1 | 1 | 1 | 3/3 |

Thus m=5 ranks two correct targets first **within this supplied candidate list** while generating all three incorrectly without that list. The restricted list gives strong additional information. Even the two top-ranked correct numbers have very low absolute sequence probabilities (approximately 3.1e-7 and 6.2e-6). Ranking first in a small list is compatible with most probability mass lying outside that list. This does not show successful open generation, recovery from an unconstrained search, relearning savings, complete knowledge preservation, or erasure. There is one dataset, seed42, one subset draw, and three probes. These are not independent replications or evidence for a population claim.

The script also compares each correct number's log probability under its own name with the log mean probability under the other two probe names. This supplies a limited name-conditioning check; all values and all 420 candidate scores are retained in [the diagnostic JSON](review-2026-10-21/probe-specificity-exploration/diagnostic.json) and [21 per-probe rows](review-2026-10-21/probe-specificity-exploration/per-probe.csv). Its control names are only the other two probes. Positive controls and the Phase-A reference rank all targets first. The numerical tie rule is explicit.

Validation: [10 focused tests passed](review-2026-10-21/probe-analysis-tests.txt), covering hand-computed ranks/margins/wrong-name controls, near ties, candidate permutations and invalid matrices. Every correct-target candidate CE reconciles with its independently saved teacher score within the declared CPU float32 tolerance. The sealed 337-file pilot artifacts are checked before and after scoring. The historical 614-test baseline is unchanged; this turn adds the separate diagnostic and its ten focused tests.

## Priorities and exit conditions

1. **Make the evidence inspectable.** Prepare a reviewed commit/PR containing the implementation, readable raw/derived evidence, clear reproduction commands, and this exploratory addendum. Current new work is local/uncommitted. Keep the original 300-update endpoint and all negative outcomes visible. Full-local verification requires the ignored weights/binaries/meta; describe that requirement rather than implying a source-only clone contains them. Existing public repository: [comp560_harrison_nguyen](https://github.com/hmhngx/comp560_harrison_nguyen).
2. **Meet the professor with one concrete question.** “Does increasing a fixed-weight replay memory preserve discrimination of old associations excluded from adaptation, even after free autoregressive recall fails—and what would make that a worthwhile contribution?” Bring the exact-recall table, the per-entry heterogeneity, the candidate-set limitation, and the existing local Taylor residual. Ask whether the strongest contribution would be controlled replication, a retrieval/decoding investigation, or a different theoretical question. No professor endorsement or meeting is claimed here.
3. **After that scientific decision, lock a new protocol before new training.** Choose independent books and an adequate probe panel, define the experimental unit, retain free exact recall as an explicitly declared outcome, and predeclare any candidate-scoring/decoding/relearning diagnostics and all failure rules. A relearning claim requires a fresh-learning comparator with the same architecture and matched optimizer, data and work; its different starting weights must be documented. A recovery procedure receives additional information and must be accounted for. The exploratory three-probe pattern cannot justify a threshold, a winning memory size, or an automatically expanded grid.

The immediate next engineering task is reviewable repository packaging. The immediate next scientific task is the professor discussion and a focused related-work comparison. Additional optimizer sweeps, regularizer implementations, and the provisional 90-run matrix have lower value until those decisions are made.

## Related-work check and novelty limits

A targeted search of primary sources was conducted for this reassessment; this is not an exhaustive literature review. [Li et al., 2024, *Revisiting Catastrophic Forgetting in Large Language Model Tuning*](https://arxiv.org/abs/2406.04836) already studies forgetting in relation to loss-landscape flatness. Adding a curvature diagnostic alone is therefore not evidence of novelty.

[Xu et al., 2026, *Unlearning Isn’t Deletion: Investigating Reversibility of Machine Unlearning in LLMs*](https://proceedings.mlr.press/v306/xu26cd.html) examines apparent forgetting and subsequent recovery in **machine unlearning**. That setting differs from this sequential phonebook adaptation, but it makes recovery/erasure distinctions relevant prior work. It does not validate this toy result or establish that this proposed question is new. The professor should help identify a contribution beyond reproducing a known distinction.

## Updated meeting wording

> My repository is https://github.com/hmhngx/comp560_harrison_nguyen. In a controlled development study, partial replay retained its rehearsed entries but got zero of three unreplayed probe numbers exactly right. A subsequent exploratory check found that the five-entry memory still ranked the correct number highest among twenty supplied candidates for two probes, while free generation failed. The effect is heterogeneous and not independently confirmed. Could we meet to discuss whether this recall-versus-discrimination gap is worth investigating, which related work it must be compared with, and what theoretical or controlled experimental contribution would be convincing?

This is a draft, not a sent message. The additional diagnostic does not change the requested October12–20 retrospective or the original study's primary endpoint.
