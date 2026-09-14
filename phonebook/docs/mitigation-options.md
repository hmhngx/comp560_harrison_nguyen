# Candidate forgetting-mitigation techniques

Prep notes for the 2026-09-16 lab meeting. Three options, evaluated
specifically against this codebase's actual constraints (tiny models,
CPU-only, `train.py --adapt-from` fine-tuning) rather than in the abstract.

**Why any of this is needed:** [`phonebook/sweep_lr/README.md`](../sweep_lr/README.md)
shows that tuning the learning rate alone gets you to at best 50% exact
retention of phonebook A (`lr=0.0001`, 100 epochs) — every other tested
point does worse, and none reach full retention. LR-tuning has a ceiling;
that's the concrete evidence for trying an actual mitigation next.

## 1. L2-SP regularization

**What:** penalize fine-tuned weights for drifting from the Phase-A
checkpoint, adding λ·Σᵢ‖θᵢ − θᵢ_old‖² to the loss, where θ_old is a frozen
snapshot taken right after `--adapt-from` loads the Phase-A weights.

**Source:** Li et al., 2018. Already has a full working code sketch in this
repo at [`phonebook/docs/L2-SP-regularization-chat.md`](L2-SP-regularization-chat.md).

**Implementation cost here — Low.** `train.py`'s training loop is a flat,
simple loop (no framework abstraction to fight). Needs: (a) snapshot
`model.state_dict()` immediately after `--adapt-from` loads it, before any
Phase-B gradient step; (b) add the penalty term to the loss inside
`completion_core/training.py`'s `run_epoch`; (c) expose λ as a new CLI
flag, same pattern as `--seed` (added this week). Realistic first draft:
1–2 hours.

**Known limitation:** penalizes parameter drift generically — doesn't target
specific facts, so it may slow forgetting without fully preventing it.

## 2. Elastic Weight Consolidation (EWC)

**What:** like L2-SP, but the penalty is weighted per-parameter by an
estimated "importance" for Phase-A (the diagonal of the Fisher information
matrix, computed from Phase-A data), so parameters that matter more for A
are protected more.

**Source:** Kirkpatrick et al., 2017 — general ML background, **not**
something already in this repo. The phonebook README names it only in
passing ("consult the instructor... replay methods... analogous to sleep
consolidation"); there's no code sketch to build from, unlike L2-SP.

**Implementation cost here — Medium-High.** Requires an extra pass over
Phase-A data computing per-parameter squared gradients (the Fisher
approximation) before Phase-B training starts, storing that alongside the
weight snapshot, then using it to weight the penalty term. More moving
parts than L2-SP, nothing in-repo to sanity-check the implementation
against.

**Upside:** theoretically more targeted than L2-SP; could preserve model
capacity for future learning better since it isn't punishing every
parameter equally.

## 3. Replay / rehearsal (data mixing)

**What:** mix repeated copies of phonebook A into Phase-B training (the
README's "repeat phase-B data k times" idea, applied to A instead) so the
model keeps seeing A while learning B.

**Source:** general continual-learning technique; explicitly proposed in
[`phonebook/README.md`](../README.md)'s "Next steps."

**Implementation cost here — Low.** Pure data-assembly change — no
`train.py` or model code touched at all. Build a training file that
concatenates *k* copies of `phonebookA.txt` with `phonebookB.txt` before
running `prepare_phonebook.py`.

**Known limitation:** interacts with something this week's sweep just
surfaced — Phase-B currently has exactly one batch per epoch
(10 sequences, `batch_size=64`). Adding repeated A data changes the
training-set size and therefore the batch count per epoch, which changes
the *number of gradient steps per epoch*, not just their content. Worth a
small pilot run before trusting results at face value, rather than assuming
this is a clean, single-variable change.

## Recommendation to bring to the meeting

**L2-SP first.** Lowest implementation risk (working code sketch already
exists, small diff to `train.py`), a clean well-motivated result achievable
in one work session, and it pairs naturally with replay as a cheap second
follow-up — giving a regularization-vs-rehearsal comparison (and their
combination) rather than one isolated technique. EWC is the stretch goal
once the first two produce results, since it needs infrastructure
(Fisher-info computation) the other two don't require.
