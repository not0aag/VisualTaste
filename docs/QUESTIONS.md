# Questions asked of the owner

The brief's §3 allows six kinds of interruption and no others. This file records
every one that was used: the question, the default that would have been taken,
and the answer.

---

## Q2 — Storage: no archive drive, internal drive under the 150 GB threshold

**Asked:** 2026-10-30
**Context:** The §6.1 archive-root ladder reached rule 5. No non-system drive was
attached, and `C:` had 135.9 GB free against the 150 GB that rules 3 and 4
require. The full 80k candidate pool needs roughly 100 GB all-in (~82 GB
originals + ~12 GB derivatives + ~6 GB environment and weights).

**Options offered:**

1. Internal archive at `C:\VisualAI\archive`, full 80k pool, with a disk guard
   and a >8 MB original skip (recommended).
2. Internal archive, pool reduced to ~60k candidates (~62 GB originals).
3. Wait for an external drive and auto-detect it.

**Default if unanswered:** option 1, as the only non-blocking choice that keeps
the specified 80k → 50k target.

**Answer:** Option 1 — internal, full 80k pool.

**Consequence:** Implemented as `docs/DECISIONS.md` #2, including both
mitigations and the recorded backup risk.

---

## Q1 — GPU unusable

**Not asked.** `nvidia-smi` reports an RTX 3060 Laptop GPU on driver 576.80, and
after installing torch 2.11.0+cu128 `torch.cuda.is_available()` returned True.
The condition never arose.

---

## Q3 and Q4

**Not asked.** No admin prompt, login wall or physical action has been needed
(Q3), and no data source refused the planned use (Q4) — all four were reachable
and all four finished under a 0.01% failure rate.

---

## Q5 — Safety screen flagged more than 0.5% of candidates

**Asked:** 2026-10-31
**Context:** The screen flagged **789 of 78,986** images (0.999%), above the
0.5% in §3 Q5.

**The important caveat, given with the question:** that trip is mechanical, not
a finding. §7.2 step 7 requires flagging *the top 1% by score regardless of
content*, so the flag rate is always ≥1% and Q5 fires however clean the corpus
is. Measured breakdown:

| | count | share |
| --- | --- | --- |
| Above the calibrated threshold (0.0478) | 676 | 0.856% |
| Flagged by the top-1% rule (by construction) | 789 | 0.999% |

Margins ran from −0.107 to 0.115, with the top-1% cutoff at 0.046 — a narrow
band, so this was "the 789 most unsafe-looking images", not 789 confident
judgements. PD12M is largely museum collections, so many were likely to be
anatomical studies, classical nudes and medical illustration.

**Options offered:** (1) keep them quarantined for manual review in the app
(recommended); (2) auto-remove all 789 (the brief's stated default);
(3) drop the top-1% rule and keep only the calibrated threshold.

**Default if unanswered:** option 2, per the brief.

**Answer:** Option 2 — auto-remove all 789.

**Applied:** each flagged image set to `status='removed'` and recorded in
`removals` with the reason
`safety_screen: auto-removed by owner decision (Q5, 2026-10-31)`. Nothing was
deleted from disk: the originals, derivatives and rows all remain, so the
decision is reversible with a single `UPDATE` if he changes his mind. The
50,000-image selection was unaffected — it had already excluded them.

---

## Q6 — End of build

Pending. Sent once every Phase 1-14 Definition of Done holds, per §12.

