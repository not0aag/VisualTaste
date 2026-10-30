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

## Q3, Q4, Q5

**Not asked yet.** No admin prompt, login wall or physical action has been
needed (Q3); no data source has refused the planned use (Q4); the safety screen
runs in Phase 5, so its flag rate is not yet known (Q5).

---

## Q6 — End of build

Pending. Sent once every Phase 1-14 Definition of Done holds, per §12.
