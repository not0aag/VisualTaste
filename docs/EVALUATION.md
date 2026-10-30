# Evaluation

**Read this first: every number in this file comes from a simulated user, not
from Alen.**

No real labels can exist until he starts swiping. So each rung of the
recommendation ladder is measured against the planted-taste simulator
(`backend/visualtaste/eval/sim.py`): a fake user with three hidden interest
centroids and a known `p(like | image)`, driving the real feed code path on a
profile flagged `is_simulated = 1`.

What that buys us, and what it does not:

- **It does** prove the machinery works end to end -- that the feed serves, the
  labels land, the models train, the metrics compute, and that a rung which
  should learn a planted taste actually recovers it.
- **It does not** tell us which rung will serve Alen best. A simulator whose
  truth is "cosine to one of three centroids" is close to the inductive bias of
  Rocchio and logistic regression, and nothing like a person. Expect the
  simulator to flatter the simple rungs and to understate what a neural ranker
  could do with real, messier labels.

Results are filled in from Phase 7 onwards. Figures live in `docs/figures/`.

---

## Hypotheses to be tested

The brief's §7.7 names five, to be reported honestly whether or not they hold:

| # | Hypothesis | Status |
| --- | --- | --- |
| H1 | R1 (Rocchio) and R2 (kNN) beat R0 (cold start) | not yet run |
| H2 | R3 (logistic regression) beats R2 after roughly 150 labels | not yet run |
| H3 | R4 (Thompson sampling) finds all 3 planted interests faster than R3 | not yet run |
| H4 | R3+R5 (multi-interest features) beats R3 | not yet run |
| H5 | R6 (neural ranker) beats R3 only once labels are plentiful | not yet run |

---

## Results by rung

_Filled in from Phase 8 onwards: like-rate curve, AUC and log loss via
time-ordered replay, recall of the planted interests, intra-batch diversity,
cluster coverage._

| Rung | Labels | Like rate | AUC | Log loss | NDCG@10 | Interests recalled |
| --- | --- | --- | --- | --- | --- | --- |
| _(pending Phase 8)_ | | | | | | |

---

## Latency

_Filled in at Phase 14: `/feed/next` p95, target < 50 ms for R1-R5._
