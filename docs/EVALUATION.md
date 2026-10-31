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
| H1 | R1 (Rocchio) and R2 (kNN) beat R0 (cold start) | **supported** (+0.138 and +0.161 like rate, 5 seeds x 1,000 swipes, 2026-10-31) — but both find only 2 of 3 planted interests |
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

---

## Study run 2026-10-31 21:34 UTC

```
python -m visualtaste.jobs.evaluate --rungs r0,r1,r2 --seeds 5 --swipes 1000
```

Pool: 50,000 selected images. Rungs: `r0`, `r1`, `r2`. Retrained every 20 labels, five cards per batch.

Simulated user: `sigma(10 (cos - offset))` with the offset calibrated per seed to a 50% baseline like rate (offsets used: 0.648, 0.668, 0.674, 0.675, 0.677; the brief's constant 0.25 saturates on this corpus -- see `backend/visualtaste/eval/sim.py`).

### Simulator study (2026-10-31 21:34 UTC)

**These are simulated results.** 5 seeds x 1000 swipes per rung, against a planted-taste user with three hidden interest centroids. The simulator's truth is cosine distance to those centroids, which is close to the inductive bias of Rocchio and of logistic regression and nothing like a person -- so it flatters the simpler rungs and is near-blind to what a neural ranker would be for. Read `backend/visualtaste/eval/sim.py` before drawing conclusions.

Every number is mean ± sd across seeds.

| rung | like rate | last 100 | batch diversity | coverage | novelty | interests found |
| --- | --- | --- | --- | --- | --- | --- |
| `r0` | 0.454 ± 0.015 | 0.460 ± 0.059 | 0.482 ± 0.002 | 1.000 ± 0.000 | 0.261 ± 0.002 | 1.00 ± 0.00 |
| `r1` | 0.592 ± 0.100 | 0.630 ± 0.089 | 0.241 ± 0.013 | 0.400 ± 0.021 | 0.116 ± 0.014 | 0.67 ± 0.00 |
| `r2` | 0.614 ± 0.082 | 0.688 ± 0.084 | 0.225 ± 0.021 | 0.453 ± 0.031 | 0.103 ± 0.009 | 0.67 ± 0.00 |

**Rolling like rate** (window 50, left to right over the session):

- `r0      ` ▁▄▅▅▅▄▄▅▅▄▄▅▅▃▄▅▄▄▄▄▄▃▄▅▅▅▅▅▅▄▄▄▄▃▃▃▄▅▅▅  0.00 -> 0.50
- `r1      ` ▁▅▆▇▆▇▇▆▇▇▇▇▇▆▇▇▇▇▇▇▆▆▆▆▆▇▆▆▇▆▇▆▆▅▆▆▆▇▇▇  0.00 -> 0.76
- `r2      ` ▁▅▅▆▅▅▆▇▇▇▇▇▇▆▇▇▆▇▇▆▆▅▆▆▆▇▆▆▇▆▆▆▆▅▅▆▅▆▆▆  0.00 -> 0.64

**Scoring cost** (`model.score` over the whole candidate pool, mean of 10 runs, after the session's labels). The `/feed/next` budget is a 50 ms p95 for the entire request, so a rung over roughly 40 ms here cannot be served as it stands.

| rung | labels | candidates | mean | max |
| --- | --- | --- | --- | --- |
| `r0` | 1000 | 50,000 | 13.4 ms | 19.2 ms |
| `r1` | 1000 | 50,000 | 7.3 ms | 7.8 ms |
| `r2` | 1000 | 50,000 | 608.8 ms | 655.3 ms |

---

## Study run 2026-10-31 21:42 UTC

```
python -m visualtaste.jobs.evaluate --rungs r0,r1,r2 --seeds 5 --swipes 1000
```

Pool: 50,000 selected images. Rungs: `r0`, `r1`, `r2`. Retrained every 20 labels, five cards per batch.

Simulated user: `sigma(10 (cos - offset))` with the offset calibrated per seed to a 50% baseline like rate (offsets used: 0.648, 0.668, 0.674, 0.675, 0.677; the brief's constant 0.25 saturates on this corpus -- see `backend/visualtaste/eval/sim.py`).

### Simulator study (2026-10-31 21:42 UTC)

**These are simulated results.** 5 seeds x 1000 swipes per rung, against a planted-taste user with three hidden interest centroids. The simulator's truth is cosine distance to those centroids, which is close to the inductive bias of Rocchio and of logistic regression and nothing like a person -- so it flatters the simpler rungs and is near-blind to what a neural ranker would be for. Read `backend/visualtaste/eval/sim.py` before drawing conclusions.

Every number is mean ± sd across seeds.

| rung | like rate | last 100 | batch diversity | coverage | novelty | interests found |
| --- | --- | --- | --- | --- | --- | --- |
| `r0` | 0.454 ± 0.015 | 0.460 ± 0.059 | 0.482 ± 0.002 | 1.000 ± 0.000 | 0.261 ± 0.002 | 1.00 ± 0.00 |
| `r1` | 0.592 ± 0.100 | 0.630 ± 0.089 | 0.241 ± 0.013 | 0.400 ± 0.021 | 0.116 ± 0.014 | 0.67 ± 0.00 |
| `r2` | 0.615 ± 0.084 | 0.690 ± 0.087 | 0.225 ± 0.020 | 0.450 ± 0.030 | 0.103 ± 0.009 | 0.67 ± 0.00 |

**Rolling like rate** (window 50, left to right over the session):

- `r0      ` ▁▄▅▅▅▄▄▅▅▄▄▅▅▃▄▅▄▄▄▄▄▃▄▅▅▅▅▅▅▄▄▄▄▃▃▃▄▅▅▅  0.00 -> 0.50
- `r1      ` ▁▅▆▇▆▇▇▆▇▇▇▇▇▆▇▇▇▇▇▇▆▆▆▆▆▇▆▆▇▆▇▆▆▅▆▆▆▇▇▇  0.00 -> 0.76
- `r2      ` ▁▅▅▆▅▅▆▇▇▇▇▇▇▆▇▇▆▇▇▆▆▅▆▆▆▇▆▆▇▆▆▆▆▅▅▆▅▆▆▆  0.00 -> 0.64

**Scoring cost** (`model.score` over the whole candidate pool, mean of 10 runs, after the session's labels). The `/feed/next` budget is a 50 ms p95 for the entire request, so a rung over roughly 40 ms here cannot be served as it stands.

| rung | labels | candidates | mean | max |
| --- | --- | --- | --- | --- |
| `r0` | 1000 | 50,000 | 13.6 ms | 18.6 ms |
| `r1` | 1000 | 50,000 | 7.2 ms | 8.5 ms |
| `r2` | 1000 | 50,000 | 271.8 ms | 300.1 ms |

---

## Phase 8 analysis: R0 vs R1 vs R2

Run on 2026-10-31 (the second study section above, which uses the FAISS
implementation of R2; the first used a dense NumPy one and produced the same
like rates, differing only in scoring cost). 5 seeds x 1,000 swipes, 50,000
selected images, offset calibrated per seed.

### Per-seed like rates

| seed | `r0` | `r1` | `r2` |
| --- | --- | --- | --- |
| 1729 | 0.474 | 0.740 | 0.694 |
| 1730 | 0.430 | 0.541 | 0.500 |
| 1731 | 0.464 | 0.439 | 0.551 |
| 1732 | 0.453 | 0.622 | 0.722 |
| 1733 | 0.450 | 0.619 | 0.607 |

### Paired comparison

Paired across seeds, since every rung faced the same five planted users.
`eval/metrics.paired_bootstrap` (90% interval, 10,000 resamples) and
`eval/metrics.sign_test` (two-sided):

| comparison | mean difference | 90% interval | seeds won | sign test |
| --- | --- | --- | --- | --- |
| `r1` − `r0` | +0.138 | +0.068 to +0.208 | 4 of 5 | p = 0.375 |
| `r2` − `r0` | +0.161 | +0.107 to +0.217 | 5 of 5 | p = 0.062 |
| `r2` − `r1` | +0.023 | −0.031 to +0.077 | 2 of 5 | p = 1.000 |

### H1 — "R1 and R2 beat R0"

**Supported, with a caveat about how strongly.** Both rungs raise the like rate
by about 14 to 16 points over cold start, and both bootstrap intervals exclude
zero. The sign test does not reach significance at five seeds — it cannot:
with five paired samples the smallest two-sided p available is 0.062, so a
clean sweep is the best outcome possible and `r2` achieved it. R1 lost to R0 on
seed 1731 (0.439 vs 0.464), which is a real loss and not noise to be waved
away: a single averaged direction can land between two of the three planted
interests and do worse than spreading out.

**R2 vs R1 is not distinguishable here.** The interval spans zero and R2 won
only two of five seeds. The two-separate-interests argument for kNN
(`tests/test_r1_r2.py::TestKnnRecommender`) is real in principle, but this
simulator plants three *well-separated* centroids and the batch is only five
cards, so R1's single direction is rarely forced to compromise badly enough for
the difference to show. A study designed to separate them would plant two
nearby interests and one far away.

### What the other columns say, and why they matter more than they look

| | `r0` | `r1` | `r2` |
| --- | --- | --- | --- |
| cluster coverage | 1.000 | 0.400 | 0.450 |
| novelty | 0.261 | 0.116 | 0.103 |
| planted interests found | 3 of 3 | 2 of 3 | 2 of 3 |

Both learned rungs found **two of the three** planted interests, every seed,
while cold start found all three. They buy their higher like rate by narrowing:
coverage drops from the whole collection to about 40% of it, and novelty more
than halves.

That is the clearest result in this study, and it is the argument for R4. A
model that locks onto the interests it found first will never discover the
third, and its like rate looks *better* for having given up — nothing in the
like rate penalises never having tried. Exploration (§7.5: Thompson sampling,
MMR, and the one-in-five quota) exists precisely to pay the small like-rate
cost of looking.

### Scoring cost is a result, not an implementation detail

| rung | labels | mean | verdict against the 50 ms p95 |
| --- | --- | --- | --- |
| `r0` | 1,000 | 13.6 ms | fine |
| `r1` | 1,000 | 7.2 ms | fine — constant in the label count |
| `r2` | 1,000 | 271.8 ms | **over budget by 5x** |

R1's cost does not grow with labels: it averages them into one vector and
scores with one dot product. R2's does, because it compares every candidate
against every label — so it meets the budget up to roughly 100 labels and
misses it from there on, worsening as Alen swipes. FAISS halves the constant
(`docs/BENCHMARKS.md`) but does not change the shape.

So on this evidence R2 is a teaching rung rather than a serving one. R3 scores
on 128 PCA dimensions with a single weight vector, which is constant in labels
and about a millisecond; that is the rung to serve at scale, and Phase 9 should
be read with this measurement in mind rather than as a free choice.

### Honest limitations of all of the above

- The simulator's truth is cosine to three centroids, which is nearly R1's own
  hypothesis class. R1's result here is close to an upper bound on how well it
  could ever do, and says nothing about whether it will suit Alen.
- The offset was calibrated to make the comparison possible at all
  (`docs/DECISIONS.md` #12). Without it every rung scored 0.88 and the study
  was vacuous — that run is reproducible with `--brief-offset`.
- Five seeds is the brief's number and it is few. Every interval above is wide
  enough that a sixth seed could move the means by several points.
- No real human has rated a single image yet.
