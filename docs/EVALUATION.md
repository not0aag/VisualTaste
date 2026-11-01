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
| H2 | R3 (logistic regression) beats R2 after roughly 150 labels | **not supported for ranking** (replay, 20 seeds, 2026-11-01): R3 has lower log loss at every cut from 400 labels (significant at 9 of 11), but R2 has higher AUC at every cut from 150 on (significant at 9 of 16) |
| H3 | R4 (Thompson sampling) finds all 3 planted interests faster than R3 | **supported for discovery, not for like rate** (20 seeds × 1,000 swipes per scenario, 2026-11-01): R4 finds all interests in more seeds (+0.25 S1, +0.55 S5, +0.85 S6) and sooner (−288, −266 and −773 swipes), with +50 to +58 points of coverage; but it keeps −0.21, −0.31 and −0.34 of R3's like-rate gain over R0, against a 70% bar |
| H4 | R3+R5 (multi-interest features) beats R3 | **inconclusive with exploration** (R4+R5 vs R4, 20 seeds on S1, S5, S6, 2026-11-01): no metric differs; the plain R3+R5 vs R3 comparison is part of `phase14_final` |
| H5 | R6 (neural ranker) beats R3 only once labels are plentiful | **not supported** (replay, 10 seeds × 5,000 swipes on S1 and S4, BCE and BCE+BPR, 2026-11-01): no AUC win at any of 500 / 1,500 / 3,000 / 4,900 labels (worse at 3,000 in both scenarios); one log-loss win (S4, BCE, 4,900 labels, −0.008). A neural ranker did not earn its cost at this scale |

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

---

## Phase 12 training logs: R4 as the logging policy, S1 and S4, 5,000 swipes (2026-11-01 04:18 UTC)

```
python -m visualtaste.jobs.evaluate --study phase12_logs --seeds 10 --swipes 5000
```

**SIMULATED.** Variants and their calibration: `backend/visualtaste/eval/sim.py`. Every number is mean ± sd across seeds; comparisons are paired by seed (Part 2 §4.1). An interest counts as found at 5 likes.

### S1_semantic

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r4` | 10 | 0.564 ± 0.016 | 3.00 ± 0.00 | 1.00 ± 0.00 | 58 ± 16 | -- | -- | 0.87 ± 0.04 | 0.505 ± 0.010 | 0.195 ± 0.010 | -- | 0.698 ± 0.014 | -- | 0.027 ± 0.007 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r4` | 30 | 0 of 30 |

### S4_conjunctive

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r4` | 10 | 0.484 ± 0.012 | 3.00 ± 0.00 | 1.00 ± 0.00 | 53 ± 15 | -- | -- | 0.94 ± 0.02 | 0.525 ± 0.008 | 0.225 ± 0.007 | -- | 0.643 ± 0.010 | -- | 0.020 ± 0.008 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r4` | 28 | 0 of 30 |

### Comparisons

| variant | comparison | metric | mean diff | 90% CI | sign test (W-L) | losing seeds | seeds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

---

## R0-R3 on S1, S2, S3 and S5 (2026-11-01 04:31 UTC)

```
python -m visualtaste.jobs.evaluate --study phase9_rungs --seeds 20 --swipes 1000
```

**SIMULATED.** Variants and their calibration: `backend/visualtaste/eval/sim.py`. Every number is mean ± sd across seeds; comparisons are paired by seed (Part 2 §4.1). An interest counts as found at 5 likes.

### S1_semantic

| spec | seeds | like rate | found | all found | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r0` | 20 | 0.467 ± 0.014 | 3.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.484 ± 0.004 | 0.261 ± 0.002 | -- | 0.655 ± 0.021 | -- | -- |
| `r1` | 20 | 0.557 ± 0.094 | 1.60 ± 0.66 | 0.10 ± 0.30 | 0.14 ± 0.05 | 0.237 ± 0.019 | 0.114 ± 0.016 | 0.501 ± 0.023 | 0.582 ± 0.028 | 0.688 ± 0.027 | -- |
| `r2` | 20 | 0.550 ± 0.076 | 1.95 ± 0.67 | 0.20 ± 0.40 | 0.17 ± 0.06 | 0.222 ± 0.021 | 0.100 ± 0.012 | 0.493 ± 0.021 | 0.602 ± 0.043 | 0.692 ± 0.018 | -- |
| `r3` | 20 | 0.622 ± 0.045 | 2.75 ± 0.43 | 0.75 ± 0.43 | 0.39 ± 0.11 | 0.318 ± 0.019 | 0.167 ± 0.016 | 0.510 ± 0.027 | 0.649 ± 0.026 | 0.653 ± 0.029 | 0.059 ± 0.021 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r0` | 33 | 0 of 60 |
| `r1` | 21 | 28 of 60 |
| `r2` | 25 | 21 of 60 |
| `r3` | 30 | 5 of 60 |

### S2_appearance

| spec | seeds | like rate | found | all found | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r0` | 20 | 0.482 ± 0.018 | 3.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.484 ± 0.004 | 0.261 ± 0.002 | -- | 0.613 ± 0.015 | -- | -- |
| `r1` | 20 | 0.576 ± 0.101 | 2.30 ± 0.78 | 0.50 ± 0.50 | 0.14 ± 0.05 | 0.239 ± 0.020 | 0.117 ± 0.013 | 0.505 ± 0.028 | 0.634 ± 0.053 | 0.680 ± 0.038 | -- |
| `r2` | 20 | 0.547 ± 0.108 | 2.50 ± 0.67 | 0.60 ± 0.49 | 0.16 ± 0.05 | 0.228 ± 0.025 | 0.106 ± 0.012 | 0.504 ± 0.029 | 0.626 ± 0.070 | 0.687 ± 0.033 | -- |
| `r3` | 20 | 0.675 ± 0.040 | 3.00 ± 0.00 | 1.00 ± 0.00 | 0.43 ± 0.16 | 0.334 ± 0.029 | 0.179 ± 0.019 | 0.511 ± 0.030 | 0.732 ± 0.024 | 0.606 ± 0.045 | 0.046 ± 0.018 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r0` | 34 | 0 of 60 |
| `r1` | 25 | 14 of 60 |
| `r2` | 26 | 10 of 60 |
| `r3` | 34 | 0 of 60 |

### S3_colour

| spec | seeds | like rate | found | all found | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r0` | 20 | 0.436 ± 0.014 | 2.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.484 ± 0.004 | 0.261 ± 0.002 | -- | 0.670 ± 0.014 | -- | -- |
| `r1` | 20 | 0.472 ± 0.058 | 2.00 ± 0.00 | 1.00 ± 0.00 | 0.14 ± 0.06 | 0.237 ± 0.021 | 0.115 ± 0.015 | 0.504 ± 0.023 | 0.643 ± 0.039 | 0.701 ± 0.016 | -- |
| `r2` | 20 | 0.474 ± 0.043 | 1.95 ± 0.22 | 0.95 ± 0.22 | 0.17 ± 0.07 | 0.223 ± 0.026 | 0.106 ± 0.009 | 0.493 ± 0.022 | 0.651 ± 0.033 | 0.705 ± 0.013 | -- |
| `r3` | 20 | 0.511 ± 0.034 | 2.00 ± 0.00 | 1.00 ± 0.00 | 0.67 ± 0.13 | 0.387 ± 0.039 | 0.214 ± 0.027 | 0.507 ± 0.021 | 0.648 ± 0.027 | 0.703 ± 0.013 | 0.051 ± 0.017 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r0` | 22 | 0 of 40 |
| `r1` | 22 | 0 of 40 |
| `r2` | 23 | 1 of 40 |
| `r3` | 22 | 0 of 40 |

### S5_drift

| spec | seeds | like rate | found | all found | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r0` | 20 | 0.448 ± 0.022 | 3.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.484 ± 0.004 | 0.261 ± 0.002 | -- | 0.672 ± 0.021 | -- | -- |
| `r1` | 20 | 0.528 ± 0.071 | 1.85 ± 0.65 | 0.15 ± 0.36 | 0.14 ± 0.05 | 0.236 ± 0.019 | 0.114 ± 0.017 | 0.500 ± 0.021 | 0.636 ± 0.055 | 0.692 ± 0.020 | -- |
| `r2` | 20 | 0.535 ± 0.071 | 1.70 ± 0.64 | 0.10 ± 0.30 | 0.17 ± 0.06 | 0.225 ± 0.021 | 0.101 ± 0.011 | 0.502 ± 0.021 | 0.656 ± 0.047 | 0.687 ± 0.015 | -- |
| `r3` | 20 | 0.592 ± 0.059 | 2.40 ± 0.58 | 0.45 ± 0.50 | 0.47 ± 0.18 | 0.319 ± 0.023 | 0.166 ± 0.015 | 0.504 ± 0.024 | 0.680 ± 0.038 | 0.677 ± 0.046 | 0.065 ± 0.025 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r0` | 64 | 0 of 60 |
| `r1` | 508 | 23 of 60 |
| `r2` | 488 | 26 of 60 |
| `r3` | 48 | 12 of 60 |

### Comparisons

| variant | comparison | metric | mean diff | 90% CI | sign test (W-L) | losing seeds | seeds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_semantic | `r3` vs `r1` | like_rate | +0.0652 | [+0.0298, +0.1014] | 14-5 (p = 0.0636) | 1736, 1740, 1741, 1742, 1745 | 20 | **supported** |
| S1_semantic | `r3` vs `r1` | interests_found | +1.1500 | [+0.9000, +1.4000] | 17-0 (p = 1.53e-05) | none | 20 | **supported** |
| S1_semantic | `r3` vs `r1` | replay_auc | +0.0091 | [-0.0052, +0.0232] | 13-7 (p = 0.263) | 1730, 1734, 1738, 1740, 1742, 1746, 1747 | 20 | **inconclusive** |
| S1_semantic | `r3` vs `r2` | like_rate | +0.0725 | [+0.0487, +0.0972] | 19-1 (p = 4.01e-05) | 1740 | 20 | **supported** |
| S1_semantic | `r3` vs `r2` | interests_found | +0.8000 | [+0.5000, +1.1000] | 13-1 (p = 0.00183) | 1746 | 20 | **supported** |
| S1_semantic | `r3` vs `r2` | replay_auc | +0.0166 | [+0.0060, +0.0269] | 13-7 (p = 0.263) | 1730, 1734, 1738, 1740, 1742, 1744, 1747 | 20 | **supported** |
| S2_appearance | `r3` vs `r1` | like_rate | +0.0982 | [+0.0663, +0.1304] | 17-3 (p = 0.00258) | 1733, 1735, 1743 | 20 | **supported** |
| S2_appearance | `r3` vs `r1` | interests_found | +0.7000 | [+0.4000, +1.0000] | 10-0 (p = 0.00195) | none | 20 | **supported** |
| S2_appearance | `r3` vs `r1` | replay_auc | +0.0062 | [-0.0069, +0.0194] | 10-10 (p = 1) | 1731, 1732, 1733, 1737, 1738, 1739, 1741, 1742, 1743, 1745 | 20 | **inconclusive** |
| S2_appearance | `r3` vs `r2` | like_rate | +0.1280 | [+0.0939, +0.1614] | 18-2 (p = 0.000402) | 1733, 1735 | 20 | **supported** |
| S2_appearance | `r3` vs `r2` | interests_found | +0.5000 | [+0.2500, +0.7500] | 8-0 (p = 0.00781) | none | 20 | **supported** |
| S2_appearance | `r3` vs `r2` | replay_auc | +0.0069 | [-0.0043, +0.0184] | 12-8 (p = 0.503) | 1731, 1737, 1739, 1740, 1741, 1742, 1744, 1745 | 20 | **inconclusive** |
| S3_colour | `r3` vs `r1` | like_rate | +0.0384 | [+0.0184, +0.0582] | 16-4 (p = 0.0118) | 1729, 1740, 1743, 1744 | 20 | **supported** |
| S3_colour | `r3` vs `r1` | interests_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 20 | **inconclusive** |
| S3_colour | `r3` vs `r1` | replay_auc | +0.0034 | [-0.0087, +0.0158] | 9-11 (p = 0.824) | 1731, 1734, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1744, 1747 | 20 | **inconclusive** |
| S3_colour | `r3` vs `r2` | like_rate | +0.0371 | [+0.0211, +0.0539] | 17-3 (p = 0.00258) | 1739, 1740, 1744 | 20 | **supported** |
| S3_colour | `r3` vs `r2` | interests_found | +0.0500 | [+0.0000, +0.1500] | 1-0 (p = 1) | none | 20 | **inconclusive** |
| S3_colour | `r3` vs `r2` | replay_auc | +0.0140 | [+0.0041, +0.0241] | 14-6 (p = 0.115) | 1729, 1734, 1737, 1738, 1744, 1747 | 20 | **supported** |
| S5_drift | `r3` vs `r1` | like_rate | +0.0642 | [+0.0355, +0.0926] | 15-5 (p = 0.0414) | 1732, 1734, 1739, 1742, 1743 | 20 | **supported** |
| S5_drift | `r3` vs `r1` | interests_found | +0.5500 | [+0.2500, +0.8500] | 13-3 (p = 0.0213) | 1729, 1746, 1747 | 20 | **supported** |
| S5_drift | `r3` vs `r1` | replay_auc | +0.0044 | [-0.0074, +0.0160] | 11-9 (p = 0.824) | 1729, 1730, 1732, 1734, 1738, 1741, 1742, 1746, 1747 | 20 | **inconclusive** |
| S5_drift | `r3` vs `r2` | like_rate | +0.0573 | [+0.0314, +0.0833] | 16-4 (p = 0.0118) | 1732, 1739, 1740, 1742 | 20 | **supported** |
| S5_drift | `r3` vs `r2` | interests_found | +0.7000 | [+0.3500, +1.0000] | 15-2 (p = 0.00235) | 1746, 1747 | 20 | **supported** |
| S5_drift | `r3` vs `r2` | replay_auc | +0.0025 | [-0.0089, +0.0140] | 11-9 (p = 0.824) | 1730, 1734, 1735, 1738, 1740, 1741, 1742, 1744, 1747 | 20 | **inconclusive** |

---

## H3: R4 (Thompson + MMR + quota) vs R3 on S1, S5 and S6 (2026-11-01 04:34 UTC)

```
python -m visualtaste.jobs.evaluate --study phase10_h3 --seeds 20 --swipes 1000
```

**SIMULATED.** Variants and their calibration: `backend/visualtaste/eval/sim.py`. Every number is mean ± sd across seeds; comparisons are paired by seed (Part 2 §4.1). An interest counts as found at 5 likes.

### S1_semantic

| spec | seeds | like rate | found | all found | swipes to find all | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r0` | 20 | 0.467 ± 0.014 | 3.00 ± 0.00 | 1.00 ± 0.00 | 62 ± 22 | 1.00 ± 0.00 | 0.484 ± 0.004 | 0.261 ± 0.002 | -- | 0.655 ± 0.021 | -- | -- |
| `r3` | 20 | 0.622 ± 0.045 | 2.75 ± 0.43 | 0.75 ± 0.43 | 351 ± 396 | 0.39 ± 0.11 | 0.318 ± 0.019 | 0.167 ± 0.016 | 0.510 ± 0.027 | 0.649 ± 0.026 | 0.653 ± 0.029 | 0.059 ± 0.021 |
| `r4` | 20 | 0.433 ± 0.022 | 3.00 ± 0.00 | 1.00 ± 0.00 | 64 ± 24 | 0.97 ± 0.02 | 0.544 ± 0.007 | 0.297 ± 0.006 | 0.588 ± 0.023 | 0.689 ± 0.021 | 0.682 ± 0.008 | 0.036 ± 0.009 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r0` | 33 | 0 of 60 |
| `r3` | 30 | 5 of 60 |
| `r4` | 33 | 0 of 60 |

### S5_drift

| spec | seeds | like rate | found | all found | swipes to find all | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r0` | 20 | 0.448 ± 0.022 | 3.00 ± 0.00 | 1.00 ± 0.00 | 532 ± 22 | 1.00 ± 0.00 | 0.484 ± 0.004 | 0.261 ± 0.002 | -- | 0.672 ± 0.021 | -- | -- |
| `r3` | 20 | 0.592 ± 0.059 | 2.40 ± 0.58 | 0.45 ± 0.50 | 818 ± 210 | 0.47 ± 0.18 | 0.319 ± 0.023 | 0.166 ± 0.015 | 0.504 ± 0.024 | 0.680 ± 0.038 | 0.677 ± 0.046 | 0.065 ± 0.025 |
| `r4` | 20 | 0.403 ± 0.024 | 3.00 ± 0.00 | 1.00 ± 0.00 | 552 ± 31 | 0.97 ± 0.02 | 0.548 ± 0.010 | 0.301 ± 0.008 | 0.583 ± 0.026 | 0.695 ± 0.020 | 0.676 ± 0.010 | 0.045 ± 0.016 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r0` | 64 | 0 of 60 |
| `r3` | 48 | 12 of 60 |
| `r4` | 66 | 0 of 60 |

### S6_six_interests

| spec | seeds | like rate | found | all found | swipes to find all | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r0` | 20 | 0.463 ± 0.012 | 6.00 ± 0.00 | 1.00 ± 0.00 | 130 ± 34 | 1.00 ± 0.00 | 0.484 ± 0.004 | 0.261 ± 0.002 | -- | 0.649 ± 0.021 | -- | -- |
| `r3` | 20 | 0.609 ± 0.039 | 4.80 ± 0.75 | 0.15 ± 0.36 | 933 ± 181 | 0.40 ± 0.12 | 0.299 ± 0.029 | 0.154 ± 0.016 | 0.513 ± 0.031 | 0.646 ± 0.032 | 0.660 ± 0.021 | 0.054 ± 0.019 |
| `r4` | 20 | 0.413 ± 0.023 | 6.00 ± 0.00 | 1.00 ± 0.00 | 161 ± 50 | 0.98 ± 0.02 | 0.548 ± 0.005 | 0.300 ± 0.006 | 0.575 ± 0.017 | 0.679 ± 0.017 | 0.679 ± 0.011 | 0.034 ± 0.012 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r0` | 66 | 0 of 120 |
| `r3` | 66 | 24 of 120 |
| `r4` | 72 | 0 of 120 |

### Comparisons

| variant | comparison | metric | mean diff | 90% CI | sign test (W-L) | losing seeds | seeds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_semantic | `r4` vs `r3` | all_found | +0.2500 | [+0.1000, +0.4000] | 5-0 (p = 0.0625) | none | 20 | **supported** |
| S1_semantic | `r4` vs `r3` | interests_found | +0.2500 | [+0.1000, +0.4000] | 5-0 (p = 0.0625) | none | 20 | **supported** |
| S1_semantic | `r4` vs `r3` | time_to_all_found | -287.5500 | [-438.2600, -148.7500] | 12-5 (p = 0.143) | 1731, 1738, 1739, 1743, 1744 | 20 | **supported** |
| S1_semantic | `r4` vs `r3` | coverage_last_500 | +0.5836 | [+0.5445, +0.6219] | 20-0 (p = 1.91e-06) | none | 20 | **supported** |
| S1_semantic | `r4` vs `r3` | diversity | +0.2263 | [+0.2181, +0.2346] | 20-0 (p = 1.91e-06) | none | 20 | **supported** |
| S1_semantic | `r4` vs `r3` | like_rate | -0.1889 | [-0.2044, -0.1741] | 0-20 (p = 1.91e-06) | 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748 | 20 | **not supported** |
| S5_drift | `r4` vs `r3` | all_found | +0.5500 | [+0.3500, +0.7500] | 11-0 (p = 0.000977) | none | 20 | **supported** |
| S5_drift | `r4` vs `r3` | interests_found | +0.6000 | [+0.4000, +0.8000] | 11-0 (p = 0.000977) | none | 20 | **supported** |
| S5_drift | `r4` vs `r3` | time_to_all_found | -266.4000 | [-343.4500, -186.8000] | 16-4 (p = 0.0118) | 1734, 1738, 1739, 1742 | 20 | **supported** |
| S5_drift | `r4` vs `r3` | coverage_last_500 | +0.5000 | [+0.4352, +0.5641] | 20-0 (p = 1.91e-06) | none | 20 | **supported** |
| S5_drift | `r4` vs `r3` | diversity | +0.2295 | [+0.2188, +0.2398] | 20-0 (p = 1.91e-06) | none | 20 | **supported** |
| S5_drift | `r4` vs `r3` | like_rate | -0.1890 | [-0.2108, -0.1668] | 0-20 (p = 1.91e-06) | 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748 | 20 | **not supported** |
| S6_six_interests | `r4` vs `r3` | all_found | +0.8500 | [+0.7000, +0.9500] | 17-0 (p = 1.53e-05) | none | 20 | **supported** |
| S6_six_interests | `r4` vs `r3` | interests_found | +1.2000 | [+0.9500, +1.5000] | 17-0 (p = 1.53e-05) | none | 20 | **supported** |
| S6_six_interests | `r4` vs `r3` | time_to_all_found | -772.5500 | [-838.2025, -689.5000] | 19-1 (p = 4.01e-05) | 1743 | 20 | **supported** |
| S6_six_interests | `r4` vs `r3` | coverage_last_500 | +0.5789 | [+0.5312, +0.6281] | 20-0 (p = 1.91e-06) | none | 20 | **supported** |
| S6_six_interests | `r4` vs `r3` | diversity | +0.2494 | [+0.2387, +0.2605] | 20-0 (p = 1.91e-06) | none | 20 | **supported** |
| S6_six_interests | `r4` vs `r3` | like_rate | -0.1956 | [-0.2101, -0.1807] | 0-20 (p = 1.91e-06) | 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748 | 20 | **not supported** |

**Like-rate gain kept** (H3 needs R4 to keep >= 70% of R3's gain over R0): (R4 - R0) / (R3 - R0), paired bootstrap 90% CI over seeds.

| variant | R4 gain / R3 gain | 90% CI | meets 70%? |
| --- | --- | --- | --- |
| S1_semantic | -0.21 | [-0.27, -0.16] | no |
| S5_drift | -0.31 | [-0.40, -0.24] | no |
| S6_six_interests | -0.34 | [-0.41, -0.28] | no |

---

## H4: R4+R5 (interests on top of exploration) vs R4 on S1, S5 and S6 (2026-11-01 04:54 UTC)

```
python -m visualtaste.jobs.evaluate --study phase11_h4 --seeds 20 --swipes 1000
```

**SIMULATED.** Variants and their calibration: `backend/visualtaste/eval/sim.py`. Every number is mean ± sd across seeds; comparisons are paired by seed (Part 2 §4.1). An interest counts as found at 5 likes.

### S1_semantic

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r4` | 20 | 0.433 ± 0.022 | 3.00 ± 0.00 | 1.00 ± 0.00 | 64 ± 24 | -- | -- | 0.97 ± 0.02 | 0.544 ± 0.007 | 0.297 ± 0.006 | 0.588 ± 0.023 | 0.689 ± 0.021 | 0.682 ± 0.008 | 0.036 ± 0.009 |
| `r4+r5` | 20 | 0.437 ± 0.029 | 3.00 ± 0.00 | 1.00 ± 0.00 | 64 ± 23 | -- | 1.00 ± 0.00 | 0.96 ± 0.03 | 0.543 ± 0.008 | 0.295 ± 0.010 | 0.595 ± 0.029 | 0.690 ± 0.029 | 0.680 ± 0.015 | 0.035 ± 0.010 |
| `r5` | 20 | 0.616 ± 0.055 | 3.00 ± 0.00 | 1.00 ± 0.00 | 103 ± 192 | -- | 1.00 ± 0.00 | 0.13 ± 0.06 | 0.243 ± 0.017 | 0.126 ± 0.010 | 0.503 ± 0.024 | 0.628 ± 0.033 | 0.674 ± 0.042 | -- |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r4` | 33 | 0 of 60 |
| `r4+r5` | 33 | 0 of 60 |
| `r5` | 33 | 0 of 60 |

### S5_drift

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r4` | 20 | 0.403 ± 0.024 | 3.00 ± 0.00 | 1.00 ± 0.00 | 552 ± 31 | 52 ± 31 | -- | 0.97 ± 0.02 | 0.548 ± 0.010 | 0.301 ± 0.008 | 0.583 ± 0.026 | 0.695 ± 0.020 | 0.676 ± 0.010 | 0.045 ± 0.016 |
| `r4+r5` | 20 | 0.406 ± 0.024 | 3.00 ± 0.00 | 1.00 ± 0.00 | 555 ± 21 | 55 ± 21 | 1.00 ± 0.00 | 0.96 ± 0.02 | 0.548 ± 0.009 | 0.300 ± 0.009 | 0.594 ± 0.024 | 0.695 ± 0.019 | 0.676 ± 0.017 | 0.042 ± 0.013 |
| `r5` | 20 | 0.576 ± 0.076 | 2.75 ± 0.43 | 0.75 ± 0.43 | 713 ± 204 | 203 ± 209 | 1.00 ± 0.00 | 0.14 ± 0.06 | 0.243 ± 0.018 | 0.126 ± 0.010 | 0.492 ± 0.029 | 0.653 ± 0.048 | 0.685 ± 0.042 | -- |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r4` | 66 | 0 of 60 |
| `r4+r5` | 58 | 0 of 60 |
| `r5` | 55 | 5 of 60 |

### S6_six_interests

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r4` | 20 | 0.413 ± 0.023 | 6.00 ± 0.00 | 1.00 ± 0.00 | 161 ± 50 | -- | -- | 0.98 ± 0.02 | 0.548 ± 0.005 | 0.300 ± 0.006 | 0.575 ± 0.017 | 0.679 ± 0.017 | 0.679 ± 0.011 | 0.034 ± 0.012 |
| `r4+r5` | 20 | 0.419 ± 0.018 | 6.00 ± 0.00 | 1.00 ± 0.00 | 169 ± 49 | -- | 1.00 ± 0.00 | 0.97 ± 0.02 | 0.544 ± 0.006 | 0.299 ± 0.005 | 0.578 ± 0.021 | 0.678 ± 0.016 | 0.681 ± 0.009 | 0.036 ± 0.010 |
| `r5` | 20 | 0.587 ± 0.045 | 5.60 ± 0.58 | 0.65 ± 0.48 | 448 ± 415 | -- | 1.00 ± 0.00 | 0.14 ± 0.06 | 0.245 ± 0.023 | 0.129 ± 0.011 | 0.498 ± 0.022 | 0.620 ± 0.031 | 0.692 ± 0.029 | -- |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r4` | 72 | 0 of 120 |
| `r4+r5` | 73 | 0 of 120 |
| `r5` | 64 | 8 of 120 |

### Comparisons

| variant | comparison | metric | mean diff | 90% CI | sign test (W-L) | losing seeds | seeds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_semantic | `r4+r5` vs `r4` | replay_log_loss | -0.0024 | [-0.0074, +0.0027] | 10-10 (p = 1) | 1730, 1731, 1734, 1735, 1738, 1742, 1743, 1744, 1746, 1747 | 20 | **inconclusive** |
| S1_semantic | `r4+r5` vs `r4` | interests_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 20 | **inconclusive** |
| S1_semantic | `r4+r5` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 20 | **inconclusive** |
| S1_semantic | `r4+r5` vs `r4` | like_rate | +0.0035 | [-0.0066, +0.0134] | 12-8 (p = 0.503) | 1730, 1732, 1734, 1735, 1737, 1741, 1743, 1746 | 20 | **inconclusive** |
| S6_six_interests | `r4+r5` vs `r4` | replay_log_loss | +0.0023 | [-0.0007, +0.0054] | 8-12 (p = 0.503) | 1730, 1731, 1732, 1736, 1737, 1738, 1739, 1742, 1743, 1745, 1747, 1748 | 20 | **inconclusive** |
| S6_six_interests | `r4+r5` vs `r4` | interests_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 20 | **inconclusive** |
| S6_six_interests | `r4+r5` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 20 | **inconclusive** |
| S6_six_interests | `r4+r5` vs `r4` | like_rate | +0.0060 | [-0.0001, +0.0123] | 12-8 (p = 0.503) | 1729, 1730, 1733, 1734, 1735, 1741, 1744, 1746 | 20 | **inconclusive** |
| S5_drift | `r4+r5` vs `r4` | drift_recovery | +3.3000 | [-6.3000, +12.4525] | 8-12 (p = 0.503) | 1729, 1730, 1731, 1733, 1734, 1739, 1740, 1742, 1743, 1744, 1746, 1747 | 20 | **inconclusive** |
| S5_drift | `r4+r5` vs `r4` | like_rate | +0.0027 | [-0.0060, +0.0116] | 10-10 (p = 1) | 1729, 1730, 1732, 1733, 1737, 1738, 1739, 1741, 1745, 1746 | 20 | **inconclusive** |

---

## R4 sensitivity: tau_post, lambda_mmr and quota (S1) (2026-11-01 04:56 UTC)

```
python -m visualtaste.jobs.evaluate --study phase10_sensitivity --seeds 10 --swipes 1000
```

**SIMULATED.** Variants and their calibration: `backend/visualtaste/eval/sim.py`. Every number is mean ± sd across seeds; comparisons are paired by seed (Part 2 §4.1). An interest counts as found at 5 likes.

### S1_semantic

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r4` | 10 | 0.440 ± 0.016 | 3.00 ± 0.00 | 1.00 ± 0.00 | 58 ± 16 | -- | -- | 0.97 ± 0.02 | 0.540 ± 0.007 | 0.294 ± 0.006 | 0.602 ± 0.020 | 0.693 ± 0.022 | 0.682 ± 0.005 | 0.034 ± 0.010 |
| `r4:tau=0.1` | 10 | 0.525 ± 0.038 | 3.00 ± 0.00 | 1.00 ± 0.00 | 57 ± 17 | -- | -- | 0.80 ± 0.09 | 0.526 ± 0.012 | 0.243 ± 0.010 | 0.591 ± 0.017 | 0.711 ± 0.027 | 0.674 ± 0.011 | 0.035 ± 0.011 |
| `r4:tau=0.25` | 10 | 0.481 ± 0.032 | 3.00 ± 0.00 | 1.00 ± 0.00 | 63 ± 25 | -- | -- | 0.90 ± 0.06 | 0.536 ± 0.006 | 0.266 ± 0.006 | 0.587 ± 0.018 | 0.695 ± 0.024 | 0.682 ± 0.008 | 0.037 ± 0.010 |
| `r4:tau=0.5` | 10 | 0.462 ± 0.023 | 3.00 ± 0.00 | 1.00 ± 0.00 | 63 ± 21 | -- | -- | 0.94 ± 0.02 | 0.538 ± 0.010 | 0.282 ± 0.011 | 0.589 ± 0.024 | 0.701 ± 0.018 | 0.686 ± 0.008 | 0.040 ± 0.019 |
| `r4:tau=2` | 10 | 0.415 ± 0.020 | 3.00 ± 0.00 | 1.00 ± 0.00 | 66 ± 38 | -- | -- | 0.97 ± 0.02 | 0.554 ± 0.004 | 0.308 ± 0.004 | 0.598 ± 0.014 | 0.692 ± 0.021 | 0.675 ± 0.006 | 0.030 ± 0.006 |
| `r4:mmr=0.5` | 10 | 0.417 ± 0.029 | 3.00 ± 0.00 | 1.00 ± 0.00 | 58 ± 19 | -- | -- | 0.96 ± 0.02 | 0.576 ± 0.006 | 0.308 ± 0.008 | 0.602 ± 0.014 | 0.697 ± 0.022 | 0.676 ± 0.008 | 0.040 ± 0.011 |
| `r4:mmr=0.9` | 10 | 0.450 ± 0.028 | 3.00 ± 0.00 | 1.00 ± 0.00 | 61 ± 19 | -- | -- | 0.96 ± 0.02 | 0.498 ± 0.008 | 0.290 ± 0.007 | 0.566 ± 0.021 | 0.674 ± 0.017 | 0.687 ± 0.007 | 0.036 ± 0.010 |
| `r4:quota=0` | 10 | 0.431 ± 0.024 | 3.00 ± 0.00 | 1.00 ± 0.00 | 72 ± 50 | -- | -- | 0.96 ± 0.02 | 0.548 ± 0.008 | 0.301 ± 0.005 | 0.585 ± 0.014 | 0.694 ± 0.013 | 0.687 ± 0.008 | 0.033 ± 0.013 |
| `r4:quota=10` | 10 | 0.423 ± 0.024 | 3.00 ± 0.00 | 1.00 ± 0.00 | 68 ± 35 | -- | -- | 0.97 ± 0.02 | 0.549 ± 0.007 | 0.304 ± 0.006 | 0.585 ± 0.023 | 0.683 ± 0.026 | 0.682 ± 0.005 | 0.035 ± 0.006 |
| `r3` | 10 | 0.635 ± 0.039 | 2.70 ± 0.46 | 0.70 ± 0.46 | 411 ± 415 | -- | -- | 0.34 ± 0.11 | 0.319 ± 0.020 | 0.168 ± 0.019 | 0.511 ± 0.024 | 0.657 ± 0.023 | 0.644 ± 0.026 | 0.060 ± 0.020 |
| `r0` | 10 | 0.468 ± 0.014 | 3.00 ± 0.00 | 1.00 ± 0.00 | 54 ± 15 | -- | -- | 1.00 ± 0.00 | 0.483 ± 0.004 | 0.261 ± 0.002 | -- | 0.662 ± 0.020 | -- | -- |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r4` | 30 | 0 of 30 |
| `r4:tau=0.1` | 30 | 0 of 30 |
| `r4:tau=0.25` | 30 | 0 of 30 |
| `r4:tau=0.5` | 30 | 0 of 30 |
| `r4:tau=2` | 30 | 0 of 30 |
| `r4:mmr=0.5` | 30 | 0 of 30 |
| `r4:mmr=0.9` | 30 | 0 of 30 |
| `r4:quota=0` | 30 | 0 of 30 |
| `r4:quota=10` | 30 | 0 of 30 |
| `r3` | 29 | 3 of 30 |
| `r0` | 30 | 0 of 30 |

### Comparisons

| variant | comparison | metric | mean diff | 90% CI | sign test (W-L) | losing seeds | seeds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_semantic | `r4:tau=0.1` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 10 | **inconclusive** |
| S1_semantic | `r4:tau=0.1` vs `r4` | like_rate | +0.0856 | [+0.0664, +0.1035] | 10-0 (p = 0.00195) | none | 10 | **supported** |
| S1_semantic | `r4:tau=0.25` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 10 | **inconclusive** |
| S1_semantic | `r4:tau=0.25` vs `r4` | like_rate | +0.0412 | [+0.0270, +0.0551] | 9-1 (p = 0.0215) | 1732 | 10 | **supported** |
| S1_semantic | `r4:tau=0.5` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 10 | **inconclusive** |
| S1_semantic | `r4:tau=0.5` vs `r4` | like_rate | +0.0221 | [+0.0094, +0.0346] | 6-4 (p = 0.754) | 1731, 1735, 1736, 1737 | 10 | **supported** |
| S1_semantic | `r4:tau=2` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 10 | **inconclusive** |
| S1_semantic | `r4:tau=2` vs `r4` | like_rate | -0.0247 | [-0.0346, -0.0153] | 1-9 (p = 0.0215) | 1729, 1730, 1731, 1732, 1733, 1735, 1736, 1737, 1738 | 10 | **not supported** |
| S1_semantic | `r4:mmr=0.5` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 10 | **inconclusive** |
| S1_semantic | `r4:mmr=0.5` vs `r4` | like_rate | -0.0232 | [-0.0360, -0.0099] | 2-8 (p = 0.109) | 1729, 1731, 1732, 1733, 1735, 1736, 1737, 1738 | 10 | **not supported** |
| S1_semantic | `r4:mmr=0.9` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 10 | **inconclusive** |
| S1_semantic | `r4:mmr=0.9` vs `r4` | like_rate | +0.0104 | [-0.0003, +0.0209] | 7-3 (p = 0.344) | 1732, 1736, 1738 | 10 | **inconclusive** |
| S1_semantic | `r4:quota=0` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 10 | **inconclusive** |
| S1_semantic | `r4:quota=0` vs `r4` | like_rate | -0.0089 | [-0.0197, +0.0022] | 4-6 (p = 0.754) | 1732, 1733, 1735, 1736, 1737, 1738 | 10 | **inconclusive** |
| S1_semantic | `r4:quota=10` vs `r4` | all_found | +0.0000 | [+0.0000, +0.0000] | 0-0 (p = nan) | none | 10 | **inconclusive** |
| S1_semantic | `r4:quota=10` vs `r4` | like_rate | -0.0171 | [-0.0300, -0.0029] | 3-7 (p = 0.344) | 1730, 1731, 1732, 1735, 1736, 1737, 1738 | 10 | **not supported** |

**Like-rate gain kept** (H3 needs R4 to keep >= 70% of R3's gain over R0): (R4 - R0) / (R3 - R0), paired bootstrap 90% CI over seeds.

| variant | spec | gain kept | 90% CI | meets 70%? |
| --- | --- | --- | --- | --- |
| S1_semantic | `r4` | -0.17 | [-0.20, -0.13] | no |
| S1_semantic | `r4:tau=0.1` | 0.35 | [0.25, 0.43] | no |
| S1_semantic | `r4:tau=0.25` | 0.08 | [0.01, 0.14] | no |
| S1_semantic | `r4:tau=0.5` | -0.04 | [-0.11, 0.03] | no |
| S1_semantic | `r4:tau=2` | -0.32 | [-0.39, -0.26] | no |
| S1_semantic | `r4:mmr=0.5` | -0.31 | [-0.41, -0.22] | no |
| S1_semantic | `r4:mmr=0.9` | -0.11 | [-0.16, -0.05] | no |
| S1_semantic | `r4:quota=0` | -0.22 | [-0.29, -0.16] | no |
| S1_semantic | `r4:quota=10` | -0.27 | [-0.36, -0.19] | no |

---

## Skip weight: 0.25 vs 1.0 vs excluded (S1) (2026-11-01 05:14 UTC)

```
python -m visualtaste.jobs.evaluate --study phase9_skip --seeds 20 --swipes 1000
```

**SIMULATED.** Variants and their calibration: `backend/visualtaste/eval/sim.py`. Every number is mean ± sd across seeds; comparisons are paired by seed (Part 2 §4.1). An interest counts as found at 5 likes.

### S1_semantic

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r3` | 20 | 0.622 ± 0.045 | 2.75 ± 0.43 | 0.75 ± 0.43 | 351 ± 396 | -- | -- | 0.39 ± 0.11 | 0.318 ± 0.019 | 0.167 ± 0.016 | 0.510 ± 0.027 | 0.649 ± 0.026 | 0.653 ± 0.029 | 0.059 ± 0.021 |
| `r3:skip=1.0` | 20 | 0.621 ± 0.047 | 2.80 ± 0.40 | 0.80 ± 0.40 | 345 ± 408 | -- | -- | 0.39 ± 0.13 | 0.327 ± 0.032 | 0.173 ± 0.024 | 0.514 ± 0.025 | 0.648 ± 0.029 | 0.651 ± 0.025 | 0.096 ± 0.018 |
| `r3:skip=exclude` | 20 | 0.625 ± 0.043 | 3.00 ± 0.00 | 1.00 ± 0.00 | 199 ± 215 | -- | -- | 0.36 ± 0.13 | 0.318 ± 0.021 | 0.166 ± 0.017 | 0.508 ± 0.027 | 0.654 ± 0.027 | 0.657 ± 0.029 | 0.047 ± 0.017 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r3` | 30 | 5 of 60 |
| `r3:skip=1.0` | 30 | 4 of 60 |
| `r3:skip=exclude` | 33 | 0 of 60 |

### Comparisons

| variant | comparison | metric | mean diff | 90% CI | sign test (W-L) | losing seeds | seeds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_semantic | `r3:skip=exclude` vs `r3` | like_rate | +0.0023 | [-0.0092, +0.0132] | 11-9 (p = 0.824) | 1729, 1731, 1732, 1737, 1739, 1740, 1743, 1746, 1748 | 20 | **inconclusive** |
| S1_semantic | `r3:skip=exclude` vs `r3` | replay_log_loss | +0.0036 | [-0.0059, +0.0130] | 9-11 (p = 0.824) | 1731, 1735, 1736, 1737, 1738, 1739, 1743, 1744, 1745, 1746, 1748 | 20 | **inconclusive** |
| S1_semantic | `r3:skip=exclude` vs `r3:skip=1.0` | like_rate | +0.0033 | [-0.0070, +0.0144] | 11-9 (p = 0.824) | 1729, 1732, 1736, 1738, 1739, 1740, 1744, 1746, 1748 | 20 | **inconclusive** |
| S1_semantic | `r3:skip=exclude` vs `r3:skip=1.0` | replay_log_loss | +0.0054 | [-0.0041, +0.0156] | 9-11 (p = 0.824) | 1735, 1736, 1737, 1738, 1739, 1741, 1744, 1745, 1746, 1747, 1748 | 20 | **inconclusive** |

---

## Phase 10 checks: interleaving and IPS (S1, 2026-11-01 06:42 UTC)

```
python -m visualtaste.jobs.evaluate --study phase10_checks --seeds 20
```

**SIMULATED.** Interleaving: R3 (arm a) vs R4 (arm b), team-draft, 20 seeds x 300 swipes; a batch is won by the arm whose cards got more likes.

| batches won by R3 | won by R4 | sign test p | seeds won R3 / R4 / tied | interleaving winner | separate-run winner (like rate) | agree? |
| --- | --- | --- | --- | --- | --- | --- |
| 503 | 229 | 1.74e-24 | 18 / 1 / 1 | r3 | r3 (R3 0.622, R4 0.433) | yes |

**IPS (educational).** R3's like rate estimated by self-normalised IPS from 20 R4 sessions logged with 200-draw Monte-Carlo propensities and a shadow R3 fitted on the same labels (R3's propensity for a card is 1 if it would have put it in its own top five, else 0). Interval: bootstrap over seeds.

| IPS estimate | 90% CI | R3's actual like rate | bias | effective sample size | logged R4 cards |
| --- | --- | --- | --- | --- | --- |
| 0.613 | [0.567, 0.659] | 0.622 | -0.009 | 137 | 19,200 |

---

## Feature blocks: sig / sig+dino / sig+dino+col (S1, S2, S3) (2026-11-01 08:20 UTC)

```
python -m visualtaste.jobs.evaluate --study phase9_ablation --seeds 20 --swipes 1000
```

**SIMULATED.** Variants and their calibration: `backend/visualtaste/eval/sim.py`. Every number is mean ± sd across seeds; comparisons are paired by seed (Part 2 §4.1). An interest counts as found at 5 likes.

### S1_semantic

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r3` | 20 | 0.622 ± 0.045 | 2.75 ± 0.43 | 0.75 ± 0.43 | 351 ± 396 | -- | -- | 0.39 ± 0.11 | 0.318 ± 0.019 | 0.167 ± 0.016 | 0.510 ± 0.027 | 0.649 ± 0.026 | 0.653 ± 0.029 | 0.059 ± 0.021 |
| `r3:sig+dino` | 20 | 0.637 ± 0.046 | 2.85 ± 0.36 | 0.85 ± 0.36 | 265 ± 331 | -- | -- | 0.33 ± 0.12 | 0.291 ± 0.032 | 0.155 ± 0.023 | 0.513 ± 0.017 | 0.649 ± 0.025 | 0.643 ± 0.033 | 0.054 ± 0.023 |
| `r3:sig+dino+col` | 20 | 0.611 ± 0.049 | 2.85 ± 0.36 | 0.85 ± 0.36 | 362 ± 380 | -- | -- | 0.41 ± 0.12 | 0.323 ± 0.033 | 0.168 ± 0.022 | 0.508 ± 0.030 | 0.655 ± 0.027 | 0.657 ± 0.028 | 0.053 ± 0.017 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r3` | 30 | 5 of 60 |
| `r3:sig+dino` | 31 | 3 of 60 |
| `r3:sig+dino+col` | 31 | 3 of 60 |

### S2_appearance

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r3` | 20 | 0.675 ± 0.040 | 3.00 ± 0.00 | 1.00 ± 0.00 | 105 ± 154 | -- | -- | 0.43 ± 0.16 | 0.334 ± 0.029 | 0.179 ± 0.019 | 0.511 ± 0.030 | 0.732 ± 0.024 | 0.606 ± 0.045 | 0.046 ± 0.018 |
| `r3:sig+dino` | 20 | 0.709 ± 0.041 | 2.85 ± 0.36 | 0.85 ± 0.36 | 215 ± 334 | -- | -- | 0.41 ± 0.14 | 0.326 ± 0.027 | 0.174 ± 0.017 | 0.502 ± 0.034 | 0.718 ± 0.039 | 0.592 ± 0.047 | 0.041 ± 0.017 |
| `r3:sig+dino+col` | 20 | 0.688 ± 0.053 | 2.95 ± 0.22 | 0.95 ± 0.22 | 117 ± 206 | -- | -- | 0.47 ± 0.16 | 0.335 ± 0.027 | 0.178 ± 0.016 | 0.514 ± 0.031 | 0.722 ± 0.033 | 0.597 ± 0.057 | 0.044 ± 0.017 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r3` | 34 | 0 of 60 |
| `r3:sig+dino` | 33 | 3 of 60 |
| `r3:sig+dino+col` | 34 | 1 of 60 |

### S3_colour

| spec | seeds | like rate | found | all found | swipes to find all | S5 recovery (swipes) | id stability | coverage | diversity | novelty | replay AUC | oracle AUC | replay log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r3` | 20 | 0.511 ± 0.034 | 2.00 ± 0.00 | 1.00 ± 0.00 | 62 ± 58 | -- | -- | 0.67 ± 0.13 | 0.387 ± 0.039 | 0.214 ± 0.027 | 0.507 ± 0.021 | 0.648 ± 0.027 | 0.703 ± 0.013 | 0.051 ± 0.017 |
| `r3:sig+dino` | 20 | 0.495 ± 0.030 | 2.00 ± 0.00 | 1.00 ± 0.00 | 89 ± 149 | -- | -- | 0.60 ± 0.11 | 0.352 ± 0.036 | 0.191 ± 0.023 | 0.508 ± 0.027 | 0.645 ± 0.027 | 0.702 ± 0.014 | 0.047 ± 0.018 |
| `r3:sig+dino+col` | 20 | 0.524 ± 0.038 | 2.00 ± 0.00 | 1.00 ± 0.00 | 113 ± 167 | -- | -- | 0.61 ± 0.16 | 0.365 ± 0.029 | 0.199 ± 0.014 | 0.508 ± 0.027 | 0.635 ± 0.023 | 0.698 ± 0.018 | 0.050 ± 0.014 |

| spec | median swipes to find an interest | censored (never found) |
| --- | --- | --- |
| `r3` | 22 | 0 of 40 |
| `r3:sig+dino` | 22 | 0 of 40 |
| `r3:sig+dino+col` | 22 | 0 of 40 |

### Comparisons

| variant | comparison | metric | mean diff | 90% CI | sign test (W-L) | losing seeds | seeds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_semantic | `r3:sig+dino` vs `r3` | replay_log_loss | -0.0103 | [-0.0235, +0.0012] | 14-6 (p = 0.115) | 1729, 1731, 1734, 1737, 1743, 1744 | 20 | **inconclusive** |
| S1_semantic | `r3:sig+dino` vs `r3` | like_rate | +0.0145 | [-0.0002, +0.0307] | 13-7 (p = 0.263) | 1729, 1731, 1734, 1737, 1743, 1744, 1746 | 20 | **inconclusive** |
| S1_semantic | `r3:sig+dino+col` vs `r3:sig+dino` | replay_log_loss | +0.0144 | [+0.0028, +0.0267] | 8-12 (p = 0.503) | 1730, 1731, 1732, 1733, 1735, 1738, 1739, 1742, 1743, 1745, 1747, 1748 | 20 | **not supported** |
| S1_semantic | `r3:sig+dino+col` vs `r3:sig+dino` | like_rate | -0.0255 | [-0.0390, -0.0121] | 6-14 (p = 0.115) | 1730, 1731, 1732, 1733, 1735, 1737, 1738, 1739, 1742, 1743, 1744, 1745, 1747, 1748 | 20 | **not supported** |
| S2_appearance | `r3:sig+dino` vs `r3` | replay_log_loss | -0.0148 | [-0.0294, -0.0009] | 12-8 (p = 0.503) | 1729, 1730, 1740, 1741, 1742, 1745, 1747, 1748 | 20 | **supported** |
| S2_appearance | `r3:sig+dino` vs `r3` | like_rate | +0.0340 | [+0.0201, +0.0488] | 14-6 (p = 0.115) | 1729, 1744, 1745, 1746, 1747, 1748 | 20 | **supported** |
| S2_appearance | `r3:sig+dino+col` vs `r3:sig+dino` | replay_log_loss | +0.0050 | [-0.0130, +0.0228] | 9-11 (p = 0.824) | 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1744, 1745 | 20 | **inconclusive** |
| S2_appearance | `r3:sig+dino+col` vs `r3:sig+dino` | like_rate | -0.0211 | [-0.0389, -0.0060] | 6-13 (p = 0.167) | 1729, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1739, 1742, 1743, 1744, 1745 | 20 | **not supported** |
| S3_colour | `r3:sig+dino` vs `r3` | replay_log_loss | -0.0017 | [-0.0082, +0.0052] | 14-6 (p = 0.115) | 1730, 1731, 1736, 1741, 1745, 1748 | 20 | **inconclusive** |
| S3_colour | `r3:sig+dino` vs `r3` | like_rate | -0.0157 | [-0.0298, -0.0033] | 8-11 (p = 0.648) | 1730, 1731, 1732, 1733, 1734, 1736, 1737, 1741, 1745, 1746, 1748 | 20 | **not supported** |
| S3_colour | `r3:sig+dino+col` vs `r3:sig+dino` | replay_log_loss | -0.0035 | [-0.0107, +0.0034] | 9-11 (p = 0.824) | 1732, 1733, 1736, 1737, 1738, 1739, 1740, 1742, 1743, 1747, 1748 | 20 | **inconclusive** |
| S3_colour | `r3:sig+dino+col` vs `r3:sig+dino` | like_rate | +0.0293 | [+0.0158, +0.0434] | 16-4 (p = 0.0118) | 1734, 1742, 1743, 1744 | 20 | **supported** |

---

## H2: R3 vs R2 by label count, replayed on cold-start histories (2026-11-01 13:05 UTC)

```
python -m visualtaste.jobs.evaluate --study phase9_h2 --seeds 20
```

**SIMULATED**, S1, 20 seeds. The histories were logged by R0, so neither rung chose the images it is tested on. At each cut both rungs are fitted on the first N events and scored on the next 100. R2's similarity score is Platt-scaled out-of-sample before log loss (`eval/study.py`).

### Replay log_loss

| labels | `r3` log_loss | `r2` log_loss | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.6992 | 0.7633 | -0.0642 | [-0.0942, -0.0359] | 15-5 | supported |
| 100 | 0.7024 | 0.7198 | -0.0174 | [-0.0267, -0.0094] | 15-5 | supported |
| 150 | 0.7265 | 0.7036 | +0.0229 | [-0.0055, +0.0563] | 10-10 | inconclusive |
| 200 | 0.7097 | 0.7229 | -0.0132 | [-0.0329, +0.0090] | 14-6 | inconclusive |
| 250 | 0.7028 | 0.7101 | -0.0073 | [-0.0162, +0.0013] | 13-7 | inconclusive |
| 300 | 0.6953 | 0.7017 | -0.0064 | [-0.0136, +0.0002] | 11-9 | inconclusive |
| 350 | 0.6956 | 0.6973 | -0.0017 | [-0.0074, +0.0042] | 11-9 | inconclusive |
| 400 | 0.6922 | 0.7013 | -0.0091 | [-0.0172, -0.0019] | 12-8 | supported |
| 450 | 0.6916 | 0.6985 | -0.0070 | [-0.0120, -0.0018] | 15-5 | supported |
| 500 | 0.6893 | 0.7015 | -0.0122 | [-0.0205, -0.0046] | 14-6 | supported |
| 550 | 0.6902 | 0.7021 | -0.0119 | [-0.0172, -0.0066] | 15-5 | supported |
| 600 | 0.6905 | 0.7005 | -0.0100 | [-0.0157, -0.0040] | 15-5 | supported |
| 650 | 0.6946 | 0.6951 | -0.0005 | [-0.0056, +0.0045] | 12-8 | inconclusive |
| 700 | 0.6948 | 0.7024 | -0.0076 | [-0.0138, -0.0015] | 13-7 | supported |
| 750 | 0.6889 | 0.6967 | -0.0078 | [-0.0114, -0.0040] | 16-4 | supported |
| 800 | 0.6881 | 0.7011 | -0.0131 | [-0.0197, -0.0068] | 15-5 | supported |
| 850 | 0.6885 | 0.6989 | -0.0104 | [-0.0157, -0.0057] | 16-4 | supported |
| 900 | 0.6917 | 0.6971 | -0.0053 | [-0.0114, +0.0009] | 12-8 | inconclusive |

### Replay auc

| labels | `r3` auc | `r2` auc | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.5223 | 0.5146 | +0.0077 | [-0.0237, +0.0400] | 11-9 | inconclusive |
| 100 | 0.5520 | 0.5393 | +0.0127 | [-0.0315, +0.0606] | 7-13 | inconclusive |
| 150 | 0.5310 | 0.5554 | -0.0244 | [-0.0585, +0.0104] | 7-13 | inconclusive |
| 200 | 0.5267 | 0.5534 | -0.0267 | [-0.0604, +0.0100] | 6-14 | inconclusive |
| 250 | 0.5250 | 0.5692 | -0.0442 | [-0.0723, -0.0134] | 4-16 | not supported |
| 300 | 0.5404 | 0.5704 | -0.0300 | [-0.0573, -0.0018] | 7-13 | not supported |
| 350 | 0.5406 | 0.5598 | -0.0193 | [-0.0661, +0.0297] | 8-12 | inconclusive |
| 400 | 0.5568 | 0.5961 | -0.0393 | [-0.0607, -0.0176] | 7-13 | not supported |
| 450 | 0.5713 | 0.5887 | -0.0173 | [-0.0352, -0.0000] | 7-13 | not supported |
| 500 | 0.5671 | 0.5765 | -0.0094 | [-0.0443, +0.0332] | 5-15 | inconclusive |
| 550 | 0.5628 | 0.5763 | -0.0134 | [-0.0384, +0.0123] | 5-14 | inconclusive |
| 600 | 0.5525 | 0.5859 | -0.0335 | [-0.0527, -0.0157] | 6-14 | not supported |
| 650 | 0.5428 | 0.5714 | -0.0286 | [-0.0468, -0.0103] | 5-15 | not supported |
| 700 | 0.5543 | 0.5870 | -0.0327 | [-0.0542, -0.0100] | 4-16 | not supported |
| 750 | 0.5707 | 0.6080 | -0.0373 | [-0.0532, -0.0203] | 4-16 | not supported |
| 800 | 0.5841 | 0.6117 | -0.0275 | [-0.0456, -0.0079] | 5-15 | not supported |
| 850 | 0.5842 | 0.5845 | -0.0003 | [-0.0348, +0.0452] | 6-14 | inconclusive |
| 900 | 0.5632 | 0.5831 | -0.0199 | [-0.0397, +0.0001] | 6-14 | inconclusive |


### Reading H2

H2 is **not supported** in the form it was stated. The two metrics split:

- **Calibration goes to R3.** From 400 labels on, R3's log loss is lower at all 11 cuts, and the CI excludes zero at 9 of them. Its probabilities are closer to the truth than R2's, even after R2 is Platt-scaled.
- **Ranking goes to R2.** From 150 labels on, R2's AUC is higher at every cut, and significantly so at 9 of those 16. The feed shows the top of a ranking, so this is the metric that decides what the user sees.
- **Both are weak here.** All AUCs are between 0.51 and 0.61. The replay histories were logged by R0, so most test images sit far from any planted interest, and the next 100 events hold few positives to rank. This is a hard, low-signal test, not the on-policy one in the R0-R3 study above.

So the "after about 150 labels" crossover did not appear. R3's advantage is that it gives usable probabilities (needed by R4's posterior and by calibration plots), not that it ranks better than kNN on this replay.

---

## H5: R6 vs R3+R5 by label count, and the dwell study (SIMULATED, 2026-11-01 13:31 UTC)

```
python -m visualtaste.jobs.evaluate --study phase12_h5 --seeds 10
```

Histories: 10 seeds x 5,000 swipes per variant, logged by R4 (the default chain), so the training data carries the selection bias real data will. At each label count both models are fitted on the first N events and scored on the next 100; paired by seed. R6 is the 5-member ensemble, trained on the GPU.

### S1_semantic: `r6:minlabels=0` vs `r3+r5`, replay log_loss

| labels | `r6:minlabels=0` log_loss | `r3+r5` log_loss | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.6866 | 0.6831 | +0.0035 | [-0.0096, +0.0198] | 6-4 | inconclusive |
| 1500 | 0.6611 | 0.6562 | +0.0049 | [-0.0061, +0.0173] | 4-6 | inconclusive |
| 3000 | 0.6486 | 0.6444 | +0.0041 | [-0.0048, +0.0135] | 5-5 | inconclusive |
| 4900 | 0.6718 | 0.6737 | -0.0019 | [-0.0056, +0.0021] | 7-3 | inconclusive |

### S1_semantic: `r6:minlabels=0` vs `r3+r5`, replay auc

| labels | `r6:minlabels=0` auc | `r3+r5` auc | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.5911 | 0.6056 | -0.0144 | [-0.0283, +0.0002] | 2-7 | inconclusive |
| 1500 | 0.6266 | 0.6329 | -0.0063 | [-0.0224, +0.0101] | 4-6 | inconclusive |
| 3000 | 0.6319 | 0.6453 | -0.0134 | [-0.0275, -0.0005] | 2-7 | not supported |
| 4900 | 0.6166 | 0.6113 | +0.0053 | [-0.0028, +0.0136] | 6-4 | inconclusive |

### S1_semantic: `r6:minlabels=0,loss=bpr` vs `r3+r5`, replay log_loss

| labels | `r6:minlabels=0,loss=bpr` log_loss | `r3+r5` log_loss | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.6882 | 0.6831 | +0.0051 | [-0.0041, +0.0162] | 5-5 | inconclusive |
| 1500 | 0.6633 | 0.6562 | +0.0071 | [+0.0004, +0.0140] | 5-5 | not supported |
| 3000 | 0.6464 | 0.6444 | +0.0020 | [-0.0119, +0.0176] | 6-4 | inconclusive |
| 4900 | 0.6797 | 0.6737 | +0.0060 | [-0.0034, +0.0169] | 4-6 | inconclusive |

### S1_semantic: `r6:minlabels=0,loss=bpr` vs `r3+r5`, replay auc

| labels | `r6:minlabels=0,loss=bpr` auc | `r3+r5` auc | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.5928 | 0.6056 | -0.0128 | [-0.0257, +0.0003] | 3-7 | inconclusive |
| 1500 | 0.6284 | 0.6329 | -0.0045 | [-0.0220, +0.0130] | 6-4 | inconclusive |
| 3000 | 0.6316 | 0.6453 | -0.0137 | [-0.0318, +0.0022] | 4-6 | inconclusive |
| 4900 | 0.6181 | 0.6113 | +0.0067 | [-0.0007, +0.0141] | 7-3 | inconclusive |

### S4_conjunctive: `r6:minlabels=0` vs `r3+r5`, replay log_loss

| labels | `r6:minlabels=0` log_loss | `r3+r5` log_loss | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.6861 | 0.6830 | +0.0030 | [-0.0000, +0.0067] | 3-7 | inconclusive |
| 1500 | 0.6911 | 0.6904 | +0.0007 | [-0.0071, +0.0083] | 4-6 | inconclusive |
| 3000 | 0.6934 | 0.6872 | +0.0062 | [-0.0026, +0.0157] | 4-6 | inconclusive |
| 4900 | 0.6804 | 0.6887 | -0.0083 | [-0.0140, -0.0027] | 8-2 | supported |

### S4_conjunctive: `r6:minlabels=0` vs `r3+r5`, replay auc

| labels | `r6:minlabels=0` auc | `r3+r5` auc | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.5418 | 0.5547 | -0.0129 | [-0.0363, +0.0089] | 4-6 | inconclusive |
| 1500 | 0.5587 | 0.5656 | -0.0069 | [-0.0285, +0.0148] | 4-6 | inconclusive |
| 3000 | 0.5583 | 0.5832 | -0.0249 | [-0.0488, -0.0018] | 4-6 | not supported |
| 4900 | 0.6017 | 0.5960 | +0.0057 | [-0.0158, +0.0265] | 6-4 | inconclusive |

### S4_conjunctive: `r6:minlabels=0,loss=bpr` vs `r3+r5`, replay log_loss

| labels | `r6:minlabels=0,loss=bpr` log_loss | `r3+r5` log_loss | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.6864 | 0.6830 | +0.0034 | [-0.0006, +0.0085] | 4-6 | inconclusive |
| 1500 | 0.6925 | 0.6904 | +0.0021 | [-0.0068, +0.0113] | 4-6 | inconclusive |
| 3000 | 0.7006 | 0.6872 | +0.0134 | [+0.0058, +0.0218] | 2-8 | not supported |
| 4900 | 0.6915 | 0.6887 | +0.0028 | [-0.0098, +0.0148] | 4-6 | inconclusive |

### S4_conjunctive: `r6:minlabels=0,loss=bpr` vs `r3+r5`, replay auc

| labels | `r6:minlabels=0,loss=bpr` auc | `r3+r5` auc | diff | 90% CI | W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.5494 | 0.5547 | -0.0053 | [-0.0301, +0.0187] | 5-5 | inconclusive |
| 1500 | 0.5554 | 0.5656 | -0.0102 | [-0.0337, +0.0128] | 4-6 | inconclusive |
| 3000 | 0.5608 | 0.5832 | -0.0224 | [-0.0522, +0.0041] | 4-6 | inconclusive |
| 4900 | 0.5887 | 0.5960 | -0.0072 | [-0.0321, +0.0189] | 4-6 | inconclusive |

### Dwell (SIMULATED)

Simulated dwell is `exp(N(ln 2500 + 0.3 (p - 0.5) + 0.4 novelty, 0.5))` (`eval/sim.simulated_dwell`): weakly tied to liking, more to novelty.

| variant | AUC of dwell alone | R3 replay log loss without | with log-dwell | diff | 90% CI | W-L | promote? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1_semantic | 0.518 ± 0.010 | 0.6627 | 0.6627 | -0.0000 | [-0.0002, +0.0002] | 3-7 (p = 0.344) | no |
| S4_conjunctive | 0.512 ± 0.009 | 0.6898 | 0.6899 | +0.0001 | [-0.0000, +0.0002] | 4-6 (p = 0.754) | no |

Rule (DECISIONS #26): dwell becomes a feature only if it lowers log loss with a CI excluding 0, and even then stays off for real data until there are 1,000 real labels.

