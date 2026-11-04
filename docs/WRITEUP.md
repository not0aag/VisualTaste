# Learning one person's visual taste from swipes: what worked, what did not

VisualTaste is a local, single-user recommender for photographs. It shows five
images at a time; you like, dislike, save or skip them; and it learns what you
like. It runs on one laptop (an RTX 3060 and 16 GB of RAM), and no image, label
or model leaves the machine. This write-up covers what the project measured,
including the parts that did not work. Every number comes from the files in
`docs/` and the caches behind them.

**One caveat governs all of it.** Every comparison below uses a *simulated*
user with a planted taste, not a person. A simulator lets each idea be tested
on 20 paired seeds with a known ground truth. But its notion of taste
("closeness to a few hidden points in SigLIP 2 space") is close to what the
simpler models assume, so it flatters them. The results say which mechanisms
work as designed, not which one a given person will prefer.

## The collection

The pool is 50,000 photographs, selected from 80,036 downloaded ones
(`docs/DATASET.md`). They come from four openly licensed sources: Megalith-10M
(Flickr public domain), PD12M (CC0 and Public Domain Mark), Open Images V7
(CC BY 2.0) and Unsplash Lite. The selection steps were:

- remove near-duplicates by perceptual hash, then by DINOv2 cosine;
- run a zero-shot safety screen calibrated on 200 hand-checked images, which
  removed 789 images (about 1%);
- balance the rest across 400 SigLIP 2 clusters, so a niche interest is not
  drowned out by whatever the sources hold most of.

Two checks were done by hand:

- **Safety screen.** Of 100 sampled removals, none was explicit and 91 were
  harmless. The screen is cautious rather than precise, and the dataset card
  says so.
- **Megalith licences.** The collection is public domain at the collection
  level. A spot check of 200 images confirmed 191 individually, so the demo
  pack only redistributes Megalith images that were checked one by one.

## Two vectors per image

Every image gets two embeddings: SigLIP 2 (trained on image–text pairs, so it
encodes *what* is in the picture) and DINOv2 (self-supervised, so it leans
towards *how* the picture looks). On top of these sit a PCA-128 of each, a
64-bin CIELAB colour histogram and 64 coarse clusters used for exploration.
Embedding runs once on the GPU. Everything after that runs on the CPU in
milliseconds.

## Seven rungs

The recommender is a ladder. Each rung is simple enough to explain and adds
one idea to the one below.

| rung | idea |
| --- | --- |
| R0 | cold start: round-robin over the 64 clusters |
| R1 | Rocchio: move towards likes, away from dislikes |
| R2 | k nearest neighbours of your likes (FAISS) |
| R3 | L2 logistic regression on the PCA-128 of both embeddings, with its regularisation chosen on time-ordered splits (the colour block was tested and dropped: DECISIONS #28) |
| R4 | R3's Laplace posterior, sampled (Thompson sampling), plus MMR diversity and a small quota of probes into unexplored clusters |
| R5 | k-means "interests" over your likes, matched across refits so they keep their identity; used as extra features (R3+R5, R4+R5) |
| R6 | a small MLP ensemble |

Every served card says why it was chosen (`rec/explain.py`). For R3 that
includes the three liked images closest to it.

## How it was measured

The simulator (`eval/sim.py`) plants three to six hidden interest centroids and
draws likes from `sigmoid(10 · (cosine − offset))`. The offset is calibrated
per seed so that random images are liked half the time. It drives the real
feed code, writes real label rows to a sandbox profile, and the real models
retrain every 20 labels. Six scenarios vary the taste:

- S1: semantic interests;
- S2: appearance (DINOv2) interests;
- S3: colour;
- S4: a conjunction ("this subject *in* this palette");
- S5: drift (the taste changes mid-session);
- S6: six interests instead of three.

Each comparison is paired by seed: 20 seeds × 1,000 swipes, a bootstrap 90%
interval on the mean difference, and a sign test. Learned models are also
scored by replaying the session in time order (AUC, log loss, calibration) and
against the simulator's true probabilities. Five hypotheses were written down
before the results existed.

## What held, and what did not

**H1: learning beats the cold start. Supported, with a catch.** R1 and R2
raise the like rate by about 0.14 and 0.16 over R0. But they find only two of
three planted interests: they keep serving more of whatever got liked first.

**H2: logistic regression overtakes kNN after about 150 labels. Not supported
for ranking.** R3 is better calibrated (lower log loss from 400 labels, at 9 of
11 cuts) but ranks *worse* than R2 by AUC at every cut from 150 labels on.
Which model is "better" depends on whether you need a probability or an order.

**H3: Thompson sampling finds all interests faster than R3. Supported for
discovery, not for like rate.** R4 finds every interest in many more seeds
(+0.25 on S1, +0.55 on S5, +0.85 on S6) and hundreds of swipes sooner. It pays
for that in like rate. The bar was to keep at least 70% of R3's gain over R0.
In the H3 study, at the specified posterior temperature of 1, R4's like rate
fell below even R0's. In the final study, at temperature 0.1 (DECISIONS #25),
it keeps 40–49% of the gain on four scenarios, 77%
on S2 (inconclusive against the bar) and less than nothing on S4. Exploration
costs the likes it spends.

**H4: interest features help. Supported only under drift.** R3+R5 beats R3 by
+0.021 like rate [+0.008, +0.036] on S5, where the taste moves. On the other
five scenarios there is no detectable difference, and none on top of R4. The
interests are useful for *showing* the person their taste (the Profile page
lists them) more than for ranking.

**H5: a neural ranker wins once labels are plentiful. Not supported.** Over 10
seeds × 5,000 swipes, with a plain loss and with a pairwise one, the MLP
ensemble never beat R3+R5 on AUC at 500, 1,500, 3,000 or 4,900 labels. It won
once on log loss, by 0.008. At this scale, a well-regularised linear model on
good embeddings is hard to beat. Because the simulator's taste is close to
linear, this result does not settle the question for real people.

## The default, and why it is not the "best" model

A final study ran every serving rung on all six scenarios: 840 sessions. The
default was chosen by one rule fixed in advance: the sum of z-scores for like
rate, interests found and coverage, ranked per scenario. **R4+R5 ranks first**
(mean rank 1.8), followed by R4 (2.4), R3 (2.8) and R3+R5 (3.0). R1 is last at
5.0. R2 is excluded because it misses the 50 ms latency budget.

R3 still has the higher like rate. R4+R5 was chosen because a recommender that
never shows you the second and third things you like has failed in a way its
like rate cannot show. The app says which rung is serving, and Model Lab lets
the person switch.

## Describing a taste in words

SigLIP 2 embeds text into the same space as the images, so the app has text
search and an optional "describe what you like" box during the cold start. The
simulator was extended to type a description (the three concept words nearest
one hidden interest), and the share of cold-start cards aimed at it was
measured:

| share aimed at the description | S1 like rate | S1 interests found | S6 interests found |
| --- | --- | --- | --- |
| 60% | +0.092 | −0.60 | −2.60 |
| 40% | +0.067 | −0.20 (inconclusive) | −1.30 |
| 20% | +0.048 | ±0.00 | −0.90 |

A description raises the early like rate and costs the interests it does not
mention. The tension is the same as in H3. The app uses 20%: in S1 it loses
nothing measurable, and the cost on S6 is shown honestly.

## After the final study: five follow-ups

Each idea was added as an opt-in option, so no earlier result moved, and was
run on 20 paired seeds.

| idea | result |
| --- | --- |
| Fade R4's sampling temperature, narrow its diversity shortlist, weight relevance more | **Works.** Together they keep 70% (S1) and 80% (S6) of R3's like-rate gain, against R4's 40%, and still find every interest in every seed. They give up some coverage |
| Share each batch among the interests already found (R4+R5) | **Works.** Like rate +0.029 (S1) and +0.015 (S6), no interest lost. The first use of R5's interests that changes what is served for the better |
| Blend kNN's ranking into R3's | **Does not help.** Fewer interests found on S1, lower like rate on S2 |
| Let old labels fade, for taste drift | **Does not help.** No faster recovery after the taste changes; costs R4 like rate |
| Re-tune R3's regularisation on every third refit only | Neutral for R3, −0.011 like rate for R4. Kept for quick experiments only |

The two that work were then tested on all six scenarios
(`phase16_default`). The exploration settings raise R4+R5's like rate in
every scenario (+0.019 to +0.058) and keep 60–90% of R3's gain. Every
interest is still found in every seed in five scenarios; under drift, 4 of 20
seeds miss one. Allocation on top of them makes things worse. The rule that
chose the default ranks the tuned version first (mean rank 2.6 against 2.8),
so the app now uses it. The simulator's own models are unchanged, so no
cached result moved.

## Speed, measured end to end

The feed has a 50 ms p95 budget. Measured through the whole request (HTTP,
database, model, explanation), R0 to R3 stay inside it up to 1,000 labels
and R0 up to 3,000; the default R4+R5 takes 53–66 ms. A profile showed the
model's own work is 10–25 ms. The rest is the request around it, about
20 ms of it SQLite on Windows. Two fixes landed: explanations stopped
repeating work per card, and each profile's "already seen" set is kept
between requests instead of being re-read. Both were checked to leave every
result byte-identical. The remaining overshoot is reported in the README as
a miss, not rounded away.

## Engineering notes

- **Speed-ups were kept bit-identical.** Each optimisation was checked by
  re-running fixed seeds on every rung and comparing the outputs byte for byte
  against a stored golden set. Optimisations that would change a result went
  behind new, opt-in options with their own cache keys. One example is
  re-tuning R3's regularisation only on every k-th refit. So no published
  number had to be re-run.
- **Order of operations matters more than FLOPs.** Scoring the whole pool and
  then indexing the candidates is eight times faster than gathering the
  candidates first (7 ms against 58.5 ms): the gather is a 153 MB scattered copy,
  and the product is one pass through BLAS.
- **Off-policy estimates were validated, not assumed.** Self-normalised IPS
  and a doubly robust estimator are checked against the simulator's true
  values. The doubly robust estimator cut variance by about 6% on this data,
  much less than textbooks suggest. The tests assert the measured behaviour.
- **Everything has a test.** That covers backend unit tests, Vitest, Playwright
  end-to-end runs and an axe accessibility audit of every page, all in CI on
  Linux. The local development machine is Windows.

## Limitations

- No real person's labels have been used. The first real profile is the real
  test.
- The simulator's taste is a few points in an embedding space. Real taste can
  be contextual, can change by mood, and can depend on things neither
  embedding encodes.
- The pool is fixed at 50,000 images. Nothing new arrives, so there is no
  freshness problem to solve and no long-term novelty to measure.
- One user, one machine: no collaborative signal, by design.

## Where to look

- `docs/EVALUATION.md`: every study, with tables, intervals and verdicts.
- `docs/DECISIONS.md`: each design choice and the evidence behind it.
- `docs/DATASET.md`: the collection card.
- `docs/BENCHMARKS.md`: speed and memory measurements.
- `docs/ROADMAP.md`: what remains.
