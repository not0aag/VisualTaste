# Learning path

A reading order through the codebase, one maths unit at a time. Each step:
read a file (its module docstring states the intuition, the formula, every
variable and a worked example that a test asserts), run its notebook, then
change one thing and watch what moves.

Every notebook in `notebooks/` is saved with its outputs, so you can read one
without running it. To re-run: `.venv\Scripts\jupyter lab`, or
`jupyter nbconvert --execute --to notebook --inplace notebooks/<name>.ipynb`.

The exercises use the planted-taste simulator, so you can try them before you
have swiped anything. A simulated profile's dashboards are always marked
SIMULATED, and nothing a simulation does can touch your real profile.

---

## M1 · Vectors and cosine similarity

- **Read:** `vision/encoders.py` (L2 normalisation, and why it makes cosine a dot product), `search/index.py`.
- **Notebook:** `01_vectors_and_cosine`.
- **Test:** `tests/test_encoders.py`. A = (1,2,2), B = (2,1,2) gives 8/9.
- **Try:** in Feed, open a card's explanation (I), then History → "more like this" on an image you liked. These are the FAISS nearest neighbours by cosine. Compare the SigLIP neighbours with the DINOv2 ones (`/api/images/{id}/similar?model=dinov2_b14`): what each model counts as similar is the whole difference between S1 and S2 below.

## M2 · Rocchio and kNN (R1, R2)

- **Read:** `rec/r1_rocchio.py`, then `rec/r2_knn.py`. Read `R1.score`'s docstring for why scoring every row beats selecting candidates first (DECISIONS #13).
- **Notebook:** `02_rocchio_and_knn`.
- **Try:** change `BETA` in `rec/r1_rocchio.py` from 0.5 to 1.0, then run `python -m visualtaste.jobs.evaluate --rungs r0,r1 --seeds 5`. Dislikes now steer as hard as likes, and the like rate moves. Revert afterwards.

## M3 · Labels, weights and the handover chain

- **Read:** `rec/labels.py` (what each swipe is worth), `rec/base.py` (the interface every rung shares), `rec/feed.py` (the request path, the fallback, impressions).
- **Try:** swipe 30 likes and no dislikes on a fresh profile. The Feed keeps serving cold start, and the explanation panel's `fallback` says why ("30 positive and 0 negative labels"). That is the handover chain (Part 2 §4.4).

## M4 · PCA, standardisation and UMAP

- **Read:** `jobs/select.run_pca`, `rec/features.py` (why every feature is standardised before an L2 penalty sees it), `viz/projection.py`.
- **Notebook:** `04_pca_and_umap`.
- **Try:** open the **Embedding Map**. Colour by cluster: clean islands on the SigLIP map, mixed colours on the DINOv2 map. Then swipe 100 images and colour by "current model score", and watch the high-score region form around your likes.

## M5 · Logistic regression (R3)

- **Read:** `rec/logreg_numpy.py` (every derivative written out), then `rec/r3_logreg.py`. Its docstring explains the one setting most people get wrong: C = 1/(λS), not 1/λ.
- **Notebook:** `03_logistic_regression_by_hand`. It compares gradient descent with scikit-learn, shows the loss curve and the effect of λ, and measures how standardisation changes convergence.
- **Tests:** `tests/test_logreg.py`. The M5 step from (2, −1) to (1.708, −1.219); the two solvers agreeing to 1e-3.
- **Try:** in Model Lab → Versions, press *train r3*, then *evaluate*. The λ column is what time-ordered validation chose. Swipe 200 more and repeat: λ usually falls as labels accumulate, because more data can carry a weaker prior.

## M6 · Time-ordered evaluation

- **Read:** `eval/replay.py` (why a random split flatters a recommender), `eval/metrics.py`, `eval/study.py` (paired bootstrap, sign test, the verdict rule).
- **Notebook:** `08_evaluation_metrics`.
- **Try:** `python -m visualtaste.jobs.evaluate --study phase9_skip --seeds 20`. Read the verdict column, then open the cached seeds under `data/eval/phase9_skip/` and find the losing seeds it names.

## M7 · Thompson sampling and MMR (R4)

- **Read:** `rec/r4_thompson.py` (the Laplace posterior, the Cholesky draw), `rec/rerank.py` (MMR, the exploration quota).
- **Notebook:** `05_thompson_sampling_and_mmr`. Its section 2 is the finding in DECISIONS #25: at τ = 1, one posterior draw's noise outweighs the signal.
- **Tests:** `tests/test_thompson.py`. P(B beats A) = 0.37; the sampled covariance equals H⁻¹.
- **Try:** set `VT_R4_TAU_POST=0.25` in `.env`, restart, and train R4 in Model Lab. On the timeline, compare like rate and coverage before and after. Then start an interleaving experiment between your R3 and R4 versions and watch the sign test.

## M8 · k-means and multiple interests (R5)

- **Read:** `rec/r5_interests.py` (soft-max over interests, nudges, Hungarian matching), then R3's interest features in `rec/r3_logreg.py`. Then the leakage guarantee in `rec/features.time_respecting_features`.
- **Notebook:** `06_kmeans_and_interests`.
- **Tests:** `tests/test_interests.py`, and `TestNoLeakage` especially: rewriting the future does not change the past.
- **Try:** once you have 100+ likes, train `r4+r5` in Model Lab and open **Profile**. Each card shows an interest, its three concept words and the Beta interval of your like rate near it. Retrain after another 100 likes: the ids survive the recluster.

## M9 · Neural ranking and backprop (R6)

- **Read:** `rec/r6_neural.py`, `jobs/train_neural.py`.
- **Notebook:** `07_neural_ranker_and_backprop`. Gradients by hand vs autograd, then the real ranker overfitting from its first epoch on simulated data.
- **Try:** after 1,500 labels, `python -m visualtaste.jobs.train_neural --record`, then *evaluate* it in Model Lab beside your best R4. Compare their calibration plots.

## M10 · Ranking metrics and calibration

- **Read:** `eval/metrics.ndcg_at_k`, `calibration_bins`, `expected_calibration_error`, `exploration_yield`.
- **Try:** Model Lab's calibration plot. If the "observed" line sits below the diagonal, the model is over-confident, and the "73% chance of a like" on Feed cards overstates it.

---

## Experiments with the simulator

Each variant plants a different kind of taste (`eval/sim.py`). Run a preset,
then read its section in `docs/EVALUATION.md`:

| question | variant | command |
| --- | --- | --- |
| Do learned rungs beat cold start? | S1 | `--study phase9_rungs` |
| Does appearance (DINOv2) help when taste is about looks? | S2 | `--study phase9_ablation` |
| Does colour help when taste is about palette? | S3 | `--study phase9_ablation` |
| Does exploration find more interests? | S1, S5, S6 | `--study phase10_h3` |
| Do several interests beat one vector? | S6 | `--study phase11_h4` |
| Does a network beat a linear model on a non-linear taste? | S4 | `--study phase12_h5` |
| Does the feed follow a changed taste? | S5 | `--study phase11_h4` (the drift recovery column) |

Add `--seeds 5` for a quick look; the reported studies use 20.
