# Changelog

All notable changes. Dates are when the work landed on `main`.

## Unreleased

Work toward 1.0.0, following [`docs/ROADMAP.md`](docs/ROADMAP.md).

### Evaluation
- H2, H3, H4 and H5 all have recorded verdicts. H5: the neural ranker did not
  beat R3+R5 at up to 4,900 labels.
- `phase14_final`: every serving rung on all six scenarios, 840 sessions.
  The default rung is now **R4+R5** (DECISIONS #29), with results by rung in
  EVALUATION.md.
- `phase15_text`: a described taste raises the cold-start like rate and costs
  the interests it does not mention; the app aims 20% of cold-start cards at
  it (DECISIONS #30).
- Doubly robust off-policy estimator beside SNIPS, checked against the
  simulator's ground truth.
- Studies for the remaining research options (`phase15_explore`, `_hybrid`,
  `_drift`, `_tuning`, `_alloc`), recorded as they finish.

### Recommenders
- New options, all off by default so cached results stay unchanged unless
  named: R4 `tau_halflife` and `mmr_top`; R3 `knn` (blend in kNN's ranking),
  `decay` (older labels fade) and `tune_every` (re-tune lambda on every k-th
  fit, 2.2× faster on one seed); R4+R5 `alloc` (share each batch among the
  known interests).
- R0 can be aimed by a description in words (SigLIP 2 text embedding).
- Explanations name the three liked images a card most resembles.
- R2 and R3 cache whole-pool scores per fit, and Beta interval widths are
  memoised: R2 sessions about 2.5× faster, bit-identical results.

### App
- Viewfinder interface: ambient light from each photo, size, aspect and
  palette readouts, a learning ring, keyboard echo, verdict flashes.
- `?` lists every shortcut; each verdict is announced to screen readers.
- Search by words; "describe what you like" on an empty Feed; a Saves page;
  "more of this interest" on the Profile page.
- Every page passes an axe accessibility audit in Playwright; card sizing
  fits phones.
- Plotly loads as two partial bundles instead of the full library.
- A demo mode that answers every request in the browser from a recorded
  session, for the GitHub Pages demo.

### Engineering
- CI on every push: lint, about 1,000 backend tests on CPU PyTorch, frontend
  type check, Vitest and build, Playwright, dependency audit.
- Runs on Linux and macOS as well as Windows (`just`, `scripts/*.sh`), and on
  CPU without CUDA.
- `jobs.demo_pack` and `jobs.demo`: a 2,000-image collection that installs in
  minutes, with checksum verification.
- Image routes accept only well-formed ids; CORS and binding are tested;
  `/api/health` reports free disk against the jobs' guard.
- pytest keeps only the latest failed run's temp folders.

### Docs
- README with screenshots, a demo GIF, result charts drawn from the
  simulator caches, a stat strip and a social preview image.
- Dataset card, including a review of 100 safety-screen removals and a
  200-image Megalith licence check.
- `docs/WRITEUP.md`: the project's findings in one read.
- Notebook `01_vectors_and_cosine.ipynb`.
- Contributing guide, citation file, issue templates.

### Fixed
- `backend/visualtaste/data/` was hidden from git by an unanchored `data/`
  ignore rule; a fresh clone could not import it.

## 2026-10-30 to 2026-11-01: the build

Phases 1 to 14 of the original plan, in order:

1. Environment, repository and measured GPU benchmarks.
2. Four resumable downloaders behind a licence gate: 80,036 openly licensed
   candidates.
3. Database schema and ingestion: validation, derivatives, perceptual-hash
   de-duplication, colour features.
4. SigLIP 2 and DINOv2 embeddings, 79,779 each, in resumable float16 shards.
5. FAISS search, a calibrated safety screen, k-means balancing, and a
   50,000-image selection.
6. The model interface, R0 cold start, the feed pipeline (p95 39 ms), the
   FastAPI backend and the Feed UI.
7. The evaluation harness: planted-taste simulator, replay, paired statistics.
8. R1 Rocchio and R2 kNN, concept vocabulary and explanations.
9. R3 logistic regression, by hand and with scikit-learn, with online
   retraining and atomic hot-swap.
10. R4 Thompson sampling with a Laplace posterior, MMR and an exploration
    quota; interleaving and IPS.
11. R5 multi-interest modelling with stable interest ids.
12. R6 neural ranker ensemble (manual activation only).
13. Analytics: Model Lab, Embedding Map, History, Review.
14. Hardening: backup and restore, robustness checks, the final studies.
