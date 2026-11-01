# Progress

**Current phase:** 8 — R1 Rocchio + R2 kNN + explanations
**Last green commit:** `589b884` — phase 1, 2026-10-30
**Running jobs:** none

Resume rule (brief §2): read `VISUALTASTE_BUILD_PROMPT.md`, then this file, then
`DECISIONS.md`, then `QUESTIONS.md`, and restart from the first unchecked box.

---

## Running jobs

| Job | Command | PID | Log | Started |
| --- | --- | --- | --- | --- |
| _(none — all data jobs complete)_ | | | | |

---

## Phase 1 — Environment, repo, benchmarks

- [x] `uv` installed to user scope (0.13.6, no admin needed)
- [x] Python 3.12.16 installed by uv; venv at `.venv`
- [x] torch 2.11.0+cu128 from the CUDA index; `torch.cuda.is_available()` **True**
      on an RTX 3060 Laptop GPU (so §3 Q1 did not fire)
- [x] All §5 Python dependencies installed
- [x] Project installed editable (`visualtaste` importable; `python -m` jobs work)
- [x] `requirements.lock` frozen (167 packages; torch pinned as `2.11.0+cu128`)
- [x] Repo skeleton per §10
- [x] `.gitignore` (data, archive, logs, `*.db`, `*.npy`, `.env`, models, node_modules)
- [x] `pyproject.toml` with ruff + pytest config
- [x] `.env.example`
- [x] `config.py`: archive-root ladder (logs the rule that fired), disk guard,
      sharded path helpers, `VT_SEED=1729`
- [x] State files: PLAN, PROGRESS, DECISIONS, QUESTIONS, BENCHMARKS, EVALUATION
- [x] `vision/encoders.py`: SigLIP 2 tag read from a live
      `open_clip.list_pretrained()` (`ViT-B-16-SigLIP2`/`webli`), DINOv2 CLS token
- [x] `jobs/benchmark.py` run for both models × batch {32, 64, 128, 256};
      images/s + peak VRAM in `docs/BENCHMARKS.md`. No OOM at any size (peak
      2.8 GB of 6 GB); recommended batches 128 (SigLIP 2) and 256 (DINOv2)
- [x] CPU model and core count recorded via `Get-CimInstance Win32_Processor`
- [x] `scripts/dev.ps1`, `scripts/check.ps1`
- [x] `README.md` setup section with the torch index pinned (§5 requirement)
- [x] `git init` + first commit
- [x] `pytest -q` (39 passed) and `ruff check` / `ruff format --check` green

## Phase 2 — Acquisition

- [x] `data/http.py`: ≤4 concurrency, backoff with jitter, descriptive UA, TLS on,
      disk guard
- [x] `data/manifest.py`: resumable parquet manifest, atomic writes
- [x] `sources/megalith.py` (30,000 via the DrawThings mirror)
- [x] `sources/pd12m.py` (25,000, stratified by source organisation, MD5 verified)
- [x] `sources/openimages.py` (15,000 from the validation split, attribution kept)
- [x] `sources/unsplash_lite.py` (10,000 at `w=2048&q=85&fm=jpg`)
- [x] Licence gate: no licence URL means the image is never written
- [x] `jobs/acquire.py` driver with per-source summary
- [x] **80,036 candidates in the archive**; failure rate per source well under
      5%: megalith 0.00%, pd12m 0.00%, open_images 0.00%, unsplash_lite 0.01%
      (one transport error out of 9,984)
- [x] Re-run downloads nothing new — proven by `tests/test_sources.py`
      (Megalith shard resume) and `tests/test_manifest.py`, plus a live re-run
- [x] `tests/fixtures/` — generated synthetically instead, so each awkward
      case is explicit about what makes it awkward (see `tests/fixtures/__init__.py`)
- [x] Per-source summary table:

| source | target | stored | downloaded | rejected | failed | failure rate |
| --- | --- | --- | --- | --- | --- | --- |
| `megalith` | 30,000 | 30,000 | 30,000 | 0 | 0 | 0.00% |
| `pd12m` | 25,000 | 25,029 | 24,992 | 928 (>8 MB) | 0 | 0.00% |
| `open_images` | 15,000 | 15,002 | 14,962 | 2 (>8 MB) | 0 | 0.00% |
| `unsplash_lite` | 10,000 | 10,005 | 9,983 | 0 | 1 | 0.01% |

Total: **80,036** images, 37.1 GB of originals. The 928 PD12M rejections are
the >8 MB cap from DECISIONS.md #2 working as intended (3.6% of that source,
against the 5% predicted from the measured size distribution).

## Phase 3 — Ingestion + DB

- [x] `db/migrations/*.sql` + `init_db()` applying them in order
- [x] §7.4 schema exactly, WAL mode, foreign keys on
- [x] `db/repo.py` query layer
- [ ] `db/backup.py` (snapshot + monthly JSONL) — Phase 7, where it is tested
- [x] `data/validate.py` (verify + full load, short side ≥ 256, AR ≤ 4:1, EXIF
      transpose, animated → first frame)
- [x] `data/derivatives.py` (display 1024/q82, thumb 256/q80; pool of
      `cpu_count - 2` under `if __name__ == "__main__":`)
- [x] `data/dedupe.py` (pHash, Hamming ≤ 6, keep the highest resolution)
- [x] `data/colour.py` (CIELAB 4×4×4 → L1 → sqrt; 5 dominant colours as hex)
- [x] `jobs/ingest.py` end to end
- [x] Fixture tests: corrupt, truncated, tiny, CMYK, palette, LA, I;16,
      EXIF-rotated, animated, exact-dup, near-dup
- [x] Kill-and-resume test passes
- [x] Counts reconcile: rejections are recorded as rows with `status='rejected'`,
      so the summary balances (see `jobs/ingest.render_summary`)

## Phase 4 — Embeddings

- [x] `vision/store.py`: float16 shards of 2,048 rows, atomic, matching IDs
- [x] `jobs/embed.py --model <key>`, incremental by missing ID
- [x] Per-image failures recorded in `embedding_errors`, batch never dies
- [x] Merged float32 `vectors.npy` + `ids.npy` + complete `meta.json`
- [x] Both models embedded: **79,779 vectors each**, 0 failures
- [x] Norms = 1 ± 1e-3 test (`verify_normalised`, run after every merge)
- [x] Kill-and-resume test; incremental test (add 10 → only 10 embedded)
- [x] Real-data benchmark in BENCHMARKS.md: 54 img/s (DINOv2) and 41 img/s
      (SigLIP 2), i.e. 83-91% of the time is decode and IO, not the GPU

## Phase 5 — Search, safety, selection

- [x] `search/index.py`: `faiss.IndexFlatIP` per model, 79,779 vectors each
- [x] Embedding-level near-dups (DINOv2 cosine ≥ 0.95): 733 folded
- [x] `data/safety.py`: zero-shot screen, calibrated on 200 images, threshold 0.0478
- [x] §3 Q5 fired (789 flagged = 0.999%); asked, and the owner chose
      auto-removal — see QUESTIONS.md
- [x] `jobs/select.py`: k-means k=400, cap 187/cluster, 201 clusters capped
- [x] **50,000 selected** across 398 clusters; per-cluster min 1, median 152,
      max 172. Per-source shares, all under the 50% dominance threshold:

| source | selected | share |
| --- | --- | --- |
| `megalith` | 18,825 | 37.6% |
| `pd12m` | 15,007 | 30.0% |
| `open_images` | 10,102 | 20.2% |
| `unsplash_lite` | 6,066 | 12.1% |

Final status counts: `ok` 78,197 · `near_dup` 1,044 · `removed` 789 ·
`rejected` 1 (80,031 distinct rows)
- [x] PCA-128 fitted: 32 components explain 46.1%, 64 → 58.7%, 128 → 72.7%;
      `cluster64` and `cluster400` stored
- [ ] Notebook 01 executed with neighbour grids for both models

## Phase 6 — Backend core + Feed UI + R0

- [x] `api/main.py` on 127.0.0.1 only; offline HF env at startup; schema check
- [x] Routes: feed, events, undo, images (display/thumb, immutable cache), history, health, profile
- [x] `rec/base.py` common interface; `registry.py`; `labels.py` weights
- [x] `rec/r0_coldstart.py`: round-robin over 64 clusters until 30 labels
- [x] Vite + React + TS strict + Tailwind app, no runtime CDN requests
- [x] Feed page: gestures, full keymap, 3 preloaded, attribution, dwell timer,
      optimistic UI with rollback
- [x] Playwright smoke test: 20 keyboard swipes + 1 undo, DB rows asserted
      (5 specs, all passing)
- [x] `tsc --noEmit` and 20 Vitest tests green

## Phase 7 — Logging + evaluation harness

- [x] Impression/event/undo/supersede semantics tested
- [x] `eval/sim.py` planted-taste simulator driving the real feed code path
      (asserted: impressions and events both reach the database)
- [x] `eval/metrics.py` hand-implemented; §8 M10 asserted exactly (AUC 0.75,
      NDCG@5 0.699, M5 loss 1.313); AUC and log loss cross-checked against sklearn
- [x] `eval/replay.py` time-ordered cuts + paired bootstrap 90% interval;
      a test asserts no test event ever precedes its training cut
- [x] `eval/propensity.py` self-normalised IPS with an effective-sample-size
      gate, labelled noisy/educational in the module docstring and the output
- [x] `eval/interleave.py` team-draft with coin-flip first pick and sign test
- [x] Backups tested (DB snapshot via `conn.backup`, keep 14; append-only
      idempotent monthly JSONL that tolerates a truncated final line)

## Phase 8 — R1 + R2 + explanations

- [x] `rec/r1_rocchio.py` (α=1, β=0.5, optional 0.995^age decay)
- [x] `rec/r2_knn.py` (mean top-5 like cosine − 0.5 × top-5 dislike, via FAISS)
- [x] `rec/concepts.txt` (637 terms across 10 groups) + `concepts.py` with the
      prompt template `a photo of {}.` recorded in the cache metadata, so a
      cache built under one template is never read under another
- [x] Taste in words as z-scores against the mean concept similarity; a test
      asserts the correction separates two tastes that raw cosines do not
- [x] `rec/explain.py` reason codes + one-line human sentence in the feed.
      Sentences are *derived* from the reason and its detail, so no rung writes
      its own phrasing; a test asserts none of them flatter or overclaim
- [x] §8 M1 and M2 worked examples asserted in tests (cos 8/9 and −0.596;
      û = (0.976, −0.217) and its three scores 0.412 / 0.976 / −0.217)
- [x] Simulator study R0–R2 written to EVALUATION.md, 5 seeds × 1,000 swipes

Found and fixed while measuring, each with its own entry in `docs/DECISIONS.md`:

- [x] **The simulator was measuring nothing** (#12). The brief's
      `σ(10(cos − 0.25))` makes a user who likes 99.8% of this corpus, because
      SigLIP 2 cosines sit near 0.5 — so R0, R1 and R2 all scored 0.88. The
      offset is now calibrated per user to a 50% baseline; `--brief-offset`
      reproduces the saturated run.
- [x] **Candidate gathering cost more than the arithmetic** (#13). Both rungs
      now score the whole pool and index afterwards: 7.0 ms instead of 58.5 ms.
- [x] **R2 misses the latency budget beyond ~100 labels** (#14), reported
      rather than worked around. FAISS halves the constant; the shape stays.
- [x] **The feed never handed over from R0** (#15). Activating R1 on a fresh
      profile would have 500'd on the first request.
- [x] R0 is now `is_fitted` on construction, so it does not hand over to a
      second copy of itself

Measured, 2026-10-31 (`python -m visualtaste.jobs.evaluate --rungs r0,r1,r2
--seeds 5 --swipes 1000`):

| rung | like rate | vs R0 | interests found | scoring cost at 1,000 labels |
| --- | --- | --- | --- | --- |
| R0 | 0.454 ± 0.015 | — | 3 of 3 | 13.6 ms |
| R1 | 0.592 ± 0.100 | +0.138 (4 of 5 seeds) | 2 of 3 | 7.2 ms |
| R2 | 0.615 ± 0.084 | +0.161 (5 of 5 seeds) | 2 of 3 | 271.8 ms |

H1 is supported. The result that matters more: both learned rungs find only two
of the three planted interests, and buy their like rate by narrowing coverage
from 100% of clusters to about 40%. That is the argument for R4.

## Phase 9 — R3 + ablations

**Resume here (2026-10-31, session paused at a usage limit; uncommitted, tests green):**
- Done, untested by new tests yet:
  - `config.py`: `default_rung`, `serve_min_labels`, `serve_min_per_class`, `skip_weight`
  - `rec/base.py`: `InsufficientSignal` and `unfit_reason`
  - `rec/features.py`: standardised `sig`/`dino`/`col` blocks
  - `rec/logreg_numpy.py`
  - `rec/r3_logreg.py`: C = 1/(λS), λ tuned by time-ordered validation
  - `rec/feed.py`: the fallback is recorded in `explanation_json`, and a scoring failure falls back too
  - `load_rung("r3")`
  - DINOv2 PCA-128 and the stats files, via `jobs.select --features-only` (already run)
- Fixed: 3 unused `noqa` comments in `scripts/build_e2e_corpus.py`. This turned `check.ps1` red at pre-flight.
- Next:
  - tests (M5, agreement, C-mapping, λ ties, fallback chain)
  - DECISIONS #16 (noqa fix, standardisation) and #17 (sklearn 1.9 `l1_ratio`)
  - retrain background task and model swap, plus `/models` routes
  - simulator variants S1–S6, the evaluate cache and the §4.1 stats
  - studies, notebook 03, commit
- Plan: `C:\Users\alena\.claude\plans\synthetic-jumping-jellyfish.md`, still to be copied into PLAN.md as "Part 2 plan".

- [ ] `rec/logreg_numpy.py` hand-written weighted GD with L2; M5 example asserted
- [ ] `rec/r3_logreg.py` scikit-learn production path
- [ ] NumPy vs scikit-learn agreement test (coefficients within 1e-3)
- [ ] λ chosen by time-ordered validation over 1e-4 … 1e1
- [ ] Skip-weight comparison (0 / 0.25 / exclude) on the simulator
- [ ] Feature-block ablation (SigLIP2 / +DINOv2 / +colour)
- [ ] Online retrain every 20 labels + hot-swap, tested
- [ ] Notebook 03 executed

## Phase 10 — R4 exploration

- [ ] Laplace posterior and Cholesky Thompson draw per batch
- [ ] `rec/rerank.py` MMR λ=0.7; §8 M7 example asserted
- [ ] Exploration quota 1-of-5 decaying to 1-of-10 after 2,000 labels
- [ ] Widest-Beta cluster64 selection among clusters rated < 3 times
- [ ] Monte-Carlo propensity per impression (200 replays), approximation documented
- [ ] Interleaving toggle in Model Lab
- [ ] Simulator study R3 vs R4
- [ ] Notebook 05 executed

## Phase 11 — R5 multi-interest

- [ ] k-means on liked vectors, k = min(8, ⌊likes/25⌋)
- [ ] Log-sum-exp score, τ=0.05; §8 M8 examples asserted
- [ ] Online centroid nudge; recluster every 100 likes with Hungarian matching
- [ ] Top-3 concept label per interest
- [ ] m(x) and per-interest similarities as extra R3 features (R3+R5)
- [ ] Profile page: interest cards, taste in words, reset with confirm
- [ ] R3+R5 vs R3 on the simulator
- [ ] Notebook 06 executed

## Phase 12 — R6 neural ranker

- [ ] `rec/r6_neural.py` architecture per §7.5 with per-block LayerNorm
- [ ] Weighted BCE, then + 0.5 × BPR on same-session pairs; both selectable
- [ ] §8 M9 backprop and BPR examples asserted; hand gradients vs autograd
- [ ] `jobs/train_neural.py`: checkpoints, early stopping (patience 10), 5 seeds
- [ ] Ensemble mean = score, ensemble std = uncertainty
- [ ] Refuses to train below 1,500 labels with a clear message
- [ ] Dwell-time AUC analysis (on simulated dwell for now; stated as such)
- [ ] R6 vs R3 on the simulator with ≥ 3,000 labels
- [ ] Notebook 07 executed; activation only from Model Lab

## Phase 13 — Analytics + visualisation

- [ ] `viz/projection.py` + `jobs/project.py`: PCA-50 → UMAP, cached per model
- [ ] Remaining routes: stats (6), viz, models, eval, experiments, review, profile
- [ ] Model Lab page complete
- [ ] Embedding Map: 50k `scattergl`, colour modes, overlays, hover thumbnails,
      time slider, both models side by side
- [ ] History page; Review page
- [ ] Every dashboard number traced to a query/endpoint in ARCHITECTURE.md
- [ ] Notebooks 02, 04, 08 executed

## Phase 14 — Hardening + final evaluation

- [ ] Full simulator study, 5 seeds × 1,000 swipes, R0-R5 and R6 (≥ 3,000 labels)
- [ ] EVALUATION.md + `docs/figures/`, labelled as simulator results
- [ ] The five §7.7 hypotheses reported honestly, hold or not
- [ ] `/feed/next` p95 measured and recorded (target < 50 ms)
- [ ] README: setup with the pinned torch index, run, daily use, reset,
      backup/restore, re-embedding
- [ ] LEARNING_PATH.md mapped to M1-M10 and the notebooks
- [ ] ARCHITECTURE.md: components, Mermaid data flow, `/feed/next` path, writers
- [ ] Fresh-clone setup verified in a temp venv from the README alone
- [ ] `scripts/check.ps1` green
- [ ] Final empty real profile with R0 active

---

## Future work

Converted from any TODO left at the end of the build. Empty so far.
