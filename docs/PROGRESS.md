# Progress

**Current phase:** 4 — Embeddings (Phase 2 downloads + Phase 3 ingestion running)
**Last green commit:** `589b884` — phase 1, 2026-10-30
**Running jobs:** none

Resume rule (brief §2): read `VISUALTASTE_BUILD_PROMPT.md`, then this file, then
`DECISIONS.md`, then `QUESTIONS.md`, and restart from the first unchecked box.

---

## Running jobs

| Job | Command | PID | Log | Started |
| --- | --- | --- | --- | --- |
| acquire (all 4 sources) | `python -m visualtaste.jobs.acquire` | 37440 | `logs/acquire_full.log` | 2026-10-30 19:37 |
| ingest (megalith) | `python -m visualtaste.jobs.ingest --source megalith --workers 8 --skip-dedupe` | — | `logs/ingest_megalith2.log` | 2026-10-30 20:14 |

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
- [ ] ~80k candidates in the archive; failure rate per source < 5%
      (running; smoke-tested at small targets: open_images 0%, pd12m 0%,
      unsplash_lite 0% failures)
- [x] Re-run downloads nothing new — proven by `tests/test_sources.py`
      (Megalith shard resume) and `tests/test_manifest.py`, plus a live re-run
- [ ] `tests/fixtures/` seeded from the first Megalith shard
- [ ] Per-source summary table pasted into this file

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
- [ ] Counts reconcile: downloaded = ok + rejected + near_dup

## Phase 4 — Embeddings

- [x] `vision/store.py`: float16 shards of 2,048 rows, atomic, matching IDs
- [x] `jobs/embed.py --model <key>`, incremental by missing ID
- [x] Per-image failures recorded in `embedding_errors`, batch never dies
- [ ] Merged float32 `vectors.npy` + `ids.npy` + complete `meta.json`
- [ ] Both models embedded for all `ok` candidates
- [x] Norms = 1 ± 1e-3 test (`verify_normalised`, run after every merge)
- [x] Kill-and-resume test; incremental test (add 10 → only 10 embedded)
- [ ] Real-data loading benchmark appended to BENCHMARKS.md

## Phase 5 — Search, safety, selection

- [ ] `search/index.py`: `faiss.IndexFlatIP` per model
- [ ] Embedding-level near-dups (DINOv2 cosine ≥ 0.95)
- [ ] `data/safety.py`: zero-shot screen, threshold calibrated on 200 safe images
      to ≤ 0.5% false flags
- [ ] §3 Q5 rule applied if more than 0.5% of candidates flag
- [ ] `jobs/select.py`: k-means k=400, per-cluster cap 1.5 × (50,000/400)
- [ ] 50,000 selected; per-source and per-cluster counts pasted into this file
- [ ] PCA-128 fitted (cumulative explained variance logged), `cluster64`,
      `cluster400` stored
- [ ] Notebook 01 executed with neighbour grids for both models

## Phase 6 — Backend core + Feed UI + R0

- [ ] `api/main.py` on 127.0.0.1 only; offline HF env at startup; schema check
- [ ] Routes: feed, events, undo, images (display/thumb, immutable cache), history, health
- [ ] `rec/base.py` common interface; `registry.py`; `labels.py` weights
- [ ] `rec/r0_coldstart.py`: round-robin over 64 clusters until 30 labels
- [ ] Vite + React + TS strict + Tailwind app, self-hosted fonts
- [ ] Feed page: gestures, full keymap, 3 preloaded, attribution, dwell timer,
      optimistic UI with rollback
- [ ] Playwright smoke test: 20 keyboard swipes + 1 undo, DB rows asserted
- [ ] `tsc --noEmit`, Vitest, production build all green

## Phase 7 — Logging + evaluation harness

- [ ] Impression/event/undo/supersede semantics tested
- [ ] `eval/sim.py` planted-taste simulator driving the real feed code path
- [ ] `eval/metrics.py` hand-implemented, §8 M10 examples asserted, AUC
      cross-checked against sklearn
- [ ] `eval/replay.py` time-ordered cuts + paired bootstrap 90% interval
- [ ] `eval/propensity.py` self-normalised IPS, labelled noisy/educational
- [ ] `eval/interleave.py` team-draft with sign test
- [ ] Backups tested (DB snapshot to archive, keep 14; monthly JSONL append)

## Phase 8 — R1 + R2 + explanations

- [ ] `rec/r1_rocchio.py` (α=1, β=0.5, optional 0.995^age decay)
- [ ] `rec/r2_knn.py` (mean top-5 like cosine − 0.5 × top-5 dislike, via FAISS)
- [ ] `rec/concepts.txt` (~400 terms) + `concepts.py` with the prompt template logged
- [ ] Taste in words as z-scores against the mean concept similarity
- [ ] `rec/explain.py` reason codes + one-line human sentence in the feed
- [ ] §8 M1 and M2 worked examples asserted in tests
- [ ] Simulator study R0-R2 written to EVALUATION.md

## Phase 9 — R3 + ablations

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
