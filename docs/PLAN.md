# VisualTaste — Execution Plan

## Context

`VISUALTASTE_BUILD_PROMPT.md` specifies a local, single-user image recommendation
engine that starts knowing nothing about Alen's visual taste and learns it from
swipes — frozen pretrained embeddings, a personal classifier, Bayesian
exploration, multi-interest modelling, a small neural ranker, and an honest
evaluation harness. The code is also a textbook: Alen reads it afterwards to
learn the maths, so every ML module must teach.

`C:\VisualAI` currently holds only the build brief and the 91 KB research
blueprint. Everything else is built from scratch. The brief wins on any conflict
with the blueprint.

Because no real labels can exist until Alen swipes, every learning component is
developed and validated against the planted-taste simulator (§7.7). The build
ends with the system ready for his first swipe, every rung implemented, tested
and switchable.

This plan is mirrored to `docs/PLAN.md` as the first task of Phase 1, which is
where it is maintained from then on (the brief's §2 state-file contract).

---

## Environment (measured 2026-10-30, read-only probes)

| Fact | Value | Consequence |
| --- | --- | --- |
| GPU | RTX 3060 Laptop, **6 GB VRAM**, driver 576.80, CUDA 12.9 | No Q1. Install torch from `cu128`. 6 GB is the real constraint — batch size comes from the Phase 1 benchmark, not the spec. |
| CPU | AMD Ryzen 9 5900HX, 8C/16T | Derivative pool = 14 workers (cores−2 → use logical−2). |
| RAM | 15.4 GB | 50k × 768 × 2 models float32 = 307 MB resident: fine. UMAP on 50k is fine. k-means k=400 needs care, not much. |
| Disk | **C: only**, 135.9 GB free of 453.5 | Q2 asked and answered: internal archive, full 80k pool. |
| Python | 3.13 + 3.11 present, **no 3.12** | `uv python install 3.12` provides it; system Pythons untouched. |
| `uv` | **MISSING** | Install via the official user-scope standalone script (no admin). |
| `winget` | **MISSING** | Irrelevant — nothing needs it. Node 22.21.0 and pnpm 11.17.0 already present. |
| git | 2.45.2, **repo not initialised** | `git init` in Phase 1. Never add a remote, never push. |

### Source availability (verified live)

- **Megalith-10M** — card confirms the image mirror `drawthingsai/megalith-10m` is
  live. Use it; the Flickr-URL fallback stays implemented but unused.
- **PD12M** — images are plain unsigned HTTPS
  (`https://pd12m.s3.us-west-2.amazonaws.com/images/<uuid>.jpeg`); metadata
  parquet carries `license`, `hash` (MD5) and `source` (640 orgs → stratify on it).
  No boto3 needed here.
- **Open Images V7** — validation split via the public `open-images-dataset`
  bucket; boto3 unsigned. Attribution CSVs give `Author`/`License`/`OriginalURL`.
- **Unsplash Lite** — 25k photos, 1.5 GB, "free for commercial and
  non-commercial usage", routed through `github.com/unsplash/datasets`. Need
  10k. If a login wall appears → Q4.

---

## Decisions taken up front (go into `docs/DECISIONS.md`)

1. **Project root = `C:\VisualAI`** — the brief's §0.4 "use the directory you
   were started in" clause, not `C:\dev\visualtaste`.
2. **Archive = `C:\VisualAI\archive`** — §6.1 rule 5 fired (no external drive,
   <150 GB free); Alen chose the internal fallback with the full 80k pool.
   Mitigations, both his explicit conditions:
   - a **disk guard** in `data/http.py` + every job: pause and log when C: free
     space drops below 25 GB, never crash mid-write;
   - **skip individual originals > 8 MB** (PD12M museum scans are the only
     multi-MB source) — counted and reported in the per-source summary.
   - Accepted risk: DB snapshots live on the same physical drive as the
     originals, so there is no off-drive backup copy. Recorded, not hidden.
3. **torch from `--index-url https://download.pytorch.org/whl/cu128`**, installed
   with `uv pip install` (never `uv add`, which would swap in the CPU build).
4. **Megalith via the DrawThings mirror**, per the live card.
5. **DINOv2 CLS token** (not mean-pooled patches) for `dinov2_b14`, per §7.3.

---

## Phase plan

Each phase ends green: `pytest -q` + `ruff check` + `ruff format --check`
(+ `tsc --noEmit`, Vitest from Phase 6) pass, the §9 DoD holds, `docs/PROGRESS.md`
is ticked, and a local commit `phase N: <summary>` is made. Long jobs run in the
background to `logs/<job>_<timestamp>.log` while unblocked work continues.

### Phase 1 — Environment, repo, benchmarks

- Install `uv` (user scope) → `uv python install 3.12` → `uv venv`.
- `uv pip install` the §5 dependency list; torch trio from the cu128 index first
  so nothing downgrades it. Assert `torch.cuda.is_available()` is True.
  `uv pip freeze > requirements.lock`.
- `git init`; `.gitignore` (`data/`, `archive/`, `logs/`, `*.db`, `*.npy`,
  `.env`, `models/`, `node_modules/`).
- Repo skeleton exactly per §10. `pyproject.toml` with ruff config.
- `backend/visualtaste/config.py` — pydantic-settings, `pathlib` everywhere,
  the §6.1 root-detection ladder with the rule that fired logged, `VT_SEED=1729`,
  and the disk guard helper.
- All six `docs/` state files, `docs/PLAN.md` first.
- `jobs/benchmark.py` — synthetic-tensor throughput for both models × batch
  {32, 64, 128, 256}: images/s + peak VRAM, expecting OOM at the top end on 6 GB
  (recorded as a result, not a failure). CPU/GPU/driver captured via
  `Get-CimInstance Win32_Processor` + `nvidia-smi` → `docs/BENCHMARKS.md`.
- `scripts/dev.ps1`, `scripts/check.ps1`.

### Phase 2 — Acquisition (`jobs/acquire.py`)

- `data/http.py` — shared async httpx client: ≤4 concurrency, exponential
  backoff with jitter, `VisualTaste/0.1 (personal research; local use)` UA, TLS
  always verified, per-attempt logging, disk guard. On a persistent 403/429/login
  wall: stop, never route around it → Q4.
- `data/manifest.py` — parquet manifest per source (`candidate_id`, status,
  attempts, error reason, licence URL, bytes); atomic writes via temp +
  `os.replace`; re-running skips completed rows.
- Four downloaders in `data/sources/`: `megalith.py` (30k, mirror shards),
  `pd12m.py` (25k, stratified by `source` org, MD5 verified), `openimages.py`
  (15k, validation split + attribution CSVs), `unsplash_lite.py`
  (10k, `photos.tsv*` → `photo_image_url?w=2048&q=85&fm=jpg`).
- **Licence gate at download time**: no licence URL → never written.
- Per-source summary (requested / downloaded / failed by reason / skipped as
  >8 MB) into `docs/PROGRESS.md`. Failure rate <5% per source or investigate.
- `tests/fixtures/` — a few hundred CC0 images from the first Megalith shard, so
  every later phase has something to run against while downloads continue.

### Phase 3 — Ingestion + DB (`jobs/ingest.py`)

- `db/` — numbered `.sql` migrations applied by `init_db()`, the §7.4 schema
  verbatim, WAL + foreign keys on, `repo.py` query layer, `backup.py`
  (`conn.backup` → archive, keep 14, + monthly events JSONL).
- `data/validate.py` (verify + full load, short side ≥256, AR ≤4:1, EXIF
  transpose, animated → first frame), `derivatives.py` (display 1024/q82 +
  thumb 256/q80 WebP, 14-worker pool under `if __name__ == "__main__":`),
  `dedupe.py` (pHash, Hamming ≤6, keep highest resolution), `colour.py`
  (CIELAB 4×4×4 → L1 → sqrt Hellinger → `colour64.npy`; k=5 dominant hex).
- `image_id` = first 16 hex of SHA-256 of original bytes; exact dups collapse to
  one row with all sources in `image_sources`.
- Tests: corrupt, truncated, tiny, CMYK, palette, LA, I;16, EXIF-rotated,
  animated, exact-dup, near-dup fixtures; **kill-and-resume test**; counts
  reconcile (downloaded = ok + rejected + near_dup).

### Phase 4 — Embeddings (`jobs/embed.py`)

- `vision/encoders.py` — `siglip2_b16_224` via `open_clip` (exact tag taken from
  a live `open_clip.list_pretrained()`, logged) and `dinov2_b14` via
  `transformers` (CLS token). `eval()`, `inference_mode()`, fp16 autocast.
- `vision/store.py` — float16 shards of 2,048 rows written atomically with
  matching IDs → merged float32 `vectors.npy` + `ids.npy` + `meta.json`
  (model, tag, dim, preprocessing, date, torch/CUDA versions).
- Incremental: only IDs missing for that `model_key`. Per-image failures →
  `embedding_errors`, never a dead batch.
- Tests: norms = 1 ± 1e-3, kill-and-resume, incremental (add 10 → 10 embedded).
- Real-data loading benchmark appended to `docs/BENCHMARKS.md`.

### Phase 5 — Search, safety, selection

- `search/index.py` — `faiss.IndexFlatIP` per model.
- Embedding near-dups: DINOv2 cosine ≥0.95 → `near_dup`.
- `data/safety.py` — SigLIP 2 zero-shot unsafe-vs-neutral prompts, threshold
  calibrated on 200 obviously-safe images to ≤0.5% false flags; flagged →
  `quarantined`. If >0.5% of all candidates flag → **Q5**.
- `jobs/select.py` — k-means k=400, uniform within-cluster sampling capped at
  1.5 × (50 000/400), random fill → `selected=1`. Per-source and per-cluster
  counts into `docs/PROGRESS.md`; rebalance if one source exceeds 50%.
- PCA-128 (cumulative explained variance logged), `cluster64`, `cluster400`.
- Notebook `01_vectors_and_cosine` executed, with neighbour grids for both models.

### Phase 6 — Backend core + Feed UI + R0

- `api/` — FastAPI on `127.0.0.1:8000` only, all routes under `/api`, Pydantic
  models throughout. This phase: feed, events, undo, images (display/thumb with
  immutable cache headers), history, health. `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` set at startup; vectors + FAISS + active model loaded
  at startup; schema version verified.
- `rec/base.py` common interface (`fit`/`score`/`explain`/`save`/`load`/`params`),
  `registry.py`, `labels.py` (like 1.0 / save 2.0 / dislike 1.0 / skip 0.25,
  undone ignored, latest non-undone event wins), `r0_coldstart.py`
  (round-robin over the 64 clusters until 30 labels).
- Frontend: Vite + React + TS strict + Tailwind, self-hosted fonts, no runtime
  CDN. Feed page per §7.9 — gestures via `@use-gesture/react` (commit at 30%
  width or fast flick, tilt/tint while dragging), full keymap, next 3 preloaded,
  attribution line on every image, dwell timer on `performance.now()` paused on
  blur/visibilitychange and capped at 30 s, time-to-first-interaction,
  optimistic UI with rollback.
- Playwright smoke test: 20 keyboard swipes + 1 undo, asserting DB rows.

### Phase 7 — Logging + evaluation harness (before any learned model)

- impressions/events/undo/supersede semantics tested end to end.
- `eval/sim.py` — planted-taste simulator: 3 hidden centroids from `cluster64`,
  p(like) = σ(10·(max_k cos − 0.25)), saves at p>0.9 w.p. 0.3, skips w.p. 0.1,
  optional drift; drives the **real feed code path** on an `is_simulated=1`
  profile, which is never mixed into real training data or dashboards.
- `eval/metrics.py` — hand-implemented log loss, AUC, NDCG@K (save=2, like=1),
  P@K, R@K, hit rate, intra-batch diversity, coverage, novelty, exploration
  yield with Beta intervals, calibration bins. Unit tests use the §8 M10
  examples verbatim (AUC 0.75, NDCG@5 ≈ 0.699) and cross-check AUC against
  sklearn.
- `eval/replay.py` (cut every 100 events after 200, paired bootstrap 90% CI),
  `propensity.py` (SNIPS, labelled noisy/educational), `interleave.py`
  (team-draft, coin-flip first pick, per-card arm attribution, sign test).
- Backups tested (DB snapshot + JSONL append).

### Phase 8 — R1 Rocchio + R2 kNN + explanations

- `r1_rocchio.py` (α=1, β=0.5, optional 0.995^age decay), `r2_knn.py`
  (top-5 cosine to likes − 0.5 × top-5 to dislikes via FAISS).
- `rec/concepts.py` + `concepts.txt` (~400 terms), SigLIP 2 text encoder with
  its prompt template logged; "taste in words" as **z-scores** against the mean
  concept similarity (corrects the modality gap).
- `rec/explain.py` + `feed.py` reason codes: `cold_start`, `similar_likes`
  (3 nearest liked IDs), `interest:<k>`, `explore`, `model_confident`.
- Tests use the §8 M1/M2 examples verbatim (cos ≈ 0.8889, −0.596; û ≈
  (0.976, −0.217) and its three scores).
- Simulator study for R0–R2 → `docs/EVALUATION.md`, reported honestly.

### Phase 9 — R3 logistic regression (+ ablations)

- `logreg_numpy.py` — hand-written batch GD with sample weights + L2, docstring
  per §8 (intuition, formula, variables, worked example); test asserts the M5
  example exactly (p ≈ 0.731, loss ≈ 1.313, one η=0.5 step → w ≈ (1.708, −1.219),
  b ≈ −0.366, p ≈ 0.567).
- `r3_logreg.py` — sklearn `LogisticRegression(C=1/λ)` with `sample_weight`
  (production); **agreement test** vs the NumPy version within 1e-3.
- λ by time-ordered validation over 1e-4…1e1; skip-weight comparison
  (0 / 0.25 / exclude); feature-block ablation (SigLIP2 / +DINOv2 / +colour).
- Online retrain every 20 labels as a FastAPI background task + hot-swap, tested.
- Notebook `03_logistic_regression_by_hand` executed.

### Phase 10 — R4 exploration

- Laplace posterior (Σ⁻¹ = λI + Σ sᵢpᵢ(1−pᵢ)xᵢxᵢᵀ), Cholesky Thompson draw per
  batch, `rerank.py` MMR λ=0.7 (test asserts the §8 M7 numbers: b → 0.331,
  c → 0.470, c picked next), exploration quota 1-of-5 decaying to 1-of-10 after
  2 000 labels, drawn from the widest-Beta `cluster64` among clusters rated <3
  times.
- Monte-Carlo propensity: 200 offline replays of sampling + re-rank per batch,
  fraction of runs selecting that image at that slot; the approximation is
  documented in the docstring.
- Interleaving toggle in Model Lab; simulator study R3 vs R4; notebook 05.
- `/feed/next` p95 measured here as an early warning against the <50 ms target.

### Phase 11 — R5 multi-interest

- k-means on liked vectors, k = min(8, ⌊likes/25⌋); soft-max
  m(x) = τ·log Σ exp(cos/τ), τ=0.05 (test asserts the §8 M8 log-sum-exp values
  0.400 and 1.294, and the 1-D k-means converging to 1.5 / 9.5); online nudge
  c_k ← c_k + 0.05(x − c_k); full recluster every 100 likes with Hungarian
  matching for stable IDs; top-3 concept labels per interest.
- m(x) + per-interest similarities as extra R3 features ("R3+R5").
- Profile page (interest cards: 6 central images, 3 concept words, like count,
  Beta interval; taste in words; reset with confirm).
- R3+R5 vs R3 on the simulator; notebook 06.

### Phase 12 — R6 neural ranker

- `r6_neural.py` — [SigLIP2 768 | DINOv2 768 | colour 64 | 8 profile features]
  with per-block LayerNorm → 1608→256 GELU Drop(0.3) → 256→64 GELU Drop(0.3)
  → 64→1. Weighted BCE first, then + 0.5 × BPR on same-session (like, dislike)
  pairs, both selectable. Tests assert the §8 M9 numbers (grads ≈ −0.134 /
  −0.538, post-step p ≈ 0.753; BPR loss ≈ 0.371) and check hand gradients
  against autograd.
- `jobs/train_neural.py` — AdamW lr 1e-3 wd 1e-2, batch 256, ≤200 epochs, early
  stopping on the time-ordered last 15% (patience 10), 5 seeds → ensemble mean =
  score, std = uncertainty. Never trains in the API process; refuses below 1 500
  labels with a clear message. Activation only from Model Lab.
- Dwell-time AUC analysis implemented (runs on simulator dwell now — stated as
  such; dwell stays a logged signal, never a label).
- vs R3 on the simulator with ≥3 000 labels; notebook 07.

### Phase 13 — Analytics + visualisation

- `viz/projection.py` + `jobs/project.py` — PCA-50 → UMAP (cosine,
  n_neighbors=30, min_dist=0.1), cached per model; `umap.transform` for the
  preference-vector time slider.
- Remaining API routes: stats (summary / learning-curve / clusters / diversity /
  calibration / exploration), viz, models, eval, experiments, review, profile.
- Model Lab, Embedding Map (Plotly `scattergl`, 50k points, colour by
  cluster/source/score, like/dislike/save overlays, hover thumbnails, SigLIP 2
  vs DINOv2 side by side), History, Review pages.
- Every dashboard number traced to a query/endpoint in `docs/ARCHITECTURE.md`.
- Notebooks 02, 04, 08 executed.

### Phase 14 — Hardening + final evaluation

- Full simulator study: R0–R5 at 5 seeds × 1 000 swipes, R6 with ≥3 000 labels
  → `docs/EVALUATION.md` + `docs/figures/`, explicitly labelled simulator
  results, with the five §7.7 hypotheses reported honestly whether or not they
  hold.
- `/feed/next` p95 measured and recorded; profile and fix if over 50 ms.
- `README.md` (setup with the pinned torch index, run, daily use, reset,
  backup/restore, re-embedding), `docs/LEARNING_PATH.md` (reading order mapped
  to M1–M10 + notebooks), `docs/ARCHITECTURE.md` (Mermaid data flow,
  `/feed/next` request path, per-table writers).
- Fresh-clone setup verified in a temp venv from the README alone, config
  pointed at the existing data (no re-download).
- Remaining TODOs converted to "Future work" in `docs/PROGRESS.md`.
- Final empty real profile with R0 active.

---

## Parallelism

Phase 2–5 jobs are hours of GPU/network time. While they run: the Phase 6
frontend, the Phase 7 harness, and the R1–R3 code are built against
`tests/fixtures/`. Nothing idles on a poll loop; every job is resumable, so a
reboot costs minutes.

## Verification

- `scripts/check.ps1` — ruff check + format check, `pytest -q`, `tsc --noEmit`,
  Vitest, Playwright smoke. Green at every phase commit; the gate for §12.
- Resumability proven by kill-and-resume tests for ingest and embed, not by
  assertion.
- Maths proven by the §8 worked examples as literal test fixtures.
- Every number in the docs carries its command and date. Anything not actually
  run is labelled "estimate".

## Finish

One final message per §12 (done / done-with-blockers, `scripts/dev.ps1` + URL,
per-source dataset summary, one-line simulator headline, blockers, and
`docs/LEARNING_PATH.md` as the starting point). Then stop.
