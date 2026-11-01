# VisualTaste

A local, single-user image recommendation engine that starts knowing nothing
about your visual taste and learns it from swipes. Frozen pretrained image
embeddings, a personal classifier trained on your labels, Bayesian exploration,
multi-interest modelling and a small neural ranker — all measured by an honest
evaluation harness.

Everything runs on this machine. Nothing about your interactions leaves it:
no cloud, no accounts, no telemetry, no analytics, and the server binds only to
`127.0.0.1`.

The code is also meant to be read. Start at [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md)
once the build is complete.

---

## Setup

Prerequisites, all installable without administrator rights:

| Tool | Version used here | Notes |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/) | 0.13.6 | `irm https://astral.sh/uv/install.ps1 \| iex` installs to `~\.local\bin` |
| Python | 3.12 | installed *by* uv — the system Python is left alone |
| Node | 22 LTS | for the frontend |
| pnpm | 11 | `npm install -g pnpm` |
| NVIDIA driver | 576.80 (CUDA 12.9) | any driver supporting CUDA 12.x |

```powershell
# 1. A Python 3.12 virtual environment
uv python install 3.12
uv venv --python 3.12

# 2. PyTorch, FROM THE CUDA INDEX, FIRST AND SEPARATELY.
#    cu128 matches a CUDA 12.x driver. Check your driver with `nvidia-smi` and
#    the current wheel tags at https://pytorch.org/get-started/locally/.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Everything else
uv pip install open_clip_torch timm transformers huggingface_hub faiss-cpu `
  numpy scipy pandas pyarrow pillow imagehash scikit-learn umap-learn `
  fastapi uvicorn pydantic pydantic-settings httpx boto3 tqdm typer `
  pytest pytest-asyncio ruff jupyter nbconvert matplotlib

# 4. The project itself, editable, so `python -m visualtaste.jobs.*` works
uv pip install -e . --no-deps

# 5. Confirm the GPU is actually in use
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

> **Use `uv pip install`, never `uv add`.** `uv add` re-resolves the whole
> dependency set against PyPI and will silently replace the CUDA build of torch
> with the CPU one, making every embedding job roughly 30× slower. The exact
> versions this project was built and measured against are in
> [`requirements.lock`](requirements.lock).

Optional configuration: copy `.env.example` to `.env`. Every setting has a
working default, so this is only needed to override something — most usefully
`VT_ARCHIVE_ROOT`, to put the originals archive on an external drive.

## Running

```powershell
.\scripts\dev.ps1        # API on 127.0.0.1:8000, frontend on 127.0.0.1:5173
.\scripts\check.ps1      # every linter and test the build gates on
```

Then open <http://127.0.0.1:5173>.

## Building the corpus from scratch

Only needed on a new machine. Every job is resumable: stop it at any point, run
it again, and it continues where it stopped.

```powershell
python -m visualtaste.jobs.acquire               # ~80k candidates from the four sources (hours)
python -m visualtaste.jobs.ingest                # validate, derivatives, pHash, colour64
python -m visualtaste.jobs.embed --model siglip2_b16_224
python -m visualtaste.jobs.embed --model dinov2_b14
python -m visualtaste.jobs.index                 # FAISS indexes
python -m visualtaste.jobs.select                # safety screen, 50k selection, PCA-128 (both models), cluster64
python -m visualtaste.rec.concepts               # concept vocabulary cache, for interest labels and taste in words
python -m visualtaste.jobs.project               # Embedding Map projections (several minutes)
```

`python` here means `.venv\Scripts\python.exe`, or activate the venv first.

## Daily use

1. `.\scripts\dev.ps1`, then swipe in **Feed**. Keys: → or L like, ← or J
   dislike, ↓ or Space skip, S save, Z undo, I explain, O open the source page.
2. The first 30 labels are cold start (R0): a deliberate spread over the
   collection. Once you have 30 labels, including at least 5 likes and 5
   dislikes, the default rung takes over by itself. It retrains in the
   background every 20 labels.
3. **Profile** shows what the system thinks your taste is. **Model Lab** shows
   how it is doing, and holds the train, evaluate and activate controls.
4. **R6**, the neural ranker, is never switched on automatically. Once you have
   1,500 labels:

   ```powershell
   python -m visualtaste.jobs.train_neural --record
   ```

   Then activate the new version in Model Lab.

## Resetting the profile

**Profile → Reset profile…** starts a fresh profile. The old one is archived,
not deleted: its events stay in the database under its old id, and Model Lab
can still show it. Trained models are deactivated, because they describe the
old taste.

## Backups and restore

The API snapshots the database to `<ARCHIVE>/backups/db/` on shutdown and once
a day, keeping 14. It also appends new events to a monthly JSONL file in
`<ARCHIVE>/backups/events/`.

```powershell
python -m visualtaste.jobs.backup                    # back up now
python -m visualtaste.jobs.backup --list             # snapshots and their row counts
python -m visualtaste.jobs.backup --restore latest   # stop the API first
```

A restore snapshots the current database before overwriting it, so it can be
undone. It also checks that the restored row counts match the snapshot's.

On this machine the archive is on the same disk as everything else (DECISIONS
#2). Set `VT_ARCHIVE_ROOT` to an external drive to survive a disk failure.

## Re-embedding with a new model

1. Add the model to `vision/encoders.py` under a new `model_key`.
2. Run `python -m visualtaste.jobs.embed --model <key>` (incremental and
   resumable), then `python -m visualtaste.jobs.index`.
3. Run `python -m visualtaste.jobs.select --features-only` to refit the PCA and
   standardisation statistics, and `python -m visualtaste.jobs.project` for its map.
4. Restart the API, then retrain from Model Lab. Versions trained on the old
   features stay in the version table but should not be reactivated.

## Re-running the simulator studies

Every study in `docs/EVALUATION.md` is one command, and every seed is cached
under `data/eval/<study>/`, so re-running reuses what exists:

```powershell
python -m visualtaste.jobs.evaluate --study phase9_rungs --seeds 20
python -m visualtaste.jobs.evaluate --study phase10_h3 --seeds 20
python -m visualtaste.jobs.evaluate --study phase14_final --seeds 20
```

Delete a study's folder under `data/eval/` to recompute it from scratch. To run
several at once without contending for the database, give each its own copy:

```powershell
.\scripts\run_studies.ps1 -Database sim_b -Studies "phase10_h3:20:1000"
```

Model Lab can also start a study in the background.

## Measuring latency

```powershell
python -m visualtaste.jobs.latency --rungs r0,r1,r3,r4 --labels 0,100,1000,3000
```

This runs 500 `/feed/next` requests per row through the real app, and appends a
p50/p95 table to `docs/BENCHMARKS.md`. The budget is p95 under 50 ms.

## Repository map

| Path | What is in it |
| --- | --- |
| `backend/visualtaste/` | the Python package: config, db, data, vision, search, rec, eval, viz, api, jobs |
| `frontend/` | Vite + React + TypeScript app (Feed, History, Profile, Model Lab, Embedding Map, Review) |
| `notebooks/` | executed notebooks 01–08, one per maths unit |
| `docs/` | plan, progress, decisions, questions, benchmarks, evaluation, architecture, learning path |
| `tests/` | unit and integration tests, including the hand-worked maths examples |
| `scripts/` | `dev.ps1`, `check.ps1`, `run_studies.ps1` (parallel simulator studies) |
| `data/` | git-ignored: derivatives, embeddings, database, model artefacts, caches |
| `archive/` | git-ignored: write-once originals, manifests, backups |

`VISUALTASTE_BUILD_PROMPT.md` is the specification this was built from, and
`VisualTaste — ML Architecture Blueprint.md` is the research report behind it.
