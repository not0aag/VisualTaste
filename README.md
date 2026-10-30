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

## Repository map

| Path | What is in it |
| --- | --- |
| `backend/visualtaste/` | the Python package: config, db, data, vision, search, rec, eval, viz, api, jobs |
| `frontend/` | Vite + React + TypeScript app (Feed, History, Profile, Model Lab, Embedding Map, Review) |
| `notebooks/` | executed notebooks 01–08, one per maths unit |
| `docs/` | plan, progress, decisions, questions, benchmarks, evaluation, architecture, learning path |
| `tests/` | unit and integration tests, including the hand-worked maths examples |
| `scripts/` | `dev.ps1`, `check.ps1` |
| `data/` | git-ignored: derivatives, embeddings, database, model artefacts, caches |
| `archive/` | git-ignored: write-once originals, manifests, backups |

`VISUALTASTE_BUILD_PROMPT.md` is the specification this was built from, and
`VisualTaste — ML Architecture Blueprint.md` is the research report behind it.

<!-- Daily use, reset, backup/restore and re-embedding are documented in Phase 14. -->
