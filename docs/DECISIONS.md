# Decision log

Numbered, append-only. Every non-trivial choice the build brief did not make for
us, with the options considered and why one won. Referenced from code comments
as `docs/DECISIONS.md #N`.

---

## 1. Project root is `C:\VisualAI`, not `C:\dev\visualtaste`

**Date:** 2026-10-30
**Options:** (a) create `C:\dev\visualtaste` and move there; (b) build in the
directory the session started in.
**Chosen:** (b).
**Why:** The brief's §0.4 says "use the directory you were started in" when the
project already exists elsewhere. The session started in `C:\VisualAI`, which
already held the brief and the blueprint. Moving would orphan those two files
and buy nothing. Shorter path is also a small win against Windows' 260-character
limit.

---

## 2. Originals archive on the internal drive, full 80k candidate pool

**Date:** 2026-10-30
**Trigger:** The §6.1 detection ladder reached **rule 5**: no non-system drive
is attached, and `C:` had 135.9 GB free, below the 150 GB that rules 3 and 4
require. That is the brief's §3 **Q2** condition, so the owner was asked.
**Options offered:** (a) internal archive with the full 80k pool; (b) internal
archive with a reduced ~60k pool; (c) wait for an external drive.
**Chosen:** (a), by the owner.
**Conditions attached to the choice, both implemented:**

- **Disk guard.** `config.check_disk_headroom()` raises `DiskGuardTripped` when
  free space on the archive volume falls below `VT_MIN_FREE_GB` (default 25).
  Long jobs call it between batches, so they stop at a resumable boundary rather
  than dying mid-write.
- **Large-original skip.** Originals above `VT_MAX_ORIGINAL_MB` (default 8) are
  not downloaded. PD12M museum scans are the only multi-MB source, and at a
  1024 px display size there is nothing to gain from a 40 MB TIFF-derived JPEG.
  Skipped files are counted and reported in the per-source summary, not silently
  dropped.

**Accepted risk, recorded rather than hidden:** DB snapshots and the events
JSONL live on the same physical disk as the originals, so there is no off-drive
backup copy. A drive failure loses the interaction history. The blueprint's
recommendation (a 2 TB external HDD) remains the fix; `VT_ARCHIVE_ROOT` in
`.env` is all that is needed to adopt one later, and rule 2 of the ladder will
find it automatically if it is named `VisualTasteArchive`.

**Budget check (estimate, from the blueprint's §5 table):** ~82 GB originals +
~12 GB WebP derivatives + ~2 GB model weights + ~6 GB venv. After the Python
environment was installed, free space measured 126.8 GB, leaving roughly 25-30
GB of headroom once the data lands -- which is exactly what the disk guard
protects.

---

## 3. PyTorch from the `cu128` wheel index

**Date:** 2026-10-30
**Options:** `cu126`, `cu128`, `cu130`, or the default (CPU) index.
**Chosen:** `cu128`, installed first and separately with
`uv pip install --index-url https://download.pytorch.org/whl/cu128`.
**Why:** `nvidia-smi` reports driver 576.80 / CUDA 12.9, which runs any CUDA 12.x
build. `cu128` is the mainstream build for this driver generation. Resolved to
**torch 2.11.0+cu128**, and `torch.cuda.is_available()` is True on an RTX 3060
Laptop GPU, so §3 **Q1** did not fire.
Installed with `uv pip install`, never `uv add`: the latter re-resolves the whole
dependency set from PyPI and silently replaces the CUDA build with the CPU one.

---

## 4. Megalith images come from the `drawthingsai/megalith-10m` mirror

**Date:** 2026-10-30
**Options:** (a) the community image archive linked from the dataset card;
(b) download from the 9.58 M Flickr URLs in the original parquet.
**Chosen:** (a), with (b) implemented as a fallback.
**Why:** The live dataset card for `madebyollin/megalith-10m` confirms the
DrawThings mirror hosts the actual image files; the primary repo holds only
`url` / `url_highres` / `url_source` columns. Pulling a few shards from a
Hugging Face mirror is one polite, resumable, rate-limit-friendly transfer
instead of 30,000 individual requests to Flickr. The Flickr path stays in the
code because the brief names it, and because a mirror can disappear.

---

## 5. DINOv2 uses the CLS token, not mean-pooled patch tokens

**Date:** 2026-10-30
**Options:** (a) `last_hidden_state[:, 0]` (CLS); (b) mean of the patch tokens;
(c) concatenate both (1536 dims).
**Chosen:** (a).
**Why:** The brief requires the choice to be logged. CLS is the token DINOv2's
self-supervised objective trains to summarise the whole image, and it is what
the paper's own retrieval evaluations use. Our use for DINOv2 is instance-level
appearance similarity and near-duplicate detection, which is precisely that
retrieval setting. (c) would double the storage for a marginal gain and would
break the clean "both encoders are 768-dim" assumption that the neural ranker's
input block sizes rely on.

---

## 6. Package installed editable, `backend/` as the package directory

**Date:** 2026-10-30
**Options:** (a) `sys.path` manipulation in each entry point; (b) `pip install -e .`
with hatchling mapping `backend/visualtaste`; (c) move the package to the repo root.
**Chosen:** (b), plus `pythonpath = ["backend"]` in the pytest config so tests
work even in a fresh checkout before the editable install.
**Why:** The brief's §10 layout puts the package under `backend/`, and its job
commands are `python -m visualtaste.jobs.ingest`, which needs the package
importable. An editable install is the standard way to get that without path
hacks. Installed with `--no-deps`, because the dependency set is managed
directly by `uv pip install` (see #3) and a re-resolve would fight the CUDA
torch build.
