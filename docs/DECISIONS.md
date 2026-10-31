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

---

## 7. Open Images integrity is checked against the S3 ETag, not `OriginalMD5`

**Date:** 2026-10-30
**Discovered by measurement, not documentation.** The attribution CSV gives an
`OriginalSize` and an `OriginalMD5` per image, so the obvious design is to
verify downloads against them. Checked against the bucket first:

| object | S3 size | CSV `OriginalSize` | MD5 match |
| --- | --- | --- | --- |
| `validation/fe600639ac5f36c1.jpg` | 86,534 | 242,145 | no |
| `validation/ba82c70cc6cdf449.jpg` | 133,986 | 1,658,910 | no |

Open Images redistributes its **own re-encoded copies** (long side ~1024 px);
the CSV describes the Flickr original each was made from. Verifying against
`OriginalMD5` would have rejected every single image, and the failure would have
looked like a corrupt bucket rather than a mismatched expectation.

**Chosen:** verify against S3's `ETag`, which for these single-part objects is
the MD5 of the stored bytes. `OriginalMD5` and `OriginalURL` are still kept, as
§7.1 asks, as provenance of the upstream file -- in a `upstream_md5` manifest
column that is explicitly never used as a verifier.

Also decided here: rows with a non-zero `Rotation` are skipped. Open Images
expects the consumer to apply that rotation, which is not part of this
project's ingestion rules, and the validation split has 41,620 candidates for a
15,000 target -- so skipping 476 of them is cheaper than handling a case that
would otherwise show images on their side.

Also: an Open Images row with no `Author` is dropped. These are CC BY 2.0, which
requires attribution; an image we cannot credit is one we will not display.

---

## 8. Megalith records collection-level provenance as its licence URL

**Date:** 2026-10-30
**The problem.** The brief requires a licence URL per image and excludes images
without one. Megalith-10M publishes **no per-image licence field**: the original
parquet has `url`, `url_highres`, `url_source`, and the mirror's has `caption`,
`url`, `key`, `sha256` and dimensions. The guarantee is collection-level -- the
dataset was built by filtering Flickr to "no known copyright restrictions", CC0,
Public Domain Mark and US Government Work.

**Options considered:**

1. Record the dataset's documented provenance as the licence URL, and store the
   Flickr photo page as `source_url` so any individual image stays checkable by
   a human in one click.
2. Stamp every image with a specific licence, e.g. CC0.
3. Query the Flickr API for each image's true licence.
4. Drop Megalith and redistribute its 30,000 to the other three sources (Q4).

**Chosen:** 1.
**Why:** 2 is the tempting one and the wrong one -- it would record a precise
claim we have not verified, which looks like evidence while being a guess, and
this is exactly the kind of number the brief's §0.8 forbids fabricating. 3 needs
an API key and 30,000 calls, outside what this brief authorises. 4 would discard
the largest and cleanest block of the corpus over a metadata limitation rather
than a licensing problem.

The honest position is recorded rather than hidden: the licence URL points at
the provenance statement, the per-image Flickr page is stored alongside it, and
the module docstring in `data/sources/megalith.py` says plainly that this is
collection-level and why.

---

## 9. Ingestion tuning: WebP effort 2, cheaper swatch k-means, one thread per worker

**Date:** 2026-10-30
**Trigger:** the first real ingestion run managed about 4 images/second, which
projected to roughly two hours for Megalith alone. Profiled rather than guessed,
on 10-15 real Megalith images.

| Finding | Measured | Change |
| --- | --- | --- |
| WebP `method=4` (Pillow's default) | 412 ms/encode, 122.8 KB | `method=2`: 211 ms, 125.3 KB |
| `KMeans(n_init=4)` over 4,096 px for the display swatches | 1,350 ms | 1,024 px, `n_init=1`, `max_iter=25`: 139 ms |
| `np.unique(lab, axis=0)` guarding `k` | 14 ms | pack RGB into one integer: negligible |
| 8 worker processes x 16 OpenMP threads | thrashing | pin `OMP_NUM_THREADS=1` per worker |

Also found: `import sklearn` costs 9.2 seconds, which cProfile attributes to the
first image and made the per-image cost look twice as bad as it is. It is paid
once per worker process, so it is irrelevant across 30,000 images -- worth
knowing before optimising the wrong thing.

**Result, measured after the change: 40 images/second, a 10x improvement.**
Megalith's 30,000 ingested in about 12 minutes instead of two hours.

The trade accepted: 2% larger display WebP files (about 200 MB across 80,000
images) in exchange for roughly half an hour of wall clock, and display-only
colour swatches computed from 1,024 sampled pixels instead of 4,096. The
64-bin colour *descriptor* -- the thing the models actually learn from -- is
unchanged and still computed from every pixel of the thumbnail.

---

## 10. Near-duplicate grouping uses connected components, not one-group-per-pass

**Date:** 2026-10-31
**Found by:** running `jobs.index` twice and noticing the second run folded
more images instead of nothing.

**The bug.** The original algorithm walked the images and, for each one,
claimed every *unclaimed* image similar to it. That resolves only one layer of
overlap per pass. Given a chain where A resembles B and B resembles C but A
does not resemble C, A claims B, and then C is skipped because its only
partner is already claimed. C survives, and a later run folds it. Successive
runs folded 48 images, then 12, then more — slowly converging.

Nothing in the database was *wrong*, but "run the job twice and the second run
is a no-op" is a property the whole resumability contract (§6.2) rests on, and
this quietly did not have it.

**The fix.** Similarity is now treated as an undirected graph: collect every
pair above the threshold, then take connected components with union-find, and
collapse each component to its highest-resolution member. One pass, and a
second pass finds nothing. Applied to both the pHash and the embedding passes,
which shared the flaw.

**Effect on the corpus.** The correct algorithm folds *more* in a single pass,
because it resolves whole components rather than pairs:

| pass | before the fix | after |
| --- | --- | --- |
| pHash (Hamming ≤ 6) | 251 | 311 |
| DINOv2 (cosine ≥ 0.95) | 48 | 733 |

The near-duplicate state was rebuilt from scratch with the fixed algorithm and
verified stable: re-running both passes folds 0.

**Tests added:** a three-image chain whose endpoints are deliberately *not*
similar to each other (so the test exercises transitivity rather than a
clique), plus an idempotency test for each pass.

---

## 11. Database connections are opened with `check_same_thread=False`

**Date:** 2026-10-31
**Found by:** the Playwright smoke test, via a 500 on `POST /api/events`.

Swiping quickly produced intermittent 500s:

    sqlite3.ProgrammingError: SQLite objects created in a thread can only be
    used in that same thread

FastAPI runs synchronous endpoints in a thread pool, and — the part that bites
— a generator dependency's setup and its teardown can land on *different* pool
threads. So a connection opened for a request was being closed from another
thread.

**Chosen:** `check_same_thread=False`, with the reasoning written at the call
site. The invariant Python's guard protects is still held by construction: one
connection per request, never shared between concurrent requests. The threads
touch it one after another, not simultaneously. Concurrent *writes* are a
different mechanism, handled by WAL plus `busy_timeout`.

This is the kind of bug that only appears under real concurrency, which is
exactly why the brief asks for an end-to-end test rather than trusting unit
tests.

---

## 12. The simulator's like-probability offset is calibrated per user, not fixed at 0.25

**Date:** 2026-10-31
**Found by:** the first R0/R1/R2 study, in which all three rungs scored a like
rate of 0.88 — including the one that picks by round-robin over clusters.

The brief (§7.7) specifies the simulated user as

    p(like | x) = sigma(10 (max_k cos(c_k, x) - 0.25))

Taken literally on this corpus, that is a user who likes everything. Measured
on the 50,000-image pool:

| quantity | mean | p5 | p50 | p95 |
| --- | --- | --- | --- | --- |
| image-to-image cosine | 0.501 | 0.388 | 0.497 | 0.636 |
| `max_k cos` to 3 planted centroids | 0.680 | 0.573 | 0.675 | 0.804 |

with `p(like)` averaging **0.983** and **99.8%** of the pool above `p = 0.9`.

The cause is the geometry of the embedding space. SigLIP 2 vectors occupy a
narrow cone rather than the sphere, so two *unrelated* photographs already sit
at cosine 0.5 — the well-known cone effect in contrastive models. An offset of
0.25 is therefore below the entire distribution, the sigmoid saturates, and
every rung is graded against a user who says yes to almost anything. The study
measures nothing: a random recommender would have scored the same 0.88.

**Chosen:** keep the formula and the slope, calibrate the offset per simulated
user to the median of the affinity *that user* feels for this corpus
(`eval/sim.calibrate_offset`), giving a 50% baseline like rate. `LOGIT_OFFSET`
still holds the brief's 0.25 and `make_user` still defaults to it, so the
brief's user remains available and reproducible — `jobs.evaluate --brief-offset`
runs it, and `docs/EVALUATION.md` shows both, because the saturated run is the
evidence that the calibration was needed.

**Why per user rather than one global number:** a taste centred on a large
cluster feels higher affinity across the whole corpus than one centred on a
small cluster. A single offset would make some seeds systematically pickier
than others, and that difference would show up in the results as if it were a
property of the model under test.

**Why the slope stays at 10:** over the measured spread of `max_k cos` (p5
0.573 to p95 0.804) a slope of 10 makes `p(like)` run from about 0.26 to 0.79 —
a soft boundary with real label noise, which is what §7.7 asks for. Raising it
would make the user more decisive and the task easier.

**What this does not fix:** the simulator's truth is still cosine to three
centroids, so it still flatters Rocchio and logistic regression. Calibration
makes the comparison possible; it does not make it a proxy for Alen.

---

## 13. Scoring the whole pool and indexing afterwards, instead of gathering candidates first

**Date:** 2026-10-31
**Found by:** the scoring-cost table in the first study — R1 at 66.9 ms for what
should be a single 50,000 × 768 dot product.

`score` received a candidate array and did the obvious thing:
`features.siglip[candidates] @ u`. Measured on the real pool shape:

| expression | mean |
| --- | --- |
| `siglip[candidates] @ u` | 58.5 ms |
| `(siglip @ u)[candidates]` | **7.0 ms** |
| `siglip[candidates]` alone (no arithmetic at all) | 93.3 ms |
| `siglip[candidates] @ liked.T` (R2, 100 labels) | 94.7 ms |
| `(siglip @ liked.T)[candidates]` | **49.8 ms** |

The fancy index is a copy: 153 MB allocated and moved, scattered, before any
arithmetic happens. The dot product reads the same 153 MB once, sequentially,
inside BLAS. The operation is bound by memory bandwidth, not by floating-point
throughput — 38 million multiply-adds is nothing for this CPU — so doing
*strictly more arithmetic* is eight times faster.

**Chosen:** both R1 and R2 score every row of the pool and index the result.
The wasted work is proportional to the number of images already seen (a few
thousand at most), so it never grows into a problem. A test asserts that
scoring a subset equals scoring everything and then subsetting, in value and in
order, because the failure mode — cards ranked by another image's score — would
be subtle enough to survive for months.

---

## 14. R2 is a teaching rung, not a serving one, and the measurement says so

**Date:** 2026-10-31

The brief specifies R2's kNN scoring "via FAISS" (§7.5) and a 50 ms p95 for
`/feed/next` across R1–R5. Both were implemented and measured, and they are
incompatible beyond about 100 labels:

| labels | R2 via FAISS | R1 |
| --- | --- | --- |
| 100 | 46 ms | 7 ms |
| 1,000 | 268 ms | 7 ms |

R2's cost is `pool x labels` because every candidate is compared against every
label; R1's is `pool` because the labels are averaged into one vector first.
FAISS halves R2's constant — worth doing, and done — but cannot change the
shape.

**Chosen:** implement R2 exactly as specified, measure it, and record the
failure against the budget in `docs/EVALUATION.md` and `docs/BENCHMARKS.md`
rather than quietly making the rung cheaper. It stays on the ladder because it
is the rung that teaches why a single preference vector is not enough, and it
is honest to serve for the first few dozen swipes.

**Rejected:** capping the labels R2 uses (changes the model while keeping its
name), shortlisting candidates from each like's neighbours (approximate, and
`build_batch` needs a score for every candidate so that MMR can run in Phase
10), and moving the multiply to the GPU (fixes it, but adds a GPU-resident
serving path the brief does not ask for — the brief trains on the GPU and
serves from NumPy).

The honest conclusion is the one the ladder is built to produce: R3 scores on
128 PCA dimensions with one weight vector, constant in the label count and
about a millisecond, and that is the rung built to serve.

---

## 15. The feed hands over from R0 to the active rung by itself

**Date:** 2026-10-31
**Found by:** the first attempt at the Phase 8 study, which crashed on its first
batch with `UnfittedModelError: r1 has not been fitted (it saw 0 labels and
needs 5)`.

Every rung above R0 has a minimum label count, and the error message already
said "the feed should have fallen back to a lower rung" — but nothing did. A
fresh profile with R1 active would have made the very first `/feed/next` return
a 500, which is the state every new profile starts in.

**Chosen:** `rec.feed.build_batch` takes a `bootstrap` model and uses it
whenever the active model is not fitted. The API passes a single
`ColdStartRecommender` held on `AppState` for the life of the process, and the
simulator passes one per session — in both cases *one* instance, because R0
keeps a round-robin position over the 64 clusters and a fresh instance per
request would restart at cluster 0 and serve the same corner of the collection
forever.

The impression records the rung that actually served (`r0_bootstrap`), not the
one that was configured. Filing a cold-start card under R1 would have corrupted
every later comparison between them, which is the kind of error that is
invisible until an evaluation disagrees with itself.

Passing no bootstrap keeps the strict behaviour: an unfitted model raises. The
fallback is opt-in, so a unit test still catches a model that should have been
fitted and was not.
