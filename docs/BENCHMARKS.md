# Benchmarks

Every number in this file comes from a run that actually happened on this
machine, with the command and date next to it. Anything derived by arithmetic
rather than measured is labelled "estimate".

## Hardware

Measured 2026-10-30 (`nvidia-smi`, `Get-CimInstance Win32_Processor`):

| Component | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB VRAM, driver 576.80, CUDA 12.9 |
| CPU | AMD Ryzen 9 5900HX with Radeon Graphics, 8 cores / 16 threads |
| RAM | 15.4 GB |
| Disk | single NVMe volume `C:`, 453.5 GB total |
| OS | Windows 11 Home Single Language 10.0.26200 |
| Python | 3.12.16 (uv-managed venv) |
| torch | 2.11.0+cu128 (CUDA build 12.8), `torch.cuda.is_available()` is True |

The 6 GB VRAM figure is the binding constraint on batch size, and the reason the
synthetic benchmark below sweeps batch sizes rather than assuming one.

<!-- Measured runs are appended below by `python -m visualtaste.jobs.benchmark`. -->

## Synthetic-tensor encoder throughput (2026-10-30 23:10 UTC)

- Command: `python -m visualtaste.jobs.benchmark`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 576.80, 6144 MiB
- CPU: AMD Ryzen 9 5900HX with Radeon Graphics / 8 cores / 16 threads
- torch 2.11.0+cu128, CUDA build 12.8, Python 3.12.16
- Method: 1 warm-up + 5 timed forward passes on random tensors, fp16 autocast, `torch.inference_mode()`. Forward pass only -- no image decoding, no disk IO.

| model_key | batch | images/s | peak VRAM (MB) | s/batch | status |
| --- | --- | --- | --- | --- | --- |
| `dinov2_b14` | 32 | 322.7 | 551 | 0.099 | ok |
| `dinov2_b14` | 64 | 335.5 | 762 | 0.191 | ok |
| `dinov2_b14` | 128 | 330.5 | 1185 | 0.387 | ok |
| `dinov2_b14` | 256 | 322.5 | 2032 | 0.794 | ok |
| `siglip2_b16_224` | 32 | 413.8 | 1606 | 0.077 | ok |
| `siglip2_b16_224` | 64 | 436.6 | 1773 | 0.147 | ok |
| `siglip2_b16_224` | 128 | 437.9 | 2106 | 0.292 | ok |
| `siglip2_b16_224` | 256 | 406.2 | 2768 | 0.630 | ok |

- `dinov2_b14`: recommended batch **256** (322 img/s, peak 2032 MB) -> ~4 min of pure GPU time for 80k images (estimate: excludes decoding and IO).
- `siglip2_b16_224`: recommended batch **128** (438 img/s, peak 2106 MB) -> ~3 min of pure GPU time for 80k images (estimate: excludes decoding and IO).

### What this means for Phase 4

Both encoders fit every batch size tried: peak VRAM tops out at 2.8 GB of the
6144 MB available, so the 6 GB card is not the constraint it was expected to be
and no `out_of_memory` row appeared. Throughput is flat across batch sizes,
which is the signature of a GPU that is already saturated by batch 32.

The headline: **~4 minutes of pure GPU time to embed 80,000 images per model**.
Embedding will therefore be bound by reading and decoding 80,000 WebP files,
not by the GPU. That is what the real-data benchmark in Phase 4 measures, and it
is why the embedding job's `DataLoader` gets worker processes rather than a
bigger batch.

## Real-data embedding throughput (2026-10-31 16:44 UTC)

- Command: `python -m visualtaste.jobs.embed --model all`
- Includes WebP decode, preprocessing, host-to-device copy and the forward pass -- unlike the synthetic benchmark above, which times the forward pass alone.

| model | embedded | failed | images/s | batch | total on disk |
| --- | --- | --- | --- | --- | --- |
| `dinov2_b14` | 79779 | 0 | 54 | 256 | 79779 |
| `siglip2_b16_224` | 79779 | 0 | 41 | 128 | 79779 |

- `dinov2_b14`: 17% of the synthetic-tensor rate (322 img/s), so the remaining 83% is decoding and IO.
- `siglip2_b16_224`: 9% of the synthetic-tensor rate (438 img/s), so the remaining 91% is decoding and IO.

## `GET /api/feed/next` latency (2026-10-31)

- Command: 40 sequential calls through FastAPI's `TestClient` against the real
  50,000-image selected pool, cold-start rung (R0), after three warm-up calls.
- Target: p95 < 50 ms (brief §7.5).

| version | median | p95 | max |
| --- | --- | --- | --- |
| first measurement | 216 ms | 245 ms | 264 ms |
| after the fix | **34.4 ms** | **39.4 ms** | 43.9 ms |

The first measurement missed the budget by five times, and the cause was one
function. `candidate_positions` queried the 50,000 selected ids on every
request and then looped over all 79,779 bundle rows testing set membership --
roughly 130,000 Python-level operations on the hot path.

Two changes fixed it. The `FeatureBundle` is now built over the selected pool
rather than the whole embedded corpus, which both removes 30,000 unusable rows
and makes the PCA-128 block align exactly (it is fitted on the selected pool,
so against the full corpus it was incomplete and therefore silently dropped).
And the candidate set is now computed as "every row, minus the seen ones",
where the per-request cost is proportional to how many images have been seen --
a few thousand at most -- instead of to the size of the corpus.

Measured on the hardware at the top of this file, with the GPU idle.

---

## R2 kNN scoring cost: dense NumPy vs FAISS

**Command:** a timing loop over the live 50,000 x 768 pool, `k = 5`, mean of 7
runs per cell, machine otherwise idle.
**Date:** 2026-10-31.

| labelled vectors | dense NumPy | FAISS `IndexFlatIP` |
| --- | --- | --- |
| 50 | 64 ms | 32 ms |
| 100 | 94 ms | 46 ms |
| 200 | 165 ms | 65 ms |
| 500 | 323 ms | 145 ms |
| 1,000 | 608 ms | 268 ms |
| 2,000 | 1,181 ms | 536 ms |

The two agree to 4e-8 — identical arithmetic, different bookkeeping. The dense
version materialises a 50,000 x N similarity matrix (200 MB at N = 1,000) and
partitions it; FAISS blocks the multiply and keeps a running top-k heap.

**The shape is the finding, not the constant.** R2's cost is proportional to
`pool x labels`, so halving it with FAISS moves the point at which it misses
the 50 ms budget from about 50 labels to about 100, and no further. R1 by
contrast is 7.2 ms at 1,000 labels and would be 7.2 ms at 100,000, because it
averages the labels into one vector first. See `docs/EVALUATION.md` for what
that means for the ladder.

---

## Candidate gathering: the cost of a fancy index

**Command:** a timing loop on arrays of the real pool's shape (50,000 x 768
float32), mean of 10 runs.
**Date:** 2026-10-31.

| expression | mean |
| --- | --- |
| `siglip[candidates] @ u` | 58.5 ms |
| `(siglip @ u)[candidates]` | **7.0 ms** |
| `siglip[candidates]` alone, no arithmetic | 93.3 ms |

Doing strictly more arithmetic is eight times faster, because the gather
allocates and scatter-copies 153 MB while the dot product streams the same
153 MB once through BLAS. This operation is memory-bandwidth bound: 38 million
multiply-adds is nothing for this CPU. Both R1 and R2 score the whole pool and
index afterwards (`docs/DECISIONS.md` #13).

## UMAP projection (2026-11-01 03:25 UTC)

```
python -m visualtaste.jobs.project --models siglip2_b16_224,dinov2_b14
```

PCA-50 then UMAP (cosine, n_neighbors 30, min_dist 0.1, seeded, so single-threaded). Machine: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD. Run alongside the simulator studies, so on a busy CPU.

| model | points | wall time |
| --- | --- | --- |
| `siglip2_b16_224` | 50,000 | 366 s |
| `dinov2_b14` | 50,000 | 153 s |

## Memory-mapping the feature matrices (ROADMAP E4): not done

**Date:** 2026-11-01. **Decision:** keep loading into private memory.

The idea was to open the vector files with `np.load(..., mmap_mode="r")` so the
API, the studies and the tests share one copy of the pages instead of each
holding its own. Measured on the real data:

| file | rows | size on disk |
| --- | --- | --- |
| `embeddings/siglip2_b16_224/vectors.npy` | 79,779 | 245 MB |
| `embeddings/dinov2_b14/vectors.npy` | 79,779 | 245 MB |
| derived features (PCA-128 x 2, colour-64, UMAP x 2) | 50,000 | 72 MB |

The archive holds every embedded image (79,779), but the recommender works on
the 50,000 selected ones. `AppState` therefore picks the selected rows out of
each archive file in pool order (`api/deps.py`, `_align`), and that gather
makes a private copy (about 153 MB per model) whether or not the file was
memory-mapped. A memory map would only share the 245 MB archive pages, which
are read once at startup and then never touched again.

Sharing pages would need a second, pool-ordered copy of each matrix on disk
(about 300 MB more), written by `jobs/select` and kept in step with it. For a single-user app
that runs one API process, the saving is about 300 MB of RAM and only while a
study runs alongside. That does not pay for a second source of truth. Each
study process holds 330–460 MB in total (measured with three studies running),
which 16 GB of RAM covers.

## Re-tuning lambda less often (ROADMAP E1)

**Date:** 2026-11-01. **Option:** `r3:tune_every=3` (also `r4:`), off by default.

R3 picks its regularisation strength on time-ordered splits at every refit
(every 20 labels), and that search dominates a simulated session's cost.
Re-tuning on every third refit, and reusing the last choice in between, made
one S1 seed of 1,000 swipes about 2.2x faster (one seed, measured while other
studies ran, so indicative only).

Numeric drift, from `phase15_tuning` (20 paired seeds, EVALUATION.md): no
detectable change for R3 (like rate -0.005 [-0.018, +0.008]); -0.011
[-0.020, -0.003] for R4. So the option is for quick exploratory studies;
results meant to be compared with the cached ones keep the default.

## `/feed/next` latency (2026-11-02 03:55 UTC)

```
python -m visualtaste.jobs.latency --rungs r0,r1,r3,r4,r4+r5 --labels 0,100,1000,3000 --requests 200
```

200 requests per row through `TestClient`, end to end, after one warm-up request. Machine: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, Python 3.12.16. "served" is the rung that actually served the batches (below its label minimum a rung falls back to R0). Budget: p95 < 50 ms.

| rung | labels | served | p50 ms | p95 ms | max ms | after response, p50 ms | within budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r0` | 0 | `r0` | 34.4 | 42.8 | 66.9 | 0 | yes |
| `r0` | 100 | `r0` | 35.8 | 46.2 | 120.5 | 0 | yes |
| `r0` | 1,000 | `r0` | 38.5 | 48.5 | 115.3 | 0 | yes |
| `r0` | 3,000 | `r0` | 48.6 | 60.0 | 90.5 | 0 | **no** |
| `r1` | 0 | `r0` | 35.3 | 43.3 | 108.3 | 0 | yes |
| `r1` | 100 | `r1` | 29.6 | 32.1 | 128.2 | 0 | yes |
| `r1` | 1,000 | `r1` | 33.3 | 36.7 | 115.5 | 0 | yes |
| `r1` | 3,000 | `r1` | 41.3 | 46.4 | 131.4 | 0 | yes |
| `r3` | 0 | `r0` | 33.2 | 43.6 | 74.8 | 0 | yes |
| `r3` | 100 | `r3` | 24.1 | 27.5 | 71.5 | 0 | yes |
| `r3` | 1,000 | `r3` | 31.6 | 36.3 | 115.9 | 0 | yes |
| `r3` | 3,000 | `r3` | 41.6 | 49.2 | 103.2 | 0 | yes |
| `r4` | 0 | `r0` | 36.7 | 48.1 | 87.7 | 0 | yes |
| `r4` | 100 | `r4` | 44.0 | 52.3 | 62.6 | 685 | **no** |
| `r4` | 1,000 | `r4` | 50.4 | 62.4 | 77.8 | 639 | **no** |
| `r4` | 3,000 | `r4` | 56.7 | 71.3 | 88.4 | 567 | **no** |
| `r4+r5` | 0 | `r0` | 34.7 | 42.6 | 163.3 | 0 | yes |
| `r4+r5` | 100 | `r4+r5` | 44.3 | 54.8 | 146.8 | 713 | **no** |
| `r4+r5` | 1,000 | `r4+r5` | 51.4 | 64.5 | 772.2 | 671 | **no** |
| `r4+r5` | 3,000 | `r4+r5` | 57.0 | 72.4 | 83.7 | 603 | **no** |

### Reading it (2026-11-02)

R4's "after response" column is its propensity estimate (200 replayed
posterior draws, 0.6-0.7 s). Production runs it after the reply is sent;
earlier runs of this benchmark counted it as request time, which is why R4
once read as 1 s. That was a measuring bug, fixed in `jobs/latency.py`.

What is left: R4 and R4+R5 miss the 50 ms p95 budget by 3-16 ms, and every
rung slows as labels pile up (R0 too, at 60 ms with 3,000 labels). A profile
of one R4 request at 3,000 labels:

- the recommender's own work (sampling, MMR, exploration) is 10-25 ms;
- "nearest liked images" in the explanation cost 7.5 ms per batch, gathering
  the same vectors five times. Now gathered once per fit: p95 at 3,000
  labels went from 71.3 to 64.8 ms (R4) and 72.4 to 66.4 ms (R4+R5);
- candidate gathering (excluding seen images) is about 10 ms and grows with
  labels, which is what slows R0;
- the rest is the request itself: database connection, impression writes,
  JSON, and `TestClient`'s thread hand-off, which a real server does not pay.

Re-measured after the nearest-liked change (200 requests per row):

| rung | labels | p50 ms | p95 ms | within budget |
| --- | --- | --- | --- | --- |
| `r0` | 1,000 | 37.5 | 46.5 | yes |
| `r0` | 3,000 | 43.5 | 56.5 | no |
| `r4` | 1,000 | 46.2 | 53.4 | no |
| `r4` | 3,000 | 53.2 | 64.8 | no |
| `r4+r5` | 1,000 | 47.4 | 61.1 | no |
| `r4+r5` | 3,000 | 52.4 | 66.4 | no |

The next step is the candidate set (keep a seen-mask per profile instead of
re-reading every seen id), since it is shared by every rung. A person
swiping does not notice a 15 ms overshoot, but the budget is the stated
target and is reported as missed.

### The shared candidate set, and where the rest goes (2026-11-04)

`candidate_positions` now keeps each profile's "not yet seen" mask between
requests and reads only new events (`tests/test_seen_cache.py` checks it
against a rebuild after new events, a restore and a new bundle; the golden
seeds on r2, r3+r5, r4 and r4+r5 are byte-identical). Re-measured, 200
requests per row:

| rung | labels | p50 ms | p95 ms | within budget |
| --- | --- | --- | --- | --- |
| `r0` | 1,000 | 35.9 | 42.6 | yes |
| `r0` | 3,000 | 38.1 | 46.5 | yes (was 56.5) |
| `r4` | 1,000 | 43.2 | 58.9 | no |
| `r4` | 3,000 | 47.6 | 64.9 | no |
| `r4+r5` | 1,000 | 45.4 | 59.7 | no |

The same R4 requests through a real uvicorn server and HTTP client (one
request every 0.9 s, as a person swipes) give p50 39-40 ms and p95 56-60 ms,
so the test client is not what inflates them.

Per R4 request at 3,000 labels, about 20 ms is SQLite on Windows: opening
the connection and its first read (about 5 ms), seven statements (about 5 ms),
the commit, and the WAL checkpoint when the request's connection closes
(about 8 ms). The recommender's own work is 10-15 ms. Two attempts that did
not help, recorded so they are not tried again:

- skipping the checkpoint on close (`SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE`): the
  close got 7 ms faster and the reads 12 ms slower, as the log grew;
- setting `journal_mode` only when it is not already WAL: the 2.5 ms is the
  connection's first touch of the file, not the pragma.

What would close the gap: one long-lived connection for the API instead of
one per request (it changes the threading model, see `db/repo.py`), or
moving impression writes after the response. Both are larger changes than
a 10-15 ms overshoot that a person swiping does not notice.

### The tuned default (2026-11-04)

The app's R4+R5 now runs with `settings.r4_options` (DECISIONS #31). Same
benchmark, 200 requests per row: p50 42.9 / 44.7 / 47.0 ms and p95 54.9 /
65.8 / 59.5 ms at 100 / 1,000 / 3,000 labels, so it is still over budget by
about the same margin. Its after-response propensity work is shorter
(419-523 ms against 601-713 ms), since MMR picks from 50 images, not 500.
