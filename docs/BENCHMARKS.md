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
