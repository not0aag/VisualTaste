# VisualTaste: what is left, and what would make it better

Written 2026-11-01, after the UI upgrade and the GitHub launch. It is an honest
review of the project as it stands, followed by a prioritised plan. Each item
lists why it matters, how to do it, the files involved, and how to know it is
done.

---

## 0. The verdict

**Satisfied with the foundations; not satisfied that it is finished.**

What is genuinely strong:

- **The recommender ladder.** Seven rungs share one interface. Each is measured
  against the one below it, and the maths is hand-checked (NumPy against
  scikit-learn to 1e-3, Laplace covariance = H⁻¹, hand gradients against
  autograd).
- **The evaluation discipline.** A planted-taste simulator, paired seeds, a
  bootstrap with sign tests, and verdicts recorded even when they go against
  the model (H2 not supported, H3 failing its like-rate bar, H4 inconclusive).
- **The engineering.** Every job resumes, nothing is ever deleted, backups
  work and are tested, feed latency is 39 ms at p95, and online retraining
  hot-swaps atomically.
- **The interface and the README.** Distinctive, data-driven and honest.

What is not good enough yet, in order of how much it would bother a careful
reviewer:

1. **The final evaluation is unfinished.** `phase14_final` is about two-thirds
   done, H5 is still running, `DEFAULT_RUNG` is not chosen, and the latency
   table is not measured.
2. **The docs disagree with themselves.** `docs/EVALUATION.md`'s hypothesis
   table still says H3 and H4 are "not yet run", while the README reports
   their results.
3. **Nobody else can run it quickly.** Building the collection takes hours of
   downloads and a CUDA GPU, and every script is PowerShell.
4. **There is no CI.** GitHub shows no green check, and nothing stops a
   regression from being pushed.
5. **The headline research result is a trade-off, not a win.** R4 finds every
   interest but gives up 0.19 of like rate. That is honest, but it is the
   obvious next research problem.
6. **The speed-ups are not merged.** The `perf` branch (2.5× faster R2,
   bit-identical) sits on the pre-publication history and has never passed the
   full suite, because that run was killed when memory ran out.

---

## 1. Priority overview

| # | Item | Priority | Effort | Impact |
| --- | --- | --- | --- | --- |
| A1 | Finish `phase14_final`, choose `DEFAULT_RUNG` | P0 | jobs running; ~1 h of writing | completes the evaluation |
| A2 | Record H5 (neural ranker) | P0 | job running; ~1 h of writing | closes the last hypothesis |
| A3 | Measure the latency table | P0 | 30 min | backs a README claim with a table |
| A4 | Make EVALUATION.md consistent; write "what we learned" | P0 | 2 h | removes a contradiction a reviewer would spot |
| A5 | Merge the `perf` branch properly | P0 | 1–2 h | faster studies, cleaner history |
| A6 | Fresh-clone check | P0 | 1 h | proves the repo works from GitHub |
| A7 | Small leftovers (notebook 01, stale hashes, checkboxes) | P1 | 1–2 h | polish |
| B1 | Five-minute demo pack | P0 | 1 day | anyone can try it |
| B2 | Cross-platform scripts | P1 | half a day | Linux and macOS users can run it |
| B3 | CPU-only path | P1 | half a day | runs without NVIDIA |
| C1 | GitHub Actions CI | P0 | half a day | green check on every push |
| C2 | Test hygiene (GPU markers, temp files, memory) | P1 | 2–3 h | prevents the 25 GB temp incident |
| D1 | Close the exploration trade-off (H3) | P1 | 2–4 days | the headline research result |
| D2 | Hybrid ranker (H2) | P2 | 1–2 days | uses kNN's ranking edge |
| D3 | Put interests to work (H4) | P2 | 1–2 days | turns R5 into a ranking gain |
| D4 | Describe your taste in words | P2 | 1 day | memorable feature, no cold start |
| D5 | Off-policy evaluation of the real profile | P2 | 1–2 days | evaluates on real data, not only simulated |
| E1 | λ-tuning cost | P1 | 1 day | 2–5× faster R3/R4 studies |
| E2 | Lighter chart bundle | P2 | half a day | Model Lab loads 4.7 MB less |
| F1–F9 | Interface improvements | P2 | 0.5–1 day each | product polish |
| G1–G4 | Data quality | P2–P3 | varies | provenance and coverage |
| H1 | Live demo on GitHub Pages | P1 | 1–2 days | biggest portfolio impact |
| H2–H7 | Release, write-up, repo hygiene | P1–P2 | small each | presentation |
| I1–I4 | Security and dependency hygiene | P2 | small each | robustness |

P0 means do it before sharing the link widely. P1 means do it this month.
P2 means worthwhile. P3 means nice to have.

---

## 2. Part A: finish what is started

### A1. Finish `phase14_final` and choose `DEFAULT_RUNG`

**State today.** The cache under `data/eval/phase14_final/` holds:

| Scenario | r0 | r1 | r2 | r3 | r3+r5 | r4 | r4+r5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 semantic | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| S2 appearance | 20 | 20 | 20 | 20 | 7 | – | – |
| S3 colour | – | – | – | – | – | – | – |
| S4 conjunctive | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| S5 drift | 20 | 20 | 20 | 20 | 10 | – | – |
| S6 six interests | – | – | – | – | – | – | – |

Two runners are filling it: `run_final_split.ps1` for S1–S3 and for S4–S6.

**Steps.**

1. Let both runners finish. They resume from the cache if the machine
   restarts; the commands are in `docs/PROGRESS.md`.
2. Write the report and the default-rung rule:
   `python -m visualtaste.jobs.evaluate --study phase14_final --seeds 20`.
   With no `--variants`, it reads the cache and appends the report.
3. Render the figures: `python -m visualtaste.jobs.figures --study phase14_final`.
4. Read the `default_rung_choice` output (`jobs/evaluate.py:480`). The
   eligible rungs are `r1, r3, r3+r5, r4, r4+r5`, judged on like rate,
   interests found and coverage over S1, S2, S3, S5 and S6. Set
   `default_rung` in `config.py` and record the reasoning as a numbered entry
   in `docs/DECISIONS.md`.
5. Re-run `scripts/readme_charts.py`, and add a "final study" chart to the
   README: the six scenarios as small multiples, like rate against interests
   found per rung.

**Done when** EVALUATION.md has the dated `phase14_final` section, the config
default matches the documented choice, and the README states the default and
why.

### A2. Record H5, the neural ranker

**State today.** `phase12_h5` (10 seeds × 5,000 swipes, GPU) is fitting R6
ensembles at increasing label counts; its log shows fits at 500, 1,500 and
3,000 labels. Nothing is cached under `data/eval/` for it yet, because it
writes when a seed completes.

**Steps.** Let it finish, then read the learning curve (R6 against R3 by label
count) and the simulated dwell study. Update the H5 row of both hypothesis
tables, README and EVALUATION.md. If R6 never beats R3, say so: "a neural
ranker did not earn its complexity at this data scale" is a valid and
interesting result.

**Done when** the H5 row has a verdict with its interval and seed count, and
the README's ⏳ is gone.

### A3. Measure the latency table

**Why.** The README quotes 39 ms p95, which was measured on 2026-10-31 for R3.
The checklist also asks for R4 at 0, 100, 1,000 and 3,000 labels. R4 draws
from a 1,537-dimensional posterior on every batch, so its latency is the one
most likely to break the 50 ms budget.

**Steps.** Wait until the CPU is idle (no studies running), then run:

```powershell
python -m visualtaste.jobs.latency --rungs r0,r1,r3,r4 --labels 0,100,1000,3000
```

It appends a p50/p95 table to `docs/BENCHMARKS.md`. If R4 misses the budget
at 3,000 labels, see E3.

**Done when** BENCHMARKS.md has the table, and the README row cites it.

### A4. Make EVALUATION.md consistent, and write "what we learned"

**The contradiction.** The table under "Hypotheses to be tested"
(`docs/EVALUATION.md`, around line 31) says H3 and H4 are "not yet run". The
studies are in the same file further down, and the README reports them.

**Steps.**

1. Update the H3 row. Discovery is supported in S1, S5 and S6 (for example,
   in S6 all six interests are found in 85 points more seeds, about 770
   swipes sooner). Like-rate retention fails the 70% bar in all three
   scenarios (−0.21, −0.31 and −0.34 of R3's gain).
2. Update the H4 row: inconclusive on every metric in S1, S5 and S6.
3. Add a closing "What we learned" section, of about one page, covering:
   - R1 and R2 beat cold start, but both narrow onto 2 of 3 interests.
   - Calibration and ranking split (H2). Choose the rung by the job: R3 for
     probabilities, R2-like scores for top-k.
   - Exploration buys discovery and coverage at a measurable like-rate cost.
     The cost is the open problem (D1).
   - Interest features did not improve ranking. Interests are valuable for
     explanation and allocation, not as features (D3).
   - Performance lessons: index after the product, never gather first; FAISS
     for kNN; the share of decode and IO in embedding throughput.
   - Methodological lessons: per-session R0 seeding (DECISIONS #19), separate
     database copies for parallel runs (#22), and caching every seed.

**Done when** no table anywhere says "not yet run" for a study that has run,
and the README and EVALUATION.md agree number for number.

### A5. Merge the `perf` branch properly

**State today.** `perf` holds one commit with the R2 and R3 score caches and
the memoised Beta widths, plus `tests/test_score_caches.py`. They were checked
bit-identical on every rung, on S1 at 300 swipes and on S5 at 1,000. But:

- the branch sits on the **pre-publication** history, which still contains the
  spec files and co-author lines;
- its commit message has the trailers that were removed from `main`;
- the full test suite never finished on it, because the run was killed when
  memory ran out.

**Steps.**

1. When no studies are running (memory), move the one commit onto the new
   history:
   ```powershell
   git switch -c perf2 main
   git cherry-pick 2b85045        # the perf commit
   git commit --amend             # remove the co-author lines from the message
   ```
2. Run the full suite: `pytest -q`, with `--basetemp` set (see C2).
3. Re-run the bit-identical check: `C:\VisualAI-perf-data\work\golden.sh` with
   the old and new trees, then `compare.py`. It must report "0 different".
4. Merge, push, and delete the `perf` worktree (`git worktree remove
   C:\VisualAI-perf`) and the `backup/pre-publish` branch once you are sure
   the old history is no longer needed.

**Done when** `main` has the caches, all tests pass, and the worktree is gone.

### A6. Fresh-clone check

**Why.** The repo was rewritten before publishing, and `.gitignore` once hid a
whole package (`backend/visualtaste/data/`). A clone-and-run test is the only
proof that nothing else is missing.

**Steps.**

1. `git clone https://github.com/not0aag/VisualTaste.git C:\tmp\vt-clone`.
2. Follow the README's Quick start exactly, with no shortcuts from your own
   machine.
3. Run `pytest -q` and `pnpm test` in the clone. Point `VT_DATA_ROOT` at an
   empty folder, so the tests cannot lean on your real data.
4. Every step that fails, or needs knowledge not in the README, is a bug:
   fix the README or the code.

**Done when** a clean clone passes the tests and the README needs no
corrections.

### A7. Small leftovers

| Item | Where | Fix |
| --- | --- | --- |
| Notebook 01 is missing | `notebooks/` starts at 02 | add `01_embeddings_and_neighbours.ipynb`: neighbour grids for SigLIP 2 and DINOv2, as PROGRESS.md line 131 asks |
| Stale commit hashes | PROGRESS.md "Last green commit: `c9c28ac`" and others | the history rewrite changed every hash; replace them with dates or the new hashes |
| `db/backup.py` box unchecked | PROGRESS.md line 83 | the backup job exists and is tested (`jobs/backup.py`); tick it, with a pointer |
| Final empty real profile and backup | PROGRESS.md line 292 | do them after A1 to A6 |
| CRLF warnings on every commit | `.gitattributes` | add `* text=auto eol=lf`, then `git add --renormalize .` once |

---

## 3. Part B: let anyone run it in five minutes

The single biggest barrier: trying VisualTaste today takes hours of downloads,
a CUDA GPU, Windows and PowerShell. A reviewer will give it five minutes at
most.

### B1. A demo pack

**Idea.** A small, pre-built corpus published as a GitHub Release asset:

| Content | Size, estimated |
| --- | --- |
| 3,000 selected images: display 1024 px WebP plus 256 px thumbnails | ~120 MB |
| SigLIP 2 and DINOv2 vectors for those images, float16 | ~9 MB |
| FAISS indexes, PCA and standardisation stats, cluster64, concept cache | ~5 MB |
| A seeded SQLite database: images table, no profile | ~3 MB |
| UMAP projections for the map | ~1 MB |

**Steps.**

1. Add `python -m visualtaste.jobs.demo_pack --images 3000` to write the pack
   from the real corpus. Sample stratified by cluster64 and source, and
   include only images whose licence allows redistribution (CC0, PDM and
   public-domain sources first; for CC BY images, include the attribution
   file).
2. Add `python -m visualtaste.jobs.demo --download` to fetch the release
   asset, verify its SHA-256, and unpack it into `data/`.
3. Add a "Try it in five minutes" box near the top of the README.
4. **Licence check.** Unsplash Lite's terms may not allow redistributing the
   images; leave that source out of the pack, or link only. Megalith's licence
   guarantee is collection-level (DECISIONS #8), so prefer PD12M and Open
   Images.

**Done when** `git clone` → `uv venv` → `pip install` → `jobs.demo --download`
→ `dev` works on a fresh machine in under ten minutes, with no GPU.

### B2. Cross-platform scripts

**Now.** `scripts/dev.ps1`, `check.ps1`, `run_studies.ps1` and
`run_final_split.ps1` are PowerShell only.

**Steps.** Add a `justfile` (or `Makefile`) with `dev`, `check`, `studies` and
`demo` recipes calling the same Python and pnpm commands, plus `scripts/dev.sh`.
Keep the `.ps1` files for Windows. Check that paths in `config.py` use
`pathlib` throughout; spot-check `resolve_archive_root`, whose drive-letter
logic is Windows-specific and needs a POSIX branch.

**Done when** `just dev` and `just check` work on Ubuntu, in CI as well (C1).

### B3. A CPU-only path

**Steps.** Make `vision/encoders.py` fall back to CPU when CUDA is absent,
with a logged warning and smaller default batches. Mark GPU-only tests (C2).
The demo pack (B1) needs no encoder at all, because embeddings ship with it.
State in the README which steps need a GPU: building a collection from scratch,
and R6 training at speed.

---

## 4. Part C: CI and test hygiene

### C1. GitHub Actions

`.github/workflows/ci.yml`, run on every push and pull request:

| Job | Runs | Notes |
| --- | --- | --- |
| lint | `ruff check`, `ruff format --check` | seconds |
| backend | `pytest -q -m "not gpu"` | install CPU torch from `https://download.pytorch.org/whl/cpu`; cache uv |
| frontend | `pnpm install --frozen-lockfile`, `pnpm typecheck`, `pnpm test`, `pnpm build` | Node 22, pnpm 11 |
| e2e | `pnpm exec playwright install --with-deps chromium`, `pnpm e2e` | `scripts/build_e2e_corpus.py` builds a synthetic corpus without any model, so this works on a CPU runner |

Then add a CI badge to the README beside the test badge, and replace the
static "tests" badge with the live workflow status.

**Done when** the repo shows a green check on `main`.

### C2. Test hygiene

| Problem seen | Cause | Fix |
| --- | --- | --- |
| 25.6 GB of pytest temp files filled the disk | pytest keeps the last 3 runs' `tmp_path` folders, and some tests write large copies | in `pyproject.toml`: `tmp_path_retention_policy = "failed"` and `tmp_path_retention_count = 1`; find the biggest writers with `du` on one run's temp folder |
| The full suite was killed for low memory | studies, the app and the full suite running together | document a memory budget; add `pytest-xdist` only with `-n 2`; never run the suite while three studies run |
| `tests/test_embed.py` hard-codes `device="cuda"` | GPU assumption | a `gpu` marker and `pytest.mark.skipif(not torch.cuda.is_available())` |
| The disk guard defaults to 25 GB free | `min_free_gb` in `config.py` | this machine dropped to 14 GB, which would have stopped any download job; surface the guard in the app's health endpoint |

---

## 5. Part D: research and the recommenders

### D1. Close the exploration trade-off (H3)

**The problem.** In every scenario R4 kept a *negative* share of R3's like-rate
gain (−0.21 to −0.34), below even cold start, while finding far more interests.
The pre-registered bar is 70%.

**Why it happens.** Three exploration mechanisms stack, and none of them fades
as the model becomes confident:

1. Thompson draws from the Laplace posterior (temperature `VT_R4_TAU_POST =
   0.1`, DECISIONS #25).
2. An MMR rerank at λ = 0.7.
3. A fixed quota: 1 card in 5 for exploration until 2,000 labels.

**Ideas, in order of expected value:**

| Idea | Change | Expected effect |
| --- | --- | --- |
| Quota that fades with certainty | the explore share follows the mean Beta-interval width of rated clusters, not a fixed 1-in-5 | most of the like-rate cost goes once interests are found |
| Stop exploring found interests | count an interest as found after N likes in its cluster; quota only for clusters never liked or rated fewer than 3 times | keeps discovery and cuts repeats |
| MMR only within the top of the list | apply MMR to the top 50 by sampled score, not a 500 shortlist | less diversity pressure on clear winners |
| Information-directed sampling | choose by regret² / information gain | principled, more work |
| Posterior temperature schedule | τ decays with labels | the Thompson part fades naturally |

**Experiment design.** Add each as a `RungSpec` option (`r4:quota=adaptive`,
`r4:mmr_top=50`) so the sensitivity study (`phase10_sensitivity`) can compare
them on S1, S5 and S6, with 20 seeds each. Success means keeping at least 70%
of R3's gain **and** finding all interests in at least 80% of seeds.

**Done when** some R4 variant passes both bars. If none does, document the
Pareto front: like rate against interests found, one curve per variant.

### D2. A hybrid ranker (H2)

**The finding.** R2 (kNN) ranks better than R3 (AUC), and R3 is better
calibrated (log loss). Use both:

- add R2's score as a feature in R3, computed time-respectingly like the
  interest features, with a leakage test; or
- a rank ensemble: R3's probability for calibration, re-ordered within the
  batch by a weighted sum of R3 and R2 ranks.

The cost is that R2 scoring is about 270 ms at 1,000 labels. The `perf`
branch's per-fit cache brings that to one search per retrain, so it fits the
budget (A5).

### D3. Put interests to work (H4)

Interest *features* did not improve ranking. Use interests for **allocation**
instead: reserve slots per interest in proportion to its like rate and
uncertainty (a per-interest bandit), and show which interest each card serves
in the explanation panel. Test on S6 (six interests), where allocation should
matter most.

### D4. Describe your taste in words

SigLIP 2 has a text encoder that the app already loads for concept words.
Add a box on the empty Feed: "describe what you like" ("foggy forests, old
maps, red cars"). Embed the text and seed R1's preference vector, or the
first batches, from it. This removes most of the cold start and is a
memorable feature.

**Measure it**: extend the simulator so the planted user gives a text
description of one interest. Compare time to first like and time to all
interests found against R0.

### D5. Off-policy evaluation of the real profile

The app already logs inclusion propensities for R4 (Monte-Carlo, 200 draws,
clipped at 0.005). Add self-normalised IPS and doubly-robust estimators in
`eval/`, so a new rung can be evaluated on **your real logged data** before it
serves. Validate first on the simulator, where the truth is known
(`phase10_ips` already checks plain IPS).

### D6. Smaller research items

| Item | Note |
| --- | --- |
| Drift (S5) | time-decay weights on old labels; measure recovery swipes |
| Skip weight | studied at 0.25 / 1.0 / excluded (#20); re-check once D1 lands |
| Dwell time | the simulated dwell study ships with H5; if dwell helps, weight labels by it |
| Fine-tuning a projection head | a small adapter on frozen embeddings, trained across simulated users; only after R6 shows learning helps |

---

## 6. Part E: performance

### E1. λ-tuning cost

**Measured.** About 88% of R4+R5 session time, and most of R3's, is the
λ grid: 12 L-BFGS fits per retrain, 15 retrains per 300 swipes.
Under the bit-identical rule, nothing faster was found: thread limits changed
results at 600+ labels.

**Options if a tolerance is acceptable** (choose one, document it, and keep
the old studies' numbers):

| Option | Expected gain | Result changes |
| --- | --- | --- |
| Tune λ every 3rd retrain and reuse it between | ~2.5× | λ lags by up to 40 labels |
| Prune the grid: after the first tune, search only the best λ's neighbours | ~3× | rare differences in the chosen λ |
| Single-threaded BLAS for study processes | 2–3× when several studies run | last-decimal differences, measured |
| float32 design matrix | ~1.5× | small numeric differences |

Run new studies under the new setting from the start, so no study mixes the
two.

### E2. A lighter chart bundle

Plotly is 4.7 MB minified (1.4 MB gzipped) and loads on Model Lab and the
Embedding Map. Options: a Plotly partial bundle with only `scatter`,
`scattergl` and `bar` (about 1 MB), or uPlot for the line charts plus a small
WebGL scatter for the map.

### E3. R4 latency at high label counts

If A3 shows R4 near the 50 ms budget at 3,000 labels: reuse the Cholesky
factor between retrains (it only changes on a refit), and draw a low-rank
sample (top-k eigenvectors of the posterior covariance).

### E4. Memory

The API holds the full feature bundle in RAM. Memory-map the float32 matrices
(`np.load(..., mmap_mode="r")`) so several processes (API, studies, tests)
share pages instead of each holding a copy.

---

## 7. Part F: interface

| # | Improvement | Detail |
| --- | --- | --- |
| F1 | Show *why* visually | in the explanation panel, show the 3 nearest liked thumbnails (R2 already returns neighbours; R3 can use its top contributing liked images) |
| F2 | Keyboard help | `?` opens an overlay with every shortcut |
| F3 | Saves page | a gallery of saved images; today saves appear only in History |
| F4 | "More of this interest" | from a Profile interest card, start a feed biased to that interest |
| F5 | Screen-reader feedback | an `aria-live` region announcing "Liked", "Disliked" or "Undone" after each verdict |
| F6 | Accessibility audit | run axe in Playwright on every page; fix contrast on muted text (`canvas-500` on `canvas-950` is borderline) |
| F7 | Phone layout | at 390 px the card leaves a large gap above the buttons; size the card from the viewport height |
| F8 | Light theme (optional) | the app is dark by design; add a light theme only if the token structure (`tailwind.config.js`) is used for both |
| F9 | Text search | with D4, a search box: "show me lighthouses" via the SigLIP text encoder |

---

## 8. Part G: data

| # | Item | Detail |
| --- | --- | --- |
| G1 | Per-image provenance for Megalith | the guarantee is collection-level (DECISIONS #8); spot-check 200 images' Flickr pages and record the result |
| G2 | Dataset card | `docs/DATASET.md`: sources, counts, filters, the safety screen's threshold and false-positive rate, known biases (a large share of sport and event photos, for example) |
| G3 | Safety screen review | 789 images were auto-removed at a calibrated threshold; review a random 100 to estimate false positives |
| G4 | More variety | the mosaic shows many documentary and event photos; consider an art or design source with a clear licence |

---

## 9. Part H: presentation and portfolio

### H1. A live demo on GitHub Pages (highest impact)

A static build of the frontend that needs no backend:

1. Add a `VITE_DEMO=1` mode where `src/api/client.ts` reads from bundled JSON
   instead of `fetch`: a recorded simulator session, its feed batches with
   scores and explanations, the profile and taste responses, model versions
   and stats.
2. Ship about 300 display images (CC0 and PDM only) with the demo build.
3. Simulate swiping on the client: replay the recorded session's next batch
   on each verdict, so the learning ring, counter, Profile and Model Lab all
   move.
4. A GitHub Action builds and deploys it to `not0aag.github.io/VisualTaste`.
5. Put the link at the top of the README: "Try the demo (runs in your
   browser, replaying a recorded session)."

Label it a replay. It shows the interface and the learning; the real thing
runs locally.

### H2. Release v1.0.0

After Part A, tag `v1.0.0` with release notes: what it is, the headline
results, and the demo pack (B1) attached as an asset. Add `CHANGELOG.md`.

### H3. Social preview

Upload `docs/images/social-preview.png` at **Settings → General → Social
preview** (browser only).

### H4. A write-up

A 1,500-word post: "Teaching a recommender my visual taste, and measuring it
honestly." Cover the coverage trap (R3 narrowing), why exploration costs likes,
H2's calibration and ranking split, and the decode-bound embedding benchmark.
Link it from the README.

### H5. A short video

A 60-second screen recording (feed, drag, explanation, Profile, Model Lab
timeline, Embedding Map) uploaded to the README with GitHub's video embed.

### H6. Repo hygiene

`CONTRIBUTING.md` (setup, `check.ps1`, style), `CITATION.cff`, issue
templates, and a pinned repository on your GitHub profile.

### H7. An honest "Limitations" section in the README

Single user; evaluated mainly in simulation; a planted-taste user is simpler
than a person; the collection leans documentary; GPU and Windows for the full
build until B2 and B3 land. Stating limits makes the claims more credible.

---

## 10. Part I: security and dependencies

| # | Item | Detail |
| --- | --- | --- |
| I1 | Image route path safety | test that `/api/images/{id}/thumb` rejects ids that are not 16 hex characters, so path traversal is impossible |
| I2 | CORS and host binding | confirm CORS allows only `127.0.0.1:5173`, and that the API refuses to bind to `0.0.0.0` without an explicit flag |
| I3 | Dependency audit | `pip-audit` and `pnpm audit` in CI (C1); Dependabot for both ecosystems |
| I4 | Backups | optional encryption of `backups/events/*.jsonl` if the archive moves to a shared or cloud drive |

---

## 11. A suggested order

**Week 1: finish and verify** (everything a reviewer could fault)

1. A1, A2: let the jobs finish, then write the results up (A4).
2. A3: latency, with the CPU idle.
3. A5: merge `perf`, with the full suite and the bit-identical check.
4. C1, C2: CI green on `main`.
5. A6: fresh-clone check; fix whatever it finds.
6. A7, H3, H7.

**Week 2: make it reachable**

7. B1: the demo pack, then H2: release v1.0.0.
8. H1: the live demo on GitHub Pages.
9. B2, B3: cross-platform and CPU.

**After that: make it better**

10. D1: close the exploration trade-off. The research headline.
11. D4 with F9: describe your taste in words. The feature headline.
12. D2, D3, D5, E1, F1–F7, G1–G4, H4–H6, I1–I4 as time allows.

---

## 12. Definition of "finished"

VisualTaste is finished when:

- every hypothesis H1–H5 has a recorded verdict, and README and EVALUATION.md
  agree;
- `main` has a green CI check, and a fresh clone passes all tests;
- someone with no GPU can try it in under ten minutes, either from the demo
  pack or the live demo;
- the default rung is chosen by the final study and documented;
- the known trade-off is either closed (D1) or mapped as a Pareto front and
  stated as a limitation.

## Status (2026-11-02)

| Item | State |
| --- | --- |
| H1–H5 verdicts recorded; README and EVALUATION.md agree | done |
| Green CI on `main` | done |
| Default rung chosen by the final study (R4+R5, DECISIONS #29) | done |
| D1 trade-off | mostly closed on R4: 0.70 (S1) and 0.80 (S6) of R3's gain with every interest found; not yet tested on R4+R5 or S2–S5 |
| D2 hybrid, D3 interest allocation, D6 label decay | studies running; results append to EVALUATION.md when they finish |
| D4 describe your taste, D5 doubly robust OPE, E1, E2, E4, F1–F9, G1–G3, H2, H4, H6, H7, I1–I4 | done (E4 and F8 decided against, with reasons) |
| Try it without a GPU (demo pack, live demo) | built; waiting on the owner to publish the releases and enable Pages |
| A3 latency with an idle CPU, A6 fresh-clone check, H5 video | after the studies finish |
| G4 new image source | future work: no clearly licensed art source found |
