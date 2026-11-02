# Dataset card: the VisualTaste collection

The 50,000 photographs VisualTaste recommends from: where they come from, what
was filtered out and why, what the collection contains, and where it is
lopsided. Every number below is read from the project database
(`data/db/visualtaste.db`) as built on 2026-10-31.

## Purpose

A broad, openly licensed pool for learning one person's visual taste. Breadth
matters more than size: a recommender can only discover an interest the pool
contains, so the collection is balanced across visual clusters rather than
sampled in proportion to what the sources hold.

## Sources and licences

| Source | Licence | Stored | Kept (`ok`) | Near-duplicates | Removed by safety screen | Selected |
| --- | --- | --- | --- | --- | --- | --- |
| [Megalith-10M](https://huggingface.co/datasets/madebyollin/megalith-10m) (Flickr) | public domain, collection-level (DECISIONS #8) | 30,000 | 29,571 | 54 | 375 | 18,825 |
| [PD12M](https://huggingface.co/datasets/Spawning/PD12M) | CC0 (6,354 selected) and Public Domain Mark (8,653) | 25,029 | 24,113 | 908 | 8 | 15,007 |
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) | CC BY 2.0 | 15,002 | 14,607 | 18 | 377 | 10,102 |
| [Unsplash Lite](https://github.com/unsplash/datasets) | Unsplash Licence | 10,005 | 9,906 | 64 | 29 | 6,066 |
| **Total** | | **80,036** | **78,197** | **1,044** | **789** | **50,000** |

"Stored" counts originals written to the archive; candidates without a
licence URL were never written at all (the licence gate in `data/sources/`).
The other columns are database rows. For three sources they reconcile
exactly; for Unsplash, one image was rejected at ingestion for its aspect
ratio and 5 archived originals have no database row.

**Attribution.** Every card in the app shows the photographer (where known),
the licence as a link, and the source page. CC BY 2.0 requires this; the
Unsplash terms ask for it; for the public-domain sources it is provenance.

**Redistribution.** The repository contains no collection images. The demo
pack and the online demo bundle only CC0, Public Domain Mark, public-domain
and CC BY images, with attribution files; Unsplash images are left out,
because the Unsplash Lite dataset terms do not clearly allow redistributing
the files.

## How the 50,000 were chosen

| Step | Rule | Code |
| --- | --- | --- |
| Download | ≤ 4 concurrent requests, backoff with jitter, TLS on; originals over 8 MB skipped (928 PD12M, 2 Open Images) | `data/http.py`, DECISIONS #2 |
| Validate | full decode; short side ≥ 256 px; aspect ratio ≤ 4:1; EXIF rotation applied; animated images keep their first frame | `data/validate.py` |
| Exact and near duplicates | perceptual hash, Hamming distance ≤ 6, keep the highest resolution | `data/dedupe.py` |
| Embedding duplicates | DINOv2 cosine ≥ 0.95 folded together (733) | `jobs/select.py` |
| Safety screen | zero-shot SigLIP 2 screen, threshold 0.0478 calibrated on 200 hand-checked images; the 789 flagged (about 1%) were removed by the owner's decision (QUESTIONS.md, Q5) | `data/safety.py` |
| Balance | k-means into 400 clusters on SigLIP 2; at most 187 images per cluster; 398 clusters represented | `jobs/select.py` |
| Features | SigLIP 2 and DINOv2 vectors, PCA-128 of each, a CIELAB colour histogram, 64 coarse clusters for exploration | `jobs/embed.py`, `jobs/select.py` |

No source exceeds 50% of the selection (the largest, Megalith, is 37.7%).

## What it contains

Each of the 64 coarse clusters was named automatically by the three concept
words closest to its SigLIP 2 centroid (the same vocabulary the Profile page
uses), then grouped by hand into themes. The grouping is a judgement call; the
cluster sizes are exact.

| Theme | Images | Share | Example cluster labels |
| --- | --- | --- | --- |
| Nature, plants and animals | 9,885 | 19.8% | wildflower, flower, seed pod · mammal, bird, owl · beach, ocean, coast |
| People, crowds and events | 8,105 | 16.2% | joyful, candid, dancer · market day, protest, crowd · ceremony, traditional dress |
| Archive photographs | 7,277 | 14.6% | tintype, group portrait, soldier · tintype, daguerreotype, cyanotype |
| Objects, food and still life | 6,330 | 12.7% | food, fish dish, cooking · tableware, jewellery, vase · coin, medal, engraving |
| Transport | 4,503 | 9.0% | car, vintage car · aircraft, airport, biplane · ship, fishing boat |
| Prints, drawings and documents | 4,438 | 8.9% | lithograph, painting, drawing · newspaper, poster · manuscript, map |
| Buildings and streets | 3,935 | 7.9% | cityscape, apartment block · shop, sign, typography |
| Sport | 3,279 | 6.6% | athlete, sport, performance · race · motorcycle, bicycle |
| Photographic style | 2,248 | 4.5% | bokeh, dreamlike · light leak, long exposure · night sky, blue hour |

Image sizes: width 256 to 10,328 px, mean about 1,270 × 1,150.

## Known limitations and biases

- **Documentary, not designed.** Archive photographs and pictures of people at
  events together make up about 31%; illustration, graphic design and
  contemporary art are thin (prints and drawings are 9%, mostly historical).
- **Old and Western-heavy archives.** Megalith and PD12M lean on museum and
  library collections, which over-represent 19th- and early 20th-century
  Europe and North America.
- **Open Images is everyday and amateur.** Its CC BY photos are mostly
  snapshots: events, vehicles, food.
- **Balancing changes proportions on purpose.** Capping clusters at 187 makes
  rare looks more common than in the sources and common looks rarer. That is
  intended (every interest should be findable) but it means the pool is not a
  sample of any natural distribution.
- **The safety screen is deliberately over-cautious, and it costs breadth.**
  A review of 100 random removals (below) found nothing explicit and 91
  plainly harmless images, so about 700 safe images were lost, mostly sport,
  swimwear at pools and beaches, and bare skin. Removed images are recorded
  in the `removals` table and never deleted from disk.
- **Collection-level provenance for Megalith, mostly confirmed.** Its
  public-domain status comes from the dataset's curation of Flickr's
  public-domain marks. A check of 200 random selected images against their
  Flickr pages today (below) confirmed 95.5%; about 1.5% now carry other
  licences, most likely changed after the dataset's snapshot.

## Megalith provenance check (2026-11-01)

200 of the 18,825 selected Megalith images were drawn at random
(`random.Random(1729)`) and their Flickr pages fetched, reading the licence
code Flickr embeds in each page.

| Flickr licence today | Images |
| --- | --- |
| Public Domain Mark 1.0 | 121 |
| CC0 1.0 | 58 |
| US Government work | 8 |
| No known copyright restrictions (Flickr Commons) | 4 |
| Page unavailable (404 or 403: deleted or made private) | 6 |
| All rights reserved | 2 (both from one uploader) |
| CC BY 2.0 | 1 |

191 of 200 (95.5%) are confirmed public domain, and 191 of the 194 still
reachable (98.5%). Extrapolated, roughly 280 selected Megalith images (1.5%)
may now be under another licence. The local app keeps them, credited to
their source page. **Everything this project redistributes** (the demo pack
and the online demo) was checked image by image, and any Megalith image not
confirmed public domain today was left out.

## Safety screen review (2026-11-01)

100 of the 789 removals were drawn at random (`random.Random(1729)`) and
checked by eye on contact sheets of 150 px thumbnails.

| Judgement | Count | What they show |
| --- | --- | --- |
| Explicit nudity, sexual content or graphic violence | 0 | none |
| Mildly suggestive | 7 | swimwear or lingerie posing, one close crop |
| Weapon or blood-like makeup | 2 | a pointed handgun; face paint |
| Clearly harmless | 91 | sport, swimming, yoga, bare feet, pets, an X-ray, concerts, a surgery photo |

With 0 of 100, the rate of genuinely unsafe images among the removals is
likely below 3% (rule of three). The screen keys on exposed skin and dark
frames, not on content. A per-image review, or a higher threshold with the
same calibration set, would recover most of the 789; restoring them is a
database update (clear the `removals` row, set `status = 'ok'`, re-run
`jobs.select`), left to the owner.

## Rebuilding it

The collection is not in the repository; every step above is a resumable job.
See the README's Quick start, or install the 2,000-image demo pack.
