-- Migration 001: the initial schema, exactly as brief section 7.4 specifies.
--
-- Read this file alongside docs/ARCHITECTURE.md, which records which component
-- writes to each table. The short version:
--
--   images, image_sources    written once by jobs/ingest, then only by the
--                            safety screen and the review page (status changes)
--   embedding_models         written by jobs/embed, one row per model_key
--   profiles                 one row per person; is_simulated=1 for the
--                            planted-taste simulator, which must never be
--                            mixed into real training data
--   impressions              written by rec/feed.py every time a card is shown
--   events                   written by the API when a card is acted on
--   interests                written by rec/r5_interests.py
--   rec_models, experiments  written by rec/registry.py and eval/interleave.py
--   embedding_errors         per-image embedding failures; a dead image must
--                            never kill a batch
--   removals                 an audit trail for anything taken out of the pool
--
-- Design notes that are easy to miss:
--
-- * `events.image_id` is NOT NULL and references images, but `impression_id` is
--   nullable: a verdict changed from the History page has no impression behind
--   it, because nothing was shown to produce it.
-- * Nothing is ever deleted in response to a user action. A changed verdict
--   sets `undone`/`superseded_by` on the old event and inserts a new one, so
--   the history stays a faithful record and time-ordered replay (eval/replay.py)
--   can reconstruct exactly what was known at any past moment.
-- * `dup_of` is a self-reference, so near-duplicate groups can be walked back
--   to the copy that was kept.

CREATE TABLE images (
  image_id TEXT PRIMARY KEY, source TEXT NOT NULL, source_id TEXT NOT NULL,
  source_url TEXT, author TEXT, author_url TEXT, title TEXT, license_url TEXT NOT NULL,
  orig_path TEXT NOT NULL, width INTEGER, height INTEGER, bytes INTEGER, phash TEXT,
  dominant_colours TEXT,
  status TEXT NOT NULL DEFAULT 'ok',   -- ok | near_dup | removed | quarantined | rejected
  dup_of TEXT REFERENCES images(image_id),
  selected INTEGER NOT NULL DEFAULT 0, cluster64 INTEGER, cluster400 INTEGER,
  created_at TEXT NOT NULL DEFAULT (datetime('now')), UNIQUE (source, source_id));

CREATE TABLE image_sources (image_id TEXT REFERENCES images(image_id), source TEXT, source_id TEXT, PRIMARY KEY (source, source_id));

CREATE TABLE embedding_models (model_key TEXT PRIMARY KEY, family TEXT, checkpoint TEXT, dim INTEGER, preprocess TEXT, created_at TEXT);

CREATE TABLE profiles (profile_id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, note TEXT, is_simulated INTEGER NOT NULL DEFAULT 0);

CREATE TABLE rec_models (version TEXT PRIMARY KEY, rung TEXT NOT NULL, embedding_key TEXT, params_json TEXT,
  artifact_path TEXT, trained_on_events INTEGER, metrics_json TEXT, is_active INTEGER NOT NULL DEFAULT 0, created_at TEXT);

CREATE TABLE impressions (impression_id INTEGER PRIMARY KEY, profile_id INTEGER REFERENCES profiles(profile_id),
  image_id TEXT REFERENCES images(image_id), model_version TEXT, experiment_id INTEGER, arm TEXT,
  batch_id TEXT, rank INTEGER, score REAL, sampled_score REAL, propensity REAL,
  reason TEXT, explanation_json TEXT, shown_at TEXT NOT NULL DEFAULT (datetime('now')));

CREATE TABLE events (event_id INTEGER PRIMARY KEY, impression_id INTEGER REFERENCES impressions(impression_id),
  profile_id INTEGER REFERENCES profiles(profile_id), image_id TEXT NOT NULL REFERENCES images(image_id),
  action TEXT NOT NULL CHECK (action IN ('like','dislike','skip','save')),
  dwell_ms INTEGER, client_ts TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now')),
  undone INTEGER NOT NULL DEFAULT 0, superseded_by INTEGER REFERENCES events(event_id));

CREATE TABLE interests (profile_id INTEGER, interest_id INTEGER, model_version TEXT, centroid BLOB,
  label TEXT, n_likes INTEGER, updated_at TEXT, PRIMARY KEY (profile_id, interest_id, model_version));

CREATE TABLE experiments (experiment_id INTEGER PRIMARY KEY, version_a TEXT, version_b TEXT, method TEXT,
  started_at TEXT, ended_at TEXT, result_json TEXT);

CREATE TABLE embedding_errors (image_id TEXT, model_key TEXT, error TEXT, at TEXT);

CREATE TABLE removals (image_id TEXT PRIMARY KEY, reason TEXT, at TEXT);

-- Indexes. Each one exists for a query that runs often enough to matter; the
-- feed's latency target is p95 < 50 ms, and at 50,000 images a missing index
-- here is the difference between a scan and a seek.

-- The feed's candidate pool: "selected and not yet seen". Without this, every
-- batch scans all 80,000 rows.
CREATE INDEX idx_images_selected_status ON images(selected, status);

-- The ingestion pipeline's per-source reconciliation, and the per-source counts
-- in the acquisition summary.
CREATE INDEX idx_images_source_status ON images(source, status);

-- Near-duplicate group lookups, and the Review page.
CREATE INDEX idx_images_dup_of ON images(dup_of) WHERE dup_of IS NOT NULL;
CREATE INDEX idx_images_phash ON images(phash) WHERE phash IS NOT NULL;

-- Cold start walks the 64 clusters; the Embedding Map colours by cluster.
CREATE INDEX idx_images_cluster64 ON images(cluster64) WHERE cluster64 IS NOT NULL;

-- "What has this profile already labelled" is read on every single feed
-- request, so it is the hottest query in the system.
CREATE INDEX idx_events_profile_image ON events(profile_id, image_id);

-- The learning curve, time-ordered replay, and "the latest verdict wins" all
-- read events in time order for one profile.
CREATE INDEX idx_events_profile_created ON events(profile_id, created_at);

-- Joining an event back to the impression that produced it, for propensity
-- and exploration-yield analysis.
CREATE INDEX idx_events_impression ON events(impression_id) WHERE impression_id IS NOT NULL;

-- Coverage and novelty metrics read the last N impressions for a profile.
CREATE INDEX idx_impressions_profile_shown ON impressions(profile_id, shown_at);
CREATE INDEX idx_impressions_image ON impressions(image_id);

-- Model Lab lists versions and finds the active one.
CREATE INDEX idx_rec_models_active ON rec_models(is_active) WHERE is_active = 1;
