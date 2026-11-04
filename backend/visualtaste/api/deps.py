"""Application state: what is loaded once at startup, and what is per-request.

The 50 ms budget again
----------------------
`GET /feed/next` has to score every unseen candidate, and it cannot afford to
read 300 MB of vectors from disk to do it. So the embeddings, the FAISS indexes
and the active model are loaded **once** at startup and held in memory for the
life of the process:

    siglip2 vectors   50,000 x 768 float32   = 150 MB
    dinov2  vectors   50,000 x 768 float32   = 150 MB
    colour            50,000 x  64 float32   =  13 MB
    pca128            50,000 x 128 float32   =  26 MB

About 340 MB, on a machine with 15.4 GB. That is a deliberate trade: memory is
plentiful and latency is not.

What is *not* cached: the database connection. SQLite connections are not safe
to share across threads, so each request gets its own -- which is cheap (a file
handle and a few pragmas) and avoids a class of concurrency bug that would be
very hard to reproduce.

Offline by default
------------------
`HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are set before any model import
(brief §7.8). The weights are already cached by then, and the API should never
reach the network: if a cache entry were missing, failing immediately with a
clear error is better than an API that intermittently stalls on a download.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from visualtaste.config import Settings, get_settings
from visualtaste.db import repo
from visualtaste.rec.base import FeatureBundle
from visualtaste.rec.labels import label_count
from visualtaste.rec.r0_coldstart import ColdStartRecommender
from visualtaste.search.text import TextEncoder
from visualtaste.vision.encoders import DINOV2_KEY, SIGLIP2_KEY
from visualtaste.vision.store import EmbeddingStore

if TYPE_CHECKING:
    from collections.abc import Iterator

    from visualtaste.rec.base import Recommender
    from visualtaste.search.index import VectorIndex

logger = logging.getLogger(__name__)


def set_offline_env() -> None:
    """Forbid network access from the Hugging Face libraries.

    Called before any `transformers` or `open_clip` import. The variables are
    read at import time, so this has to happen first.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


@dataclass
class AppState:
    """Everything loaded once and shared across requests."""

    settings: Settings
    features: FeatureBundle | None = None
    indexes: dict[str, VectorIndex] = field(default_factory=dict)
    model: Recommender | None = None
    model_version: str = "r0_bootstrap"
    profile_id: int | None = None
    notes: list[str] = field(default_factory=list)

    #: R0, serving on behalf of the active rung until it has enough labels to
    #: be fitted. One instance for the life of the process, not one per
    #: request: it keeps a round-robin position over the 64 clusters, and a
    #: fresh instance every request would restart at cluster 0 and show the
    #: same corner of the collection forever.
    bootstrap: Recommender | None = None

    #: SigLIP 2's text tower for search and described tastes; it loads the
    #: model on its first use, not here (search/text.py).
    text_encoder: TextEncoder = field(default_factory=lambda: TextEncoder())

    #: Held for the whole of an online retrain, acquired without blocking, so
    #: two retrains can never be in flight at once (api/training.py).
    retrain_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Guards the `(model, model_version)` pair, which must change together.
    _serving_lock: threading.Lock = field(default_factory=threading.Lock)

    #: A running interleaving experiment: (experiment_id, {"a": (model,
    #: version), "b": (model, version)}). The feed team-drafts while it is set.
    experiment: tuple[int, dict[str, tuple[Recommender, str]]] | None = None

    # --- The serving model ---------------------------------------------------

    def serving(self) -> tuple[Recommender, str]:
        """The model and its version name, read together.

        Read as a pair under the lock because a retrain may swap both between
        two separate attribute reads, and an impression filed under the new
        version but scored by the old model would be a silent attribution error.
        """
        with self._serving_lock:
            return self.require_model(), self.model_version

    def swap(self, model: Recommender, version: str) -> None:
        """Replace the serving model atomically (Part 2 §5.5)."""
        with self._serving_lock:
            self.model = model
            self.model_version = version

    # --- Per-request database access ----------------------------------------

    def connect(self) -> sqlite3.Connection:
        """A fresh connection. Callers close it; `session()` does that for them."""
        return repo.connect(self.settings.db_path)

    def session(self) -> Iterator[sqlite3.Connection]:
        """FastAPI dependency: one connection per request, always closed."""
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    # --- Startup ------------------------------------------------------------

    def load(self) -> None:
        """Load vectors, indexes and the active model. Tolerates a partial build.

        Deliberately forgiving: during the build the embeddings may not exist
        yet, and an API that refuses to start until every artefact is present
        would make the frontend undevelopable. Anything missing is recorded in
        `notes` and surfaced by `GET /health`, so the gap is visible rather than
        mysterious.
        """
        set_offline_env()
        self.notes = []

        connection = self.connect()
        try:
            repo.verify_schema_version(connection)
            self._load_features(connection)
            self._load_indexes()
            self._load_model(connection)
            self._load_experiment(connection)
            self.bootstrap = ColdStartRecommender(seed=self.settings.seed)
            self._ensure_profile(connection)
        finally:
            connection.close()

    def _ensure_profile(self, connection: sqlite3.Connection) -> None:
        """Find or create the real profile, tolerating a busy database.

        A long-running ingestion or embedding job holds the write lock for
        seconds at a time, and the API starting up during one is a normal thing
        to do -- the whole point of building the frontend against a partial
        corpus. Crashing on `database is locked` would make that impossible, so
        the profile is left unset and created lazily by the first request that
        needs it.
        """
        try:
            self.profile_id = repo.active_profile_id(connection)
            if self.profile_id is None:
                self.profile_id = repo.create_profile(connection, note="first profile")
                connection.commit()
                logger.info("created profile %d", self.profile_id)
        except sqlite3.OperationalError as exc:
            self.profile_id = None
            self.notes.append(
                f"could not read or create a profile at startup ({exc}); a "
                f"background job is probably holding the write lock. The first "
                f"request that needs a profile will create one."
            )
            logger.warning(self.notes[-1])

    def _load_features(self, connection: sqlite3.Connection) -> None:
        """Assemble the `FeatureBundle` for the selected pool.

        **The pool, not the whole corpus.** 79,779 images are embedded but only
        50,000 are `selected = 1`, and the difference matters for two reasons.

        PCA-128 is fitted on the selected pool (brief §7.3), so it covers
        exactly those 50,000 rows. Building the bundle over all 79,779 made the
        PCA block incomplete, and because an incomplete block is dropped rather
        than padded, R3's entire feature space quietly disappeared -- visible
        only as a note on `GET /health`.

        And the feed's candidate pool *is* the selected pool, so indexing the
        bundle to it removes 30,000 rows that no request can ever use.

        SigLIP 2 defines the row order; every other block is aligned to it. A
        block missing rows is dropped entirely rather than padded, because
        padded zeros would reach a model looking like real features.
        """
        siglip_store = EmbeddingStore(self.settings.embedding_dir(SIGLIP2_KEY), SIGLIP2_KEY)
        if not siglip_store.vectors_path.exists():
            self.notes.append(
                f"no SigLIP 2 embeddings at {siglip_store.dir}; "
                f"run `python -m visualtaste.jobs.embed`"
            )
            logger.warning(self.notes[-1])
            return

        all_vectors, all_ids = siglip_store.load()
        pool = self._selected_pool(connection)

        if pool:
            keep = [i for i, image_id in enumerate(all_ids.tolist()) if image_id in pool]
            vectors = all_vectors[keep]
            image_ids = [all_ids[i] for i in keep]
        else:
            # Before `jobs.select` has run nothing is marked selected, so fall
            # back to everything embedded. Noted, because a feed drawing from
            # the unselected corpus is not the one the brief specifies.
            self.notes.append(
                "no images are marked selected; using the whole embedded corpus. "
                "Run `python -m visualtaste.jobs.select`."
            )
            logger.warning(self.notes[-1])
            vectors = all_vectors
            image_ids = all_ids.tolist()

        dinov2 = self._aligned_block(
            EmbeddingStore(self.settings.embedding_dir(DINOV2_KEY), DINOV2_KEY),
            image_ids,
            "dinov2",
        )
        colour = self._aligned_feature_file(image_ids, "colour64", "colour")
        pca = self._aligned_feature_file(image_ids, "pca128", "pca128")
        # Optional: only R3's feature ablation uses it, so its absence is not a
        # degraded state worth a note on /health.
        dinov2_pca = (
            self._aligned_feature_file(image_ids, "dinov2_pca128", "dinov2_pca128")
            if (self.settings.features_dir / "dinov2_pca128.npy").exists()
            else None
        )

        clusters = self._cluster_array(connection, image_ids)

        self.features = FeatureBundle(
            image_ids=image_ids,
            siglip=vectors,
            dinov2=dinov2,
            colour=colour,
            pca=pca,
            cluster64=clusters,
            # The `dino` feature block for R3's ablation (rec/features.py).
            extra={} if dinov2_pca is None else {"dinov2_pca": dinov2_pca},
        )
        # R6's profile feature 8: which collection each image came from.
        sources = dict(connection.execute("SELECT image_id, source FROM images").fetchall())
        names = sorted({sources.get(i, "") for i in image_ids})
        index = {name: k for k, name in enumerate(names)}
        self.features.extra["source_index"] = np.array(
            [index[sources.get(i, "")] for i in image_ids], dtype=np.float32
        )
        self._warm_design_matrices()
        logger.info(
            "loaded features for %d images (dinov2=%s colour=%s pca=%s)",
            len(image_ids),
            dinov2 is not None,
            colour is not None,
            pca is not None,
        )

    def _warm_design_matrices(self) -> None:
        """Standardise the feature blocks now, so no request pays for it.

        The first `/feed/next` served by R3 would otherwise build the
        standardised matrix itself -- tens of milliseconds, the whole budget,
        on exactly one unlucky request.
        """
        from visualtaste.rec.features import block_available, standardised_block

        assert self.features is not None
        for block in ("sig", "dino", "col"):
            if block_available(self.features, block):
                standardised_block(self.features, block, features_dir=self.settings.features_dir)

    def _selected_pool(self, connection: sqlite3.Connection) -> set[str]:
        """The image ids the feed may draw from: selected and still usable."""
        return {
            row["image_id"]
            for row in connection.execute(
                "SELECT image_id FROM images WHERE selected = 1 AND status = 'ok'"
            )
        }

    def _aligned_block(
        self, store: EmbeddingStore, image_ids: list[str], name: str
    ) -> np.ndarray | None:
        if not store.vectors_path.exists():
            self.notes.append(f"no {name} embeddings; features for it are unavailable")
            return None
        mapping = store.load_as_dict()
        return self._align(mapping, image_ids, name)

    def _aligned_feature_file(
        self, image_ids: list[str], stem: str, name: str
    ) -> np.ndarray | None:
        vectors_path = self.settings.features_dir / f"{stem}.npy"
        ids_path = self.settings.features_dir / f"{stem}_ids.npy"
        if not vectors_path.exists() or not ids_path.exists():
            self.notes.append(f"no {name} features at {vectors_path}")
            return None

        vectors = np.load(vectors_path)
        ids = np.load(ids_path, allow_pickle=False).tolist()
        return self._align(dict(zip(ids, vectors, strict=True)), image_ids, name)

    def _align(
        self, mapping: dict[str, np.ndarray], image_ids: list[str], name: str
    ) -> np.ndarray | None:
        missing = [i for i in image_ids if i not in mapping]
        if missing:
            self.notes.append(
                f"{name} is missing {len(missing)} of {len(image_ids)} images; "
                f"dropping the block rather than padding it with zeros"
            )
            logger.warning(self.notes[-1])
            return None
        return np.stack([mapping[i] for i in image_ids])

    def _cluster_array(
        self, connection: sqlite3.Connection, image_ids: list[str]
    ) -> np.ndarray | None:
        rows = connection.execute(
            "SELECT image_id, cluster64 FROM images WHERE cluster64 IS NOT NULL"
        ).fetchall()
        if not rows:
            self.notes.append("no cluster64 assignments; cold start will fall back to random")
            return None

        lookup = {row["image_id"]: int(row["cluster64"]) for row in rows}
        # -1 marks "no cluster", which `ColdStartRecommender` treats as its own
        # bucket rather than crashing on a missing key.
        return np.array([lookup.get(i, -1) for i in image_ids], dtype=np.int32)

    def _load_indexes(self) -> None:
        from visualtaste.search.index import VectorIndex

        for model_key in (SIGLIP2_KEY, DINOV2_KEY):
            store = EmbeddingStore(self.settings.embedding_dir(model_key), model_key)
            if not store.vectors_path.exists():
                continue
            vectors, ids = store.load()
            self.indexes[model_key] = VectorIndex.load(
                self.settings.index_path(model_key), vectors, ids
            )
            logger.info("loaded %s index (%d vectors)", model_key, len(ids))

    def _load_model(self, connection: sqlite3.Connection) -> None:
        """Load the active model, or start the default rung unfitted.

        With no active version, the configured `default_rung` is loaded empty:
        the feed's bootstrap serves R0 on its behalf until the first online
        retrain fits it (Part 2 §4.4). A rung that cannot be constructed or
        loaded falls back to cold start rather than failing startup.
        """
        from visualtaste.rec import registry

        active = registry.active_version(connection)
        if active is None:
            try:
                self.model = serving_rung(self.settings.default_rung, self.settings)
            except KeyError:
                self.notes.append(f"unknown default rung {self.settings.default_rung!r}")
                self.model = ColdStartRecommender(seed=self.settings.seed)
            self.model_version = "r0_bootstrap"
            logger.info(
                "no active model; %s will take over from cold start once it can be fitted",
                self.model.rung,
            )
            return

        try:
            self.model = serving_rung(active.rung, self.settings)
            artifacts = registry.artifact_dir(
                connection, active.version, models_root=self.settings.models_dir
            )
            self.model.load(artifacts)
            self.model_version = active.version
            logger.info("loaded active model %s (%s)", active.version, active.rung)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self.notes.append(
                f"could not load active model {active.version}: {exc}; serving cold start instead"
            )
            logger.exception("falling back to cold start")
            self.model = ColdStartRecommender(seed=self.settings.seed)
            self.model_version = "r0_bootstrap"

    def _load_experiment(self, connection: sqlite3.Connection) -> None:
        """Resume an interleaving experiment that was running at shutdown.

        Without this a restart would silently stop interleaving while the
        database still called the experiment active. If either version can no
        longer be loaded, the experiment is ended instead, with a note.
        """
        try:
            row = connection.execute(
                "SELECT * FROM experiments WHERE ended_at IS NULL ORDER BY experiment_id DESC"
            ).fetchone()
        except sqlite3.OperationalError:
            return
        if row is None:
            return
        try:
            arms = {
                arm: (
                    load_version(connection, self.settings, row[f"version_{arm}"]),
                    row[f"version_{arm}"],
                )
                for arm in ("a", "b")
            }
            self.experiment = (int(row["experiment_id"]), arms)
            logger.info("resumed interleaving experiment %d", row["experiment_id"])
        except (OSError, KeyError, ValueError) as exc:
            from visualtaste.eval.interleave import finish_experiment, result_from_impressions

            finish_experiment(
                connection,
                row["experiment_id"],
                result_from_impressions(connection, row["experiment_id"]),
            )
            self.notes.append(
                f"ended experiment {row['experiment_id']}: a version failed to load ({exc})"
            )

    # --- Queries used by more than one route --------------------------------

    def label_count(self, connection: sqlite3.Connection) -> int:
        if self.profile_id is None:
            return 0
        return label_count(connection, self.profile_id)

    def require_features(self) -> FeatureBundle:
        if self.features is None:
            raise FeaturesUnavailable(
                "no embeddings are loaded. Run `python -m visualtaste.jobs.embed` "
                "and restart the API. See GET /api/health for details."
            )
        return self.features

    def require_bootstrap(self) -> Recommender:
        """The cold-start model the feed falls back to. Created lazily for tests."""
        if self.bootstrap is None:
            self.bootstrap = ColdStartRecommender(seed=self.settings.seed)
        return self.bootstrap

    def require_model(self) -> Recommender:
        if self.model is None:  # pragma: no cover - load() always sets one
            self.model = ColdStartRecommender(seed=self.settings.seed)
        return self.model


class FeaturesUnavailable(RuntimeError):
    """The API is running but the embeddings it needs have not been built."""


class DatabaseBusy(RuntimeError):
    """A background job is holding the write lock for longer than we will wait."""


def ensure_profile(connection: sqlite3.Connection, state: AppState) -> int:
    """The current profile id, creating one if there is not one yet.

    Shared by every route that needs to write an event or an impression. A
    `database is locked` here means a long job (ingest, embed, select) is
    mid-transaction, which is a temporary and completely expected condition
    during the build -- so it becomes a 503 saying so, rather than a 500 with a
    traceback that looks like a bug.
    """
    if state.profile_id is not None:
        return state.profile_id

    try:
        profile_id = repo.active_profile_id(connection)
        if profile_id is None:
            profile_id = repo.create_profile(connection, note="auto")
            connection.commit()
    except sqlite3.OperationalError as exc:
        raise DatabaseBusy(
            f"the database is busy ({exc}). A background job is holding the "
            f"write lock; try again in a moment."
        ) from exc

    state.profile_id = profile_id
    return profile_id


_LABELLERS: dict[str, object] = {}


def concept_labeller(settings: Settings):
    """A function centroid -> "word, word, word", or None without a concept cache.

    Read from `<DATA>/concepts/` (built once by `python -m visualtaste.rec.
    concepts`). Cached per data root: the vectors are 637 x 768 and every
    interest model built in a process would otherwise reread them.
    """
    key = str(settings.data)
    if key not in _LABELLERS:
        from visualtaste.rec.concepts import cache_for, label_for

        cached = cache_for(settings).load()
        if cached is None:
            _LABELLERS[key] = None
        else:
            concepts, vectors = cached
            _LABELLERS[key] = lambda centroid: label_for(centroid, vectors, concepts)
    return _LABELLERS[key]


def load_version(connection: sqlite3.Connection, settings: Settings, version: str) -> Recommender:
    """A stored version, loaded from its artefacts. Raises KeyError or OSError.

    Shared by activation and interleaving, which both refuse a version that
    cannot be loaded rather than discovering it on the next request.
    """
    from visualtaste.rec import registry

    stored = registry.get_version(connection, version)
    if stored is None:
        raise KeyError(f"no such model version: {version}")
    model = serving_rung(stored.rung, settings)
    model.load(registry.artifact_dir(connection, version, models_root=settings.models_dir))
    return model


def load_rung(rung: str, settings: Settings) -> Recommender:
    """Construct an empty model of the named rung, ready to `load`.

    A lookup rather than a registry object, because the set of rungs is fixed by
    the brief and known at import time. Rungs are added here as they are built.
    """
    if rung == "r0":
        return ColdStartRecommender(seed=settings.seed)
    if rung == "r1":
        from visualtaste.rec.r1_rocchio import RocchioRecommender

        return RocchioRecommender()
    if rung == "r2":
        from visualtaste.rec.r2_knn import KnnRecommender

        return KnnRecommender()
    if rung == "r3":
        from visualtaste.rec.r3_logreg import LogisticRecommender

        return LogisticRecommender()
    if rung == "r4":
        from visualtaste.rec.r4_thompson import ThompsonRecommender

        return ThompsonRecommender(seed=settings.seed, tau_post=settings.r4_tau_post)
    if rung == "r5":
        from visualtaste.rec.r5_interests import InterestRecommender

        return InterestRecommender(seed=settings.seed, concept_labeller=concept_labeller(settings))
    if rung == "r6":
        from visualtaste.rec.r6_neural import NeuralRecommender

        return NeuralRecommender(min_labels=settings.neural_min_labels, seed=settings.seed)
    if rung in ("r3+r5", "r4+r5"):
        # R3 or R4 with R5's interest features as an extra block (Part 2 §7.1).
        if rung == "r3+r5":
            from visualtaste.rec.r3_logreg import LogisticRecommender as Base
        else:
            from visualtaste.rec.r4_thompson import ThompsonRecommender as Base
        from visualtaste.rec.features import DEFAULT_BLOCKS
        from visualtaste.rec.r3_logreg import INTEREST_BLOCK

        model = Base(blocks=(*DEFAULT_BLOCKS, INTEREST_BLOCK))
        model.rung = rung
        model.seed_for_interests = settings.seed
        model.concept_labeller = concept_labeller(settings)
        if rung == "r4+r5":
            model.seed = settings.seed
            model.tau_post = settings.r4_tau_post
        return model
    raise KeyError(f"unknown rung {rung!r}; it may be from a later phase than this build")


def serving_rung(rung: str, settings: Settings) -> Recommender:
    """`load_rung` plus the app's R4 options (`settings.r4_options`).

    The app builds every model through this; the simulator uses `load_rung`
    and spells its options out per study, so a change here never alters what
    a cached study means.
    """
    model = load_rung(rung, settings)
    if rung in ("r4", "r4+r5") and settings.r4_options:
        from visualtaste.eval.study import RungSpec

        for attribute, value in RungSpec.parse(f"{rung}:{settings.r4_options}").overrides:
            setattr(model, attribute, value)
    return model


# The single process-wide state. Created in `main.py` at startup rather than
# here, so that tests can build their own against a temporary directory.
_state: AppState | None = None


def set_state(state: AppState) -> None:
    global _state
    _state = state


def peek_state() -> AppState | None:
    """The injected state, or None. Used by `lifespan` to avoid clobbering it."""
    return _state


def get_state() -> AppState:
    if _state is None:  # pragma: no cover - main.py always sets it
        set_state(AppState(settings=get_settings()))
    assert _state is not None
    return _state


def get_db() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency for a per-request connection."""
    yield from get_state().session()
