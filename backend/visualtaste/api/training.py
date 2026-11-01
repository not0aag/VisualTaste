"""Online retraining: refit the serving rung in the background, then swap it in.

Part 2 §5.5. Every `VT_RETRAIN_EVERY` labels (20), `POST /events` schedules a
retrain as a FastAPI background task, which runs after the response has been
sent -- so the swipe that crossed the boundary is not kept waiting for it.

Three guarantees, each tested in `tests/test_training.py`:

1. **Requests during a retrain get the old model.** Training builds a brand-new
   model object and never touches the one serving. Only when the new one is
   fitted, saved and registered does `AppState.swap` replace the reference --
   one assignment, under a lock, of a `(model, version)` pair, so no request can
   ever see the new model with the old version name or the reverse.
2. **A failed retrain keeps the old model.** Any exception is logged and
   reported; the serving reference is untouched.
3. **Never two retrains in flight.** A non-blocking lock: a retrain that finds
   one already running is skipped, not queued. The next boundary, 20 labels
   later, will pick up everything both would have seen.

Which rung is retrained: the active one, when it is a rung that retrains online
(R1-R5). Otherwise -- R0 is serving, or nothing has been trained yet -- the
configured `default_rung`, which is how a fresh profile climbs from cold start
to R3 without anyone pressing a button. R6 is never retrained here; it trains in
a separate CLI job and is activated by hand (brief §7.5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from visualtaste.rec import registry
from visualtaste.rec.labels import build_label_set, label_weights
from visualtaste.vision.encoders import SIGLIP2_KEY

if TYPE_CHECKING:
    from visualtaste.api.deps import AppState

logger = logging.getLogger(__name__)

#: Rungs refitted by the online retrain. R0 learns nothing; R6 is CLI-only.
ONLINE_RUNGS = frozenset({"r1", "r2", "r3", "r4", "r5", "r3+r5", "r4+r5"})


@dataclass
class RetrainOutcome:
    """What a retrain did, for logs, the API response and the tests."""

    status: str  # "swapped" | "registered" | "not_fitted" | "busy" | "failed" | "no_profile"
    rung: str | None = None
    version: str | None = None
    detail: str = ""
    metrics: dict[str, Any] | None = None


def training_rung(state: AppState) -> str:
    """The rung a retrain should fit, per the module docstring."""
    current = state.model.rung if state.model is not None else "r0"
    if current in ONLINE_RUNGS and state.model_version != "r0_bootstrap":
        return current
    return state.settings.default_rung


def retrain(state: AppState, *, rung: str | None = None, activate: bool = True) -> RetrainOutcome:
    """Fit `rung` on the current profile's labels; register and swap it in.

    Safe to call from a background task, a route or startup. Never raises.
    """
    if not state.retrain_lock.acquire(blocking=False):
        logger.info("a retrain is already running; skipping this one")
        return RetrainOutcome(status="busy", detail="a retrain is already in flight")

    rung = rung or training_rung(state)
    connection = None
    try:
        if state.profile_id is None or state.features is None:
            return RetrainOutcome(status="no_profile", rung=rung, detail="nothing to train on")

        from visualtaste.api.deps import load_rung

        connection = state.connect()
        labels = build_label_set(
            connection,
            state.profile_id,
            weights=label_weights(state.settings.skip_weight),
        )
        model = load_rung(rung, state.settings)
        model.fit(labels, state.features)

        if not model.is_fitted:
            logger.info("retrain of %s left it unfitted: %s", rung, model.unfit_reason)
            # Never replace a fitted model with an unfitted one. But if nothing
            # fitted is serving, swap in this attempt anyway: its `unfit_reason`
            # ("30 positive and 0 negative labels") is what the feed's fallback
            # record should say, rather than "never fitted".
            if activate and not state.serving()[0].is_fitted:
                state.swap(model, "r0_bootstrap")
            return RetrainOutcome(status="not_fitted", rung=rung, detail=model.unfit_reason or "")

        report = getattr(model, "fit_report", None)
        metrics = report() if callable(report) else {"trained_on_events": len(labels)}
        version = registry.register(
            connection,
            model,
            models_root=state.settings.models_dir,
            embedding_key=SIGLIP2_KEY,
            metrics=metrics,
            activate=activate,
        )
        from visualtaste.rec.r5_interests import persist_interests, tracker_of

        tracker = tracker_of(model)
        if tracker is not None and tracker.state.k:
            persist_interests(connection, state.profile_id, version.version, tracker)
        if activate:
            state.swap(model, version.version)
            logger.info("retrained and swapped in %s", version.version)
            return RetrainOutcome("swapped", rung, version.version, metrics=metrics)
        return RetrainOutcome("registered", rung, version.version, metrics=metrics)

    except Exception as exc:
        # Guarantee 2: whatever went wrong, the serving model is untouched.
        logger.exception("retrain of %s failed; keeping the current model", rung)
        return RetrainOutcome(status="failed", rung=rung, detail=f"{type(exc).__name__}: {exc}")
    finally:
        if connection is not None:
            connection.close()
        state.retrain_lock.release()
