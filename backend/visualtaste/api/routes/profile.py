"""The profile, and resetting it -- which archives rather than deletes.

Why reset does not delete
-------------------------
`POST /profile/reset` closes the current profile and starts a new one. The old
events stay exactly where they are, attached to the old `profile_id`.

Three reasons. The history is the only irreplaceable data in the project --
every image and embedding can be regenerated, those swipes cannot. A reset is
often a mistake, and an undoable mistake is a much better one. And the old
profile remains available to the evaluation harness, so a replay comparison can
still be run over it afterwards.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from visualtaste.api.deps import (
    AppState,
    DatabaseBusy,
    FeaturesUnavailable,
    ensure_profile,
    get_db,
    get_state,
)
from visualtaste.api.schemas import ProfileResponse, ResetResponse, TasteResponse
from visualtaste.db import repo
from visualtaste.rec.labels import build_label_set, label_count
from visualtaste.rec.r0_coldstart import coldstart_is_active

router = APIRouter(tags=["profile"])


@router.get("/profile", response_model=ProfileResponse)
def get_profile(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> ProfileResponse:
    try:
        profile_id = ensure_profile(connection, state)
    except DatabaseBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = connection.execute(
        "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    labels = label_count(connection, profile_id)
    counts = build_label_set(connection, profile_id).counts()
    model, version = state.serving()

    return ProfileResponse(
        profile_id=profile_id,
        started_at=row["started_at"] if row else "",
        is_simulated=bool(row["is_simulated"]) if row else False,
        labels=labels,
        counts=counts,
        active_model_version=version,
        # The rung actually serving: R0 while the configured one is unfitted.
        active_rung=model.rung if model.is_fitted else "r0",
        coldstart_active=coldstart_is_active(labels),
    )


@router.post("/profile/reset", response_model=ResetResponse)
def reset_profile(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> ResetResponse:
    """Close the current profile and begin a new, empty one.

    The old events are left untouched. Nothing is deleted here or anywhere
    else in this application.
    """
    previous = state.profile_id
    archived = label_count(connection, previous) if previous is not None else 0

    with repo.transaction(connection):
        if previous is not None:
            repo.end_profile(connection, previous)
        new_profile = repo.create_profile(
            connection, note=f"reset from profile {previous}" if previous else "first"
        )

    state.profile_id = new_profile

    # Back to cold start: a fresh profile has no labels, so any model trained
    # on the old one would be describing someone else's taste. The old version
    # is deactivated too, or a restart would load it straight back. The default
    # rung starts unfitted and takes over once the new profile can train it.
    from visualtaste.api.deps import load_rung
    from visualtaste.rec import registry

    registry.deactivate_all(connection)
    state.swap(load_rung(state.settings.default_rung, state.settings), "r0_bootstrap")

    return ResetResponse(
        previous_profile_id=previous,
        new_profile_id=new_profile,
        archived_labels=archived,
    )


#: Up to this many central images per interest card (Part 2 §7.2).
CENTRAL_IMAGES = 6


@router.get("/profile/taste", response_model=TasteResponse)
def get_taste(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> TasteResponse:
    """Interests and taste in words for the current profile.

    The interests are the serving model's when it has them (R5, R3+R5, R4+R5),
    so their ids are the stable ones. Otherwise they are fitted here from the
    current likes with the same tracker, which needs at least 50 likes.
    """
    import numpy as np

    from visualtaste.api.deps import concept_labeller
    from visualtaste.api.schemas import (
        ConceptWord,
        InterestCard,
        InterestEvent,
        LikeRateInterval,
    )
    from visualtaste.rec.concepts import cache_for, describe
    from visualtaste.rec.r1_rocchio import rocchio_vector
    from visualtaste.rec.r5_interests import (
        LIKES_PER_INTEREST,
        MIN_INTERESTS,
        InterestTracker,
        tracker_of,
    )
    from visualtaste.rec.rerank import CREDIBILITY

    try:
        profile_id = ensure_profile(connection, state)
        features = state.require_features()
    except (DatabaseBusy, FeaturesUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    labels = build_label_set(connection, profile_id)
    liked_ids = [i for i in labels.positives if features.position(i) is not None]
    disliked_ids = [i for i in labels.strong_negatives if features.position(i) is not None]
    liked = features.rows("siglip", liked_ids)
    disliked = features.rows("siglip", disliked_ids)

    model, version = state.serving()
    tracker = tracker_of(model) if model.is_fitted else None
    source = version
    note = ""
    if tracker is None or not tracker.state.k:
        tracker, source = InterestTracker(state.settings.seed), "computed"
        if len(liked) >= MIN_INTERESTS * LIKES_PER_INTEREST:
            tracker.update(liked, concept_labeller=concept_labeller(state.settings))
        else:
            note = (
                f"{len(liked)} likes so far; interests appear at "
                f"{MIN_INTERESTS * LIKES_PER_INTEREST} (one per {LIKES_PER_INTEREST} likes)."
            )
    interests = tracker.state

    cards: list[InterestCard] = []
    if interests.k and len(liked):
        from scipy.stats import beta as beta_distribution

        # Every rated image is assigned to its nearest interest: the likes and
        # dislikes there give that interest's Beta interval of like rate.
        rated = np.vstack([liked, disliked]) if len(disliked) else liked
        nearest = np.argmax(rated @ interests.centroids.T, axis=1)
        liked_nearest = nearest[: len(liked_ids)]
        tail = (1 - CREDIBILITY) / 2
        for index, interest_id in enumerate(interests.ids):
            members = np.flatnonzero(liked_nearest == index)
            closeness = liked[members] @ interests.centroids[index]
            central = [liked_ids[members[j]] for j in np.argsort(-closeness)[:CENTRAL_IMAGES]]
            n_like = len(members)
            n_dislike = int((nearest[len(liked_ids) :] == index).sum())
            a, b = 1 + n_like, 1 + n_dislike
            cards.append(
                InterestCard(
                    interest_id=int(interest_id),
                    label=interests.labels[index] if index < len(interests.labels) else "",
                    n_likes=n_like,
                    central_image_ids=central,
                    like_rate=LikeRateInterval(
                        likes=n_like,
                        dislikes=n_dislike,
                        mean=a / (a + b),
                        low=float(beta_distribution.ppf(tail, a, b)),
                        high=float(beta_distribution.ppf(1 - tail, a, b)),
                    ),
                )
            )

    likes_words: list[ConceptWord] = []
    dislikes_words: list[ConceptWord] = []
    cached = cache_for(state.settings).load()
    if cached is not None and len(liked):
        concepts, vectors = cached
        preference = rocchio_vector(liked, disliked)
        likes_words = [
            ConceptWord(concept=c.concept, z_score=c.z_score)
            for c in describe(preference, vectors, concepts)
        ]
        if len(disliked):
            dislikes_words = [
                ConceptWord(concept=c.concept, z_score=c.z_score)
                for c in describe(disliked.mean(axis=0), vectors, concepts)
            ]
    elif cached is None:
        note = (note + " " if note else "") + (
            "No concept vocabulary cache; run `python -m visualtaste.rec.concepts`."
        )

    return TasteResponse(
        profile_id=profile_id,
        likes=len(liked_ids),
        dislikes=len(disliked_ids),
        source=source,
        interests=cards,
        likes_in_words=likes_words,
        dislikes_in_words=dislikes_words,
        history=[
            InterestEvent(likes=h["likes"], born=h["born"], retired=h["retired"])
            for h in interests.history
        ],
        note=note,
    )
