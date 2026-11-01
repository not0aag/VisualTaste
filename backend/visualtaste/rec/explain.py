"""Turning a reason code into a sentence a person would actually read.

Why this is its own file
------------------------
Each rung produces its own `Explanation` with a `reason` code and whatever
detail that rung has (brief §7.6). This module owns the *presentation*: one
place that knows how to say each reason in English, so the wording is
consistent across rungs and the frontend only has to render a string.

Keeping it separate also means the phrasing can be improved without touching a
model, and a model can be added without inventing new phrasing.

The five reason codes (§7.6)
----------------------------
    cold_start        still covering the collection; nothing learned yet
    similar_likes     close to specific images he liked (names up to three)
    interest:<k>      matches one of several interests (names it)
    explore           chosen to find out something, not because it scored well
    model_confident   the classifier gives it a high probability
    text_seed         close to the taste the person described in words (D4)

What the sentences avoid
------------------------
No flattery and no false precision. "We think you'll love this!" is both of
those: it claims a confidence the model does not have and a feeling it cannot
know. The sentences here say what the system did -- "close to three images you
liked", "showing you something different on purpose" -- because that is both
true and more useful. A user who understands why an image appeared can tell the
system something about it; a user who is being sold to cannot.

The `explore` wording matters most. An exploration slot is *expected* to be
worse on average, and saying so plainly is what makes a dislike there
informative rather than annoying.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Fixed reason codes, so the frontend can switch on them.
REASON_CODES = (
    "cold_start",
    "similar_likes",
    "explore",
    "model_confident",
    "text_seed",
)

#: `interest:<k>` is dynamic, so it is matched by prefix.
INTEREST_PREFIX = "interest:"


def sentence_for(reason: str, detail: dict[str, Any] | None = None) -> str:
    """One line of plain English for a reason code and its detail."""
    detail = detail or {}

    if reason == "cold_start":
        return (
            "Still learning what you like -- this is one of a spread of "
            "different kinds of image, chosen to cover the collection."
        )

    if reason == "similar_likes":
        count = len(detail.get("neighbours") or [])
        if count == 0:
            return "Pointing the same way as your likes so far."
        if count == 1:
            return "Close to an image you liked."
        return f"Close to {_number_word(count)} images you liked."

    if reason.startswith(INTEREST_PREFIX):
        label = detail.get("label")
        if label:
            return f"Matches one of your interests: {label}."
        return "Matches one of the separate things you seem to like."

    if reason == "explore":
        cluster_label = detail.get("cluster_label")
        if cluster_label:
            return (
                f"Shown on purpose to find out what you think of {cluster_label} -- "
                f"a part of the collection you have barely rated."
            )
        return (
            "Shown on purpose to find out something, not because it scored "
            "well. Rating it either way is useful."
        )

    if reason == "model_confident":
        probability = detail.get("probability")
        if isinstance(probability, (int, float)):
            return f"The model puts this at about a {round(probability * 100)}% chance of a like."
        return "The model scored this highly."

    if reason == "text_seed":
        text = detail.get("text")
        if text:
            return f'Close to what you described: "{text}".'
        return "Close to the taste you described."

    # An unknown code should still produce something readable rather than an
    # empty card, but it is a bug worth logging.
    logger.warning("no sentence for reason code %r", reason)
    return "Chosen by the current model."


def _number_word(count: int) -> str:
    """Small numbers as words, because "close to 3 images" reads oddly."""
    words = {2: "two", 3: "three", 4: "four", 5: "five"}
    return words.get(count, str(count))


def interest_reason(interest_id: int) -> str:
    """The reason code for a given interest, e.g. `interest:2`."""
    return f"{INTEREST_PREFIX}{interest_id}"


def interest_id_of(reason: str) -> int | None:
    """Parse the interest index back out of a reason code, if it is one."""
    if not reason.startswith(INTEREST_PREFIX):
        return None
    try:
        return int(reason[len(INTEREST_PREFIX) :])
    except ValueError:
        return None


def is_known_reason(reason: str) -> bool:
    """Whether this code is one the UI knows how to render."""
    return reason in REASON_CODES or interest_id_of(reason) is not None
