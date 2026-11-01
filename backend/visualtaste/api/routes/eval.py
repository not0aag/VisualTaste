"""Start a simulator study from the Model Lab.

    POST /eval/run   {"study": "phase10_h3", "seeds": 20, "swipes": 1000}

A study takes minutes to hours, so it is never run inside the API process.
This starts `python -m visualtaste.jobs.evaluate` as a separate process with
its own log under `logs/`, and returns at once with the PID and the log path.
The result is appended to docs/EVALUATION.md when the study finishes, and
every seed is cached under `data/eval/`, so the run can be stopped and resumed.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from visualtaste.api.deps import AppState, get_state
from visualtaste.api.schemas import EvalRunRequest, EvalRunResponse

router = APIRouter(tags=["eval"])


def available_studies() -> list[str]:
    from visualtaste.jobs.evaluate import H2_STUDY, PRESETS

    return [*sorted(PRESETS), H2_STUDY, "phase10_checks", "phase12_h5"]


@router.get("/eval/studies", response_model=list[str])
def studies() -> list[str]:
    return available_studies()


@router.post("/eval/run", response_model=EvalRunResponse, status_code=202)
def run(
    payload: EvalRunRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> EvalRunResponse:
    if payload.study not in available_studies():
        raise HTTPException(status_code=400, detail=f"unknown study {payload.study!r}")

    settings = state.settings
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / f"eval_{payload.study}_{datetime.now(UTC):%Y%m%d_%H%M%S}.log"
    command = [
        sys.executable,
        "-m",
        "visualtaste.jobs.evaluate",
        "--study",
        payload.study,
        "--seeds",
        str(payload.seeds),
        "--swipes",
        str(payload.swipes),
    ]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=settings.project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    return EvalRunResponse(pid=process.pid, command=" ".join(command[1:]), log_path=str(log_path))
