"""
main.py — FastAPI entry point for OpenSched.

Provides API routes for:
- running benchmark tasks,
- interactive agent sessions,
- checking environment state.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.env import SchedulingEnv
from app.scheduler import resolve
from app.graders import grade_episode
from app.tasks import TASKS, get_task
from app.models import SchedulerAction, SchedulerStepResult

app = FastAPI(
    title="OpenSched",
    description="OpenEnv-style benchmark for AI-driven calendar scheduling.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Shared session env (for interactive /reset → /step → /state flow)
# NOTE:
# This benchmark uses a single in-memory session for simplicity.
# In production, session state should be isolated per user/client.
# ---------------------------------------------------------------------------

_session_env = SchedulingEnv()
_session_results: list[SchedulerStepResult] = []
_session_task_id: str | None = None


# ---------------------------------------------------------------------------
# System / Health Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"], summary="API root")
async def root():
    """Basic root endpoint for quick API discovery."""
    return {
        "message": "Welcome to OpenSched API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["System"], summary="Health check")
async def health_check():
    """Return API health status."""
    return {
        "status": "ok",
        "service": "OpenSched API",
        "version": "0.1.0",
    }


# ---------------------------------------------------------------------------
# Task Routes
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["Tasks"], summary="List all tasks")
async def list_tasks():
    """List all available benchmark tasks."""
    return [
        {"id": t.id, "difficulty": t.difficulty, "description": t.description}
        for t in TASKS
    ]


@app.get("/tasks/{task_id}", tags=["Tasks"], summary="Get task details")
async def get_task_detail(task_id: str):
    """Get full details for a specific benchmark task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task.model_dump()


# ---------------------------------------------------------------------------
# Auto-run Endpoint
# ---------------------------------------------------------------------------

@app.post("/run/{task_id}", tags=["Benchmark"], summary="Run benchmark task")
async def run_task(task_id: str):
    """Run a task using the built-in deterministic resolver and return the grade."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    env = SchedulingEnv()
    obs = env.reset(task_id=task_id)

    results: list[SchedulerStepResult] = []

    while not env.state.done:
        if not obs.incoming_request:
            break

        actions = resolve(calendar=obs.calendar, request=obs.incoming_request)

        if not actions:
            break

        for action in actions:
            result = env.step(action)
            results.append(result)
            obs = result.observation
            if result.done:
                break

        if env.state.done:
            break

    report = grade_episode(results, final_calendar=obs.calendar, task=task)

    return {
        "task_id": task_id,
        "difficulty": task.difficulty,
        "expected_outcome": task.expected_outcome,
        "report": report,
        "final_calendar": [m.model_dump() for m in obs.calendar],
    }


# ---------------------------------------------------------------------------
# Interactive Session Endpoints
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset."""
    task_id: str | None = Field(
        default=None,
        description="Benchmark task ID to load (e.g. 'easy-conflict')"
    )


@app.post("/reset", tags=["Interactive"], summary="Reset environment")
async def reset_env(body: ResetRequest | None = None):
    """
    Reset the environment and return the initial observation.

    If task_id is provided, loads that benchmark task.
    Otherwise loads the default scenario.
    """
    global _session_results, _session_task_id

    task_id = body.task_id if body else None
    _session_task_id = task_id
    _session_results = []

    if task_id and get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    try:
        obs = _session_env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "done": False,
        "task_id": task_id,
    }


@app.post("/step", tags=["Interactive"], summary="Submit one action")
async def step_env(action: SchedulerAction):
    """
    Submit a single action to the environment and get the result.

    The agent sends one SchedulerAction per call.
    Returns observation, reward, done, and info.
    """
    if _session_env.state.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call POST /reset first.")

    result = _session_env.step(action)
    _session_results.append(result)

    response = {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }

    if result.done:
        task = get_task(_session_task_id) if _session_task_id else None
        report = grade_episode(
            _session_results,
            final_calendar=result.observation.calendar,
            task=task,
        )
        response["report"] = report

    return response


@app.get("/state", tags=["Interactive"], summary="Get current environment state")
async def get_state():
    """Return the current environment state and session info."""
    return {
        "state": _session_env.state.model_dump(),
        "steps_taken": _session_env.state.steps_taken,
        "done": _session_env.state.done,
        "task_id": _session_task_id,
        "actions_so_far": len(_session_results),
    }