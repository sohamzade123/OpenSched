"""
models.py — Core Pydantic data models for OpenSched.

Defines meetings, requests, observations, actions, and state
used across the scheduling benchmark environment.
"""

import re

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Reusable validation
# ---------------------------------------------------------------------------

_HH_MM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

ALLOWED_PRIORITIES = {"low", "medium", "high"}

ALLOWED_ACTIONS = {
    "schedule_new_meeting",
    "reschedule_existing_meeting",
    "reject_request",
    "suggest_alternative_slot",
    "finalize_schedule",
}


def _check_time(v: str | None, field_name: str) -> str | None:
    """Validate an optional HH:MM string."""
    if v is not None and not _HH_MM.match(v):
        raise ValueError(f"{field_name} must be in HH:MM format, got '{v}'")
    return v


# ---------------------------------------------------------------------------
# 1) Meeting
# ---------------------------------------------------------------------------

class Meeting(BaseModel):
    """An existing meeting on the calendar."""

    title: str = Field(..., description="Meeting title")
    start: str = Field(..., description="Start time in HH:MM format")
    end: str = Field(..., description="End time in HH:MM format")
    priority: str = Field("medium", description="Priority: low / medium / high")
    attendees: list[str] = Field(default_factory=list, description="List of attendee names")

    @field_validator("start", "end")
    @classmethod
    def _validate_time(cls, v: str) -> str:
        if not _HH_MM.match(v):
            raise ValueError(f"Must be HH:MM format, got '{v}'")
        return v

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, v: str) -> str:
        if v not in ALLOWED_PRIORITIES:
            raise ValueError(f"Must be one of {ALLOWED_PRIORITIES}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# 2) MeetingRequest
# ---------------------------------------------------------------------------

class MeetingRequest(BaseModel):
    """An incoming meeting request the agent must handle."""

    title: str = Field(..., description="Requested meeting title")
    duration_minutes: int = Field(..., gt=0, description="Duration in minutes (must be > 0)")
    priority: str = Field("medium", description="Priority: low / medium / high")
    attendees: list[str] = Field(default_factory=list, description="Required attendees")
    deadline_day: str | None = Field(None, description="Latest day to schedule by (e.g. '2026-04-10')")
    preferred_time: str | None = Field(None, description="Preferred start time in HH:MM format")

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, v: str) -> str:
        if v not in ALLOWED_PRIORITIES:
            raise ValueError(f"Must be one of {ALLOWED_PRIORITIES}, got '{v}'")
        return v

    @field_validator("preferred_time")
    @classmethod
    def _validate_preferred_time(cls, v: str | None) -> str | None:
        return _check_time(v, "preferred_time")


# ---------------------------------------------------------------------------
# 3) SchedulerObservation
# ---------------------------------------------------------------------------

class SchedulerObservation(BaseModel):
    """What the agent sees at each step."""

    calendar: list[Meeting] = Field(default_factory=list, description="Current scheduled meetings")
    incoming_request: MeetingRequest | None = Field(None, description="The request to handle")
    conflicts: int = Field(0, ge=0, description="Number of current conflicts")
    steps_taken: int = Field(0, ge=0, description="Steps taken so far in this episode")
    message: str = Field("", description="Human-readable status message")


# ---------------------------------------------------------------------------
# 4) SchedulerAction
# ---------------------------------------------------------------------------

class SchedulerAction(BaseModel):
    """An action the agent submits to the environment."""

    action_type: str = Field(..., description="Type of scheduling action to perform")

    # Optional fields — used depending on action_type
    meeting_title: str | None = Field(None, description="Title of the meeting to act on")
    start: str | None = Field(None, description="Start time in HH:MM")
    end: str | None = Field(None, description="End time in HH:MM")
    new_start: str | None = Field(None, description="New start time for rescheduling")
    new_end: str | None = Field(None, description="New end time for rescheduling")
    reason: str | None = Field(None, description="Reason for the action")

    @field_validator("action_type")
    @classmethod
    def _validate_action_type(cls, v: str) -> str:
        if v not in ALLOWED_ACTIONS:
            raise ValueError(f"Must be one of {ALLOWED_ACTIONS}, got '{v}'")
        return v

    @field_validator("start", "end", "new_start", "new_end")
    @classmethod
    def _validate_times(cls, v: str | None, info) -> str | None:
        return _check_time(v, info.field_name)


# ---------------------------------------------------------------------------
# 5) SchedulerStepResult
# ---------------------------------------------------------------------------

class SchedulerStepResult(BaseModel):
    """Returned by the environment after each step."""

    observation: SchedulerObservation = Field(..., description="New observation after the action")
    reward: float = Field(0.0, description="Reward for this step")
    done: bool = Field(False, description="Whether the episode is finished")
    info: dict = Field(default_factory=dict, description="Extra metadata")


# ---------------------------------------------------------------------------
# 6) SchedulerState
# ---------------------------------------------------------------------------

class SchedulerState(BaseModel):
    """Internal state of the scheduling environment."""

    scheduled_meetings: list[Meeting] = Field(default_factory=list, description="All confirmed meetings")
    pending_request: MeetingRequest | None = Field(None, description="Current unresolved request")
    conflicts: int = Field(0, ge=0, description="Number of active conflicts")
    steps_taken: int = Field(0, ge=0, description="Total steps in this episode")
    done: bool = Field(False, description="Whether the episode is complete")
    task_id: str | None = Field(None, description="ID of the current benchmark task")
