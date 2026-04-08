"""
tasks.py — Benchmark task definitions for OpenSched.
"""

from pydantic import BaseModel, Field
from app.models import Meeting, MeetingRequest


class Task(BaseModel):
    """A single benchmark scenario."""
    id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Human-readable description")
    difficulty: str = Field(..., description="easy, medium, or hard")
    initial_calendar: list[Meeting] = Field(default_factory=list, description="Meetings already in the schedule")
    incoming_request: MeetingRequest | None = Field(None, description="The request the agent must resolve")
    expected_outcome: str = Field(..., description="Clear criteria for success / grading")
    max_steps: int = Field(10, description="Max steps the agent can take")


TASKS: list[Task] = [
    Task(
        id="easy-conflict",
        description="High-priority request overlaps with a low-priority meeting block.",
        difficulty="easy",
        initial_calendar=[
            Meeting(title="Status Sync", start="09:00", end="10:00", priority="low", attendees=["bob"])
        ],
        incoming_request=MeetingRequest(
            title="Urgent Board Call",
            duration_minutes=60,
            priority="high",
            attendees=["alice", "bob"],
            preferred_time="09:00"
        ),
        expected_outcome="Accept 'Urgent Board Call' at 09:00, displacing or rejecting 'Status Sync'.",
        max_steps=5
    ),
    Task(
        id="medium-reschedule",
        description="A conflict that can be solved by moving an existing meeting to an available slot.",
        difficulty="medium",
        initial_calendar=[
            Meeting(title="Design Review", start="10:00", end="11:00", priority="medium", attendees=["alice"]),
            Meeting(title="Lunch", start="12:00", end="13:00", priority="low", attendees=["alice"])
        ],
        incoming_request=MeetingRequest(
            title="Strategic Planning",
            duration_minutes=60,
            priority="high",
            attendees=["alice"],
            preferred_time="10:00"
        ),
        expected_outcome="Schedule 'Strategic Planning' at 10:00; reschedule 'Design Review' to 11:00.",
        max_steps=10
    ),
    Task(
        id="hard-chain-reaction",
        description="Tight schedule where multiple meetings must be shuffled to fit a long, high-priority request.",
        difficulty="hard",
        initial_calendar=[
            Meeting(title="Standup", start="09:00", end="09:30", priority="medium", attendees=["alice", "bob"]),
            Meeting(title="Client Focus", start="09:30", end="11:00", priority="medium", attendees=["alice"]),
            Meeting(title="Quick Sync", start="11:00", end="11:30", priority="high", attendees=["bob"])
        ],
        incoming_request=MeetingRequest(
            title="Workshop",
            duration_minutes=120,
            priority="high",
            attendees=["alice", "bob"],
            preferred_time="09:00"
        ),
        expected_outcome="Fit 120min 'Workshop' starting at 09:00; requires relocating 'Standup' and 'Client Focus' after 11:30.",
        max_steps=15
    )
]


def get_task(task_id: str) -> Task | None:
    """Look up a task by ID."""
    return next((t for t in TASKS if t.id == task_id), None)
