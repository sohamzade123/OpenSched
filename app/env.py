"""
env.py — Scheduling environment for OpenSched.

Exposes a reset() / step() interface for the AI agent.
Uses scheduler utilities for conflict detection and calendar updates.
"""

from app.models import (
    Meeting,
    MeetingRequest,
    SchedulerAction,
    SchedulerObservation,
    SchedulerState,
    SchedulerStepResult,
)
from app.scheduler import (
    count_conflicts,
    find_available_slots,
    minutes_to_time,
    reschedule_meeting,
    schedule_meeting,
    time_to_minutes,
)
from app.rewards import compute_reward
from app.tasks import Task, get_task


class SchedulingEnv:
    """
    Simulated scheduling environment.

    Loop: reset() → step(action) → step(action) → … until done.
    """

    DEFAULT_MAX_STEPS = 10

    def __init__(self) -> None:
        self.state = SchedulerState()
        self.max_steps: int = self.DEFAULT_MAX_STEPS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> SchedulerObservation:
        """
        Reset the environment.

        If task_id is provided, load that task's scenario.
        Otherwise fall back to a simple default scenario.
        """
        if task_id:
            task = get_task(task_id)
            if task is None:
                raise ValueError(f"Unknown task: {task_id}")
            return self._reset_from_task(task)

        # Default scenario (backwards compatible)
        self.state = SchedulerState(
            scheduled_meetings=[
                Meeting(title="Team Standup", start="09:00", end="09:30",
                        priority="high", attendees=["alice", "bob"]),
                Meeting(title="Design Review", start="10:00", end="11:00",
                        priority="medium", attendees=["alice", "carol"]),
            ],
            pending_request=MeetingRequest(
                title="Urgent Client Call", duration_minutes=60,
                priority="high", attendees=["alice", "dave"],
                preferred_time="09:00",
            ),
            conflicts=0,
            steps_taken=0,
            done=False,
        )
        self.max_steps = self.DEFAULT_MAX_STEPS
        self.state.conflicts = count_conflicts(self.state.scheduled_meetings)
        return self._observe()

    def step(self, action: SchedulerAction) -> SchedulerStepResult:
        """
        Apply an agent action and return (observation, reward, done, info).

        Supported action types:
          - schedule_new_meeting
          - reschedule_existing_meeting
          - reject_request
          - suggest_alternative_slot
          - finalize_schedule
        """
        if self.state.done:
            return self._result(0.0, info={"error": "episode already done"})

        self.state.steps_taken += 1
        info: dict = {"action_type": action.action_type}

        # Dispatch to handler
        reward = self._handle_action(action, info)

        # Recount conflicts after any calendar change
        self.state.conflicts = count_conflicts(self.state.scheduled_meetings)

        # Check termination
        if self.state.steps_taken >= self.max_steps:
            self.state.done = True
            info["termination"] = "max_steps_reached"
        if action.action_type in ("reject_request", "finalize_schedule"):
            self.state.done = True
        # Done if request was successfully scheduled
        if action.action_type == "schedule_new_meeting" and info.get("success"):
            self.state.pending_request = None
            self.state.done = True

        return self._result(reward, info=info)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_action(self, action: SchedulerAction, info: dict) -> float:
        """Dispatch an action and return the immediate reward."""
        cal = self.state.scheduled_meetings

        if action.action_type == "schedule_new_meeting":
            return self._do_schedule(action, info)

        if action.action_type == "reschedule_existing_meeting":
            return self._do_reschedule(action, info)

        if action.action_type == "reject_request":
            info["success"] = True
            return compute_reward(action, cal)

        if action.action_type == "suggest_alternative_slot":
            return self._do_suggest(action, info)

        if action.action_type == "finalize_schedule":
            info["success"] = True
            info["final_conflicts"] = count_conflicts(cal)
            return compute_reward(action, cal)

        info["error"] = f"unknown action: {action.action_type}"
        return -1.0

    def _do_schedule(self, action: SchedulerAction, info: dict) -> float:
        """Handle schedule_new_meeting."""
        if not action.start or not action.end:
            info["success"] = False
            info["error"] = "start and end required"
            return -0.5

        meeting = Meeting(
            title=action.meeting_title or "Untitled",
            start=action.start,
            end=action.end,
            priority=self.state.pending_request.priority if self.state.pending_request else "medium",
            attendees=self.state.pending_request.attendees if self.state.pending_request else [],
        )
        updated, ok = schedule_meeting(self.state.scheduled_meetings, meeting)
        info["success"] = ok

        if ok:
            self.state.scheduled_meetings = updated
            return compute_reward(action, updated)
        else:
            info["error"] = "slot conflicts with existing meeting"
            return -0.5

    def _do_reschedule(self, action: SchedulerAction, info: dict) -> float:
        """Handle reschedule_existing_meeting."""
        if not action.meeting_title or not action.new_start or not action.new_end:
            info["success"] = False
            info["error"] = "meeting_title, new_start, and new_end required"
            return -0.5

        updated, ok = reschedule_meeting(
            self.state.scheduled_meetings,
            action.meeting_title,
            action.new_start,
            action.new_end,
        )
        info["success"] = ok

        if ok:
            self.state.scheduled_meetings = updated
            return compute_reward(action, updated)
        else:
            info["error"] = "reschedule failed (not found or conflicts)"
            return -0.5

    def _do_suggest(self, action: SchedulerAction, info: dict) -> float:
        """Handle suggest_alternative_slot."""
        duration = (
            self.state.pending_request.duration_minutes
            if self.state.pending_request else 30
        )
        slots = find_available_slots(self.state.scheduled_meetings, duration)
        info["success"] = len(slots) > 0
        info["available_slots"] = slots
        return 0.0  # neutral — suggestion, not a calendar change

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_from_task(self, task: Task) -> SchedulerObservation:
        """Initialize state from a Task definition."""
        # Deep-copy meetings so tasks stay immutable
        calendar = [m.model_copy() for m in task.initial_calendar]
        request = task.incoming_request.model_copy() if task.incoming_request else None

        self.state = SchedulerState(
            scheduled_meetings=calendar,
            pending_request=request,
            conflicts=count_conflicts(calendar),
            steps_taken=0,
            done=False,
            task_id=task.id,
        )
        self.max_steps = task.max_steps
        return self._observe()

    def _observe(self) -> SchedulerObservation:
        """Build observation from current state."""
        return SchedulerObservation(
            calendar=self.state.scheduled_meetings,
            incoming_request=self.state.pending_request,
            conflicts=self.state.conflicts,
            steps_taken=self.state.steps_taken,
            message=self._status_message(),
        )

    def _status_message(self) -> str:
        """Generate a human-readable status."""
        if self.state.done:
            return "episode complete"
        if self.state.pending_request:
            return f"pending: {self.state.pending_request.title}"
        return "no pending requests"

    def _result(self, reward: float, info: dict | None = None) -> SchedulerStepResult:
        """Wrap observation + reward + done into a SchedulerStepResult."""
        return SchedulerStepResult(
            observation=self._observe(),
            reward=reward,
            done=self.state.done,
            info=info or {},
        )
