"""
rewards.py — Reward computation for OpenSched.

Simple, deterministic reward signals based on action outcomes.
"""

from app.models import Meeting, SchedulerAction
from app.scheduler import count_conflicts

# Reward constants
SCHEDULE_SUCCESS = 1.0
RESCHEDULE_SUCCESS = 0.5
REJECT_PENALTY = -0.5
CONFLICT_PENALTY = -1.0
FINALIZE_CLEAN = 1.0
FINALIZE_DIRTY = -0.5


def compute_reward(action: SchedulerAction, calendar: list[Meeting]) -> float:
    """
    Score a single scheduling action based on its type and calendar state.

    Rewards:
      +1.0  — successfully scheduled a new meeting (no conflicts)
      +0.5  — successfully rescheduled an existing meeting
      -0.5  — rejected a request
      -1.0  — action left conflicts in the calendar
      +1.0  — finalized with zero conflicts
      -0.5  — finalized with remaining conflicts
    """
    conflicts = count_conflicts(calendar)

    if action.action_type == "schedule_new_meeting":
        return SCHEDULE_SUCCESS if conflicts == 0 else CONFLICT_PENALTY

    if action.action_type == "reschedule_existing_meeting":
        return RESCHEDULE_SUCCESS if conflicts == 0 else CONFLICT_PENALTY

    if action.action_type == "reject_request":
        return REJECT_PENALTY

    if action.action_type == "finalize_schedule":
        return FINALIZE_CLEAN if conflicts == 0 else FINALIZE_DIRTY

    # suggest_alternative_slot or unknown — neutral
    return 0.0
