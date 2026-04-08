"""Tests for the scheduling environment."""

from app.env import SchedulingEnv
from app.models import SchedulerAction, SchedulerObservation, SchedulerStepResult


# ---------------------------------------------------------------------------
# Default scenario
# ---------------------------------------------------------------------------

def test_reset_returns_observation():
    env = SchedulingEnv()
    obs = env.reset()
    assert isinstance(obs, SchedulerObservation)
    assert len(obs.calendar) > 0
    assert obs.incoming_request is not None
    assert obs.steps_taken == 0


def test_step_returns_step_result():
    env = SchedulingEnv()
    env.reset()
    action = SchedulerAction(
        action_type="schedule_new_meeting",
        meeting_title="Test",
        start="14:00",
        end="15:00",
    )
    result = env.step(action)
    assert isinstance(result, SchedulerStepResult)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)


def test_step_schedule_success():
    """Scheduling into a free slot should succeed."""
    env = SchedulingEnv()
    env.reset()
    action = SchedulerAction(
        action_type="schedule_new_meeting",
        meeting_title="New Meeting",
        start="14:00",
        end="15:00",
    )
    result = env.step(action)
    assert result.info["success"] is True
    assert result.reward > 0


def test_step_schedule_conflict():
    """Scheduling into an occupied slot should fail."""
    env = SchedulingEnv()
    env.reset()
    action = SchedulerAction(
        action_type="schedule_new_meeting",
        meeting_title="Conflict",
        start="09:00",
        end="10:00",
    )
    result = env.step(action)
    assert result.info["success"] is False


def test_step_reject():
    """Rejecting a request should end the episode."""
    env = SchedulingEnv()
    env.reset()
    action = SchedulerAction(action_type="reject_request", meeting_title="X")
    result = env.step(action)
    assert result.done is True


# ---------------------------------------------------------------------------
# Task-based scenarios
# ---------------------------------------------------------------------------

def test_reset_with_task():
    env = SchedulingEnv()
    obs = env.reset(task_id="easy-conflict")
    assert obs.incoming_request is not None
    assert obs.incoming_request.title == "Urgent Board Call"
    assert len(obs.calendar) == 1


def test_reschedule_action():
    """Rescheduling a meeting to a free slot should succeed."""
    env = SchedulingEnv()
    env.reset(task_id="medium-reschedule")
    action = SchedulerAction(
        action_type="reschedule_existing_meeting",
        meeting_title="Design Review",
        new_start="11:00",
        new_end="12:00",
    )
    result = env.step(action)
    assert result.info["success"] is True
    assert result.observation.conflicts == 0
