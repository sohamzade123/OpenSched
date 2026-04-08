"""
graders.py — Episode-level evaluation for OpenSched.

Aggregates per-step results into a final report card.
Optionally compares against a task's expected_outcome.
"""

from app.models import Meeting, SchedulerStepResult
from app.scheduler import count_conflicts


def grade_episode(
    results: list[SchedulerStepResult],
    final_calendar: list[Meeting] | None = None,
    task=None,
) -> dict:
    """
    Grade agent performance over a full scheduling episode.

    Args:
        results: list of SchedulerStepResult from each env.step()
        final_calendar: the calendar state at the end of the episode
        task: optional Task object — if provided, scores against expected_outcome

    Returns:
        Dict with evaluation metrics including an overall score (0.0–1.0).
    """
    total_steps = len(results)
    total_reward = sum(r.reward for r in results)

    # Action type breakdown
    action_types = [r.info.get("action_type", "unknown") for r in results]
    scheduled = action_types.count("schedule_new_meeting")
    rescheduled = action_types.count("reschedule_existing_meeting")
    rejected = action_types.count("reject_request")
    suggestions = action_types.count("suggest_alternative_slot")

    # Success rate (actions that returned success=True in info)
    successes = sum(1 for r in results if r.info.get("success", False))

    # Final conflict count
    final_conflicts = (
        count_conflicts(final_calendar) if final_calendar is not None else -1
    )

    # Outcome score: compare final state against task.expected_outcome
    outcome_score = _score_outcome(final_calendar, task) if task else -1.0

    return {
        "total_steps": total_steps,
        "total_reward": round(total_reward, 2),
        "scheduled": scheduled,
        "rescheduled": rescheduled,
        "rejected": rejected,
        "suggestions": suggestions,
        "success_rate": round(successes / max(total_steps, 1), 2),
        "final_conflicts": final_conflicts,
        "outcome_score": outcome_score,
    }


def _score_outcome(
    final_calendar: list[Meeting] | None,
    task,
) -> float:
    """
    Score 0.0–1.0 how well the final calendar matches the task's expected outcome.

    Criteria (each worth equal weight):
      1. Was the incoming request actually scheduled? (title appears in calendar)
      2. Is the calendar conflict-free?
      3. Was the request placed at its preferred time?

    Returns -1.0 if scoring is not possible (missing data).
    """
    if final_calendar is None or task is None:
        return -1.0

    request = task.incoming_request
    if request is None:
        return 1.0  # nothing to schedule, trivially correct

    titles = [m.title for m in final_calendar]
    conflicts = count_conflicts(final_calendar)

    points = 0.0
    max_points = 3.0

    # 1. Was the request scheduled?
    if request.title in titles:
        points += 1.0

    # 2. Zero conflicts in final calendar?
    if conflicts == 0:
        points += 1.0

    # 3. Placed at preferred time?
    if request.preferred_time:
        match = next(
            (m for m in final_calendar
             if m.title == request.title and m.start == request.preferred_time),
            None,
        )
        if match is not None:
            points += 1.0
    else:
        # No preference stated — give the point for scheduling anywhere
        if request.title in titles:
            points += 1.0

    return round(points / max_points, 2)
