"""
scheduler.py — Deterministic scheduling utilities for OpenSched.

Pure functions for time conversion, conflict detection,
and calendar manipulation. All times are same-day HH:MM strings.
"""

from app.models import Meeting, MeetingRequest, SchedulerAction


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------

def time_to_minutes(time_str: str) -> int:
    """Convert 'HH:MM' to minutes from midnight. e.g. '09:30' → 570."""
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


def minutes_to_time(minutes: int) -> str:
    """Convert minutes from midnight to 'HH:MM'. e.g. 570 → '09:30'."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def overlaps(start1: str, end1: str, start2: str, end2: str) -> bool:
    """Return True if two time intervals overlap (exclusive endpoints)."""
    a0, a1 = time_to_minutes(start1), time_to_minutes(end1)
    b0, b1 = time_to_minutes(start2), time_to_minutes(end2)
    return a0 < b1 and b0 < a1


def is_slot_free(calendar: list[Meeting], start: str, end: str) -> bool:
    """Return True if [start, end) does not conflict with any meeting."""
    return all(not overlaps(start, end, m.start, m.end) for m in calendar)


def count_conflicts(calendar: list[Meeting]) -> int:
    """Count the number of pairwise overlaps in the calendar."""
    n = len(calendar)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if overlaps(calendar[i].start, calendar[i].end,
                        calendar[j].start, calendar[j].end):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Calendar operations
# ---------------------------------------------------------------------------

def _sorted_calendar(calendar: list[Meeting]) -> list[Meeting]:
    """Return a copy of the calendar sorted by start time."""
    return sorted(calendar, key=lambda m: time_to_minutes(m.start))


def schedule_meeting(
    calendar: list[Meeting], meeting: Meeting
) -> tuple[list[Meeting], bool]:
    """
    Try to add a meeting to the calendar.

    Returns (updated_calendar, success).
    If the slot conflicts, returns the original calendar unchanged.
    """
    if not is_slot_free(calendar, meeting.start, meeting.end):
        return calendar, False

    updated = calendar + [meeting]
    return _sorted_calendar(updated), True


def reschedule_meeting(
    calendar: list[Meeting],
    meeting_title: str,
    new_start: str,
    new_end: str,
) -> tuple[list[Meeting], bool]:
    """
    Move an existing meeting to a new time slot.

    Finds the meeting by title, checks the new slot is free
    (ignoring the meeting being moved), and returns the updated calendar.
    Returns (original_calendar, False) if not found or conflicts.
    """
    # Find the meeting to move
    target = None
    rest: list[Meeting] = []
    for m in calendar:
        if m.title == meeting_title and target is None:
            target = m
        else:
            rest.append(m)

    if target is None:
        return calendar, False

    # Check that the new slot is free among remaining meetings
    if not is_slot_free(rest, new_start, new_end):
        return calendar, False

    # Create the rescheduled meeting, preserving all other fields
    moved = target.model_copy(update={"start": new_start, "end": new_end})
    return _sorted_calendar(rest + [moved]), True


def find_available_slots(
    calendar: list[Meeting],
    duration_minutes: int,
    work_start: str = "09:00",
    work_end: str = "18:00",
) -> list[tuple[str, str]]:
    """
    Return all free slots within work hours that fit the requested duration.

    Output is sorted chronologically and deterministic.
    """
    ws = time_to_minutes(work_start)
    we = time_to_minutes(work_end)

    # Collect busy intervals within work hours, sorted by start
    busy = sorted(
        (time_to_minutes(m.start), time_to_minutes(m.end))
        for m in calendar
        if time_to_minutes(m.end) > ws and time_to_minutes(m.start) < we
    )

    # Walk through gaps between busy blocks
    slots: list[tuple[str, str]] = []
    cursor = ws

    for block_start, block_end in busy:
        gap = block_start - cursor
        if gap >= duration_minutes:
            slots.append((minutes_to_time(cursor), minutes_to_time(block_start)))
        cursor = max(cursor, block_end)

    # Check trailing gap after last meeting
    if we - cursor >= duration_minutes:
        slots.append((minutes_to_time(cursor), minutes_to_time(we)))

    return slots


# ---------------------------------------------------------------------------
# Priority ranking (for conflict resolution)
# ---------------------------------------------------------------------------

_PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2}


# ---------------------------------------------------------------------------
# High-level resolve (used by inference.py)
# ---------------------------------------------------------------------------

def resolve(
    calendar: list[Meeting], request: MeetingRequest
) -> list[SchedulerAction]:
    """
    Decide how to handle an incoming meeting request.

    Returns a list of actions (may include reschedules + a final schedule).

    Strategy (deterministic):
      1. If preferred slot is free → schedule directly
      2. If preferred slot is blocked → try rescheduling lower/equal-priority
         blockers to free gaps, then schedule at the preferred time
      3. If rescheduling isn't enough → schedule in earliest free slot
      4. If nothing works → reject
    """
    actions: list[SchedulerAction] = []

    if not request.preferred_time:
        # No preference — just find the first free slot
        return _schedule_in_first_free(calendar, request)

    pref_start = request.preferred_time
    pref_end_min = time_to_minutes(pref_start) + request.duration_minutes
    pref_end = minutes_to_time(pref_end_min)

    # 1. Preferred slot already free
    if is_slot_free(calendar, pref_start, pref_end):
        return [SchedulerAction(
            action_type="schedule_new_meeting",
            meeting_title=request.title,
            start=pref_start,
            end=pref_end,
            reason="preferred slot is free",
        )]

    # 2. Try to clear the preferred slot by rescheduling blockers
    req_rank = _PRIORITY_RANK.get(request.priority, 1)
    blockers = [
        m for m in calendar
        if overlaps(pref_start, pref_end, m.start, m.end)
    ]

    # Only attempt if all blockers are <= request priority
    can_clear = all(
        _PRIORITY_RANK.get(b.priority, 1) <= req_rank for b in blockers
    )

    if can_clear and blockers:
        # Build a simulated calendar without the blockers
        remaining = [m for m in calendar if m not in blockers]
        reschedule_actions: list[SchedulerAction] = []
        all_ok = True

        for blocker in blockers:
            dur = time_to_minutes(blocker.end) - time_to_minutes(blocker.start)
            # Find a new home for this blocker (excluding the preferred slot)
            test_cal = remaining + [Meeting(
                title=request.title, start=pref_start, end=pref_end,
                priority=request.priority, attendees=request.attendees,
            )]
            gaps = find_available_slots(test_cal, dur)
            if gaps:
                new_start = gaps[0][0]
                new_end = minutes_to_time(time_to_minutes(new_start) + dur)
                reschedule_actions.append(SchedulerAction(
                    action_type="reschedule_existing_meeting",
                    meeting_title=blocker.title,
                    new_start=new_start,
                    new_end=new_end,
                    reason=f"moved to make room for '{request.title}'",
                ))
                # Add the rescheduled blocker to remaining for next iteration
                remaining.append(blocker.model_copy(
                    update={"start": new_start, "end": new_end}
                ))
            else:
                all_ok = False
                break

        if all_ok:
            actions.extend(reschedule_actions)
            actions.append(SchedulerAction(
                action_type="schedule_new_meeting",
                meeting_title=request.title,
                start=pref_start,
                end=pref_end,
                reason="preferred slot cleared by rescheduling",
            ))
            return actions

    # 3. Fall back to earliest free slot
    return _schedule_in_first_free(calendar, request)


def _schedule_in_first_free(
    calendar: list[Meeting], request: MeetingRequest
) -> list[SchedulerAction]:
    """Schedule in the earliest available slot, or reject."""
    available = find_available_slots(calendar, request.duration_minutes)
    if available:
        alt_start = available[0][0]
        alt_end = minutes_to_time(
            time_to_minutes(alt_start) + request.duration_minutes
        )
        return [SchedulerAction(
            action_type="schedule_new_meeting",
            meeting_title=request.title,
            start=alt_start,
            end=alt_end,
            reason="using earliest available slot",
        )]

    return [SchedulerAction(
        action_type="reject_request",
        meeting_title=request.title,
        reason="no available slot fits the requested duration",
    )]

