"""Tests for scheduler utility functions."""

from app.scheduler import (
    count_conflicts,
    find_available_slots,
    is_slot_free,
    minutes_to_time,
    overlaps,
    reschedule_meeting,
    schedule_meeting,
    time_to_minutes,
)
from app.models import Meeting


def test_time_to_minutes():
    assert time_to_minutes("00:00") == 0
    assert time_to_minutes("09:30") == 570
    assert time_to_minutes("23:59") == 1439


def test_minutes_to_time():
    assert minutes_to_time(0) == "00:00"
    assert minutes_to_time(570) == "09:30"
    assert minutes_to_time(1439) == "23:59"


def test_overlaps():
    assert overlaps("09:00", "10:00", "09:30", "10:30") is True
    assert overlaps("09:00", "10:00", "10:00", "11:00") is False  # adjacent
    assert overlaps("09:00", "10:00", "07:00", "08:00") is False


def test_is_slot_free():
    cal = [Meeting(title="A", start="09:00", end="10:00", priority="medium")]
    assert is_slot_free(cal, "10:00", "11:00") is True
    assert is_slot_free(cal, "09:30", "10:30") is False
    assert is_slot_free([], "09:00", "10:00") is True


def test_count_conflicts():
    cal = [
        Meeting(title="A", start="09:00", end="10:00", priority="medium"),
        Meeting(title="B", start="11:00", end="12:00", priority="medium"),
    ]
    assert count_conflicts(cal) == 0

    cal.append(Meeting(title="C", start="09:30", end="10:30", priority="high"))
    assert count_conflicts(cal) == 1


def test_schedule_meeting_success():
    cal = [Meeting(title="A", start="09:00", end="10:00", priority="medium")]
    new = Meeting(title="B", start="10:00", end="11:00", priority="medium")
    updated, ok = schedule_meeting(cal, new)
    assert ok is True
    assert len(updated) == 2


def test_schedule_meeting_conflict():
    cal = [Meeting(title="A", start="09:00", end="10:00", priority="medium")]
    new = Meeting(title="B", start="09:30", end="10:30", priority="high")
    unchanged, ok = schedule_meeting(cal, new)
    assert ok is False
    assert unchanged is cal


def test_reschedule_meeting_success():
    cal = [
        Meeting(title="A", start="09:00", end="10:00", priority="medium"),
        Meeting(title="B", start="11:00", end="12:00", priority="medium"),
    ]
    updated, ok = reschedule_meeting(cal, "A", "14:00", "15:00")
    assert ok is True
    assert any(m.start == "14:00" for m in updated)


def test_reschedule_meeting_conflict():
    cal = [
        Meeting(title="A", start="09:00", end="10:00", priority="medium"),
        Meeting(title="B", start="11:00", end="12:00", priority="medium"),
    ]
    _, ok = reschedule_meeting(cal, "A", "11:00", "12:00")
    assert ok is False


def test_find_available_slots():
    cal = [
        Meeting(title="A", start="09:00", end="10:00", priority="medium"),
        Meeting(title="B", start="11:00", end="12:00", priority="medium"),
    ]
    slots = find_available_slots(cal, 60)
    assert ("10:00", "11:00") in slots
    assert ("12:00", "18:00") in slots


def test_find_available_slots_empty_calendar():
    slots = find_available_slots([], 30)
    assert len(slots) == 1
    assert slots[0] == ("09:00", "18:00")
