"""
utils.py — Shared helpers for OpenSched.

Note: Core scheduling utilities live in scheduler.py.
This module holds any additional helpers that don't fit elsewhere.
"""

from app.models import Meeting
from app.scheduler import overlaps


def meetings_overlap(a: Meeting, b: Meeting) -> bool:
    """Return True if two meetings overlap."""
    return overlaps(a.start, a.end, b.start, b.end)
