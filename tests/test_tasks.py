"""Tests for task definitions."""

from app.tasks import TASKS, get_task


def test_tasks_not_empty():
    assert len(TASKS) > 0


def test_get_task_found():
    task = get_task("easy-conflict")
    assert task is not None
    assert task.id == "easy-conflict"


def test_get_task_not_found():
    assert get_task("nonexistent") is None
