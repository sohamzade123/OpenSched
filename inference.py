"""
inference.py — Run scheduling episodes across all benchmark tasks.

Usage:  python inference.py
"""

from app.env import SchedulingEnv
from app.scheduler import resolve
from app.graders import grade_episode
from app.tasks import TASKS, get_task
from app.models import SchedulerStepResult


def run_task(task_id: str) -> dict:
    """Run a single task end-to-end using the deterministic resolver."""
    env = SchedulingEnv()
    obs = env.reset(task_id=task_id)

    results: list[SchedulerStepResult] = []

    while not env.state.done:
        if not obs.incoming_request:
            break

        # resolve() returns a list of actions (may include reschedules)
        actions = resolve(calendar=obs.calendar, request=obs.incoming_request)

        for action in actions:
            result = env.step(action)
            results.append(result)

            info = result.info
            status = "OK" if info.get("success") else "FAIL"
            print(f"    [{status}] {action.action_type}"
                  f"  reward={result.reward:+.1f}"
                  f"  conflicts={result.observation.conflicts}")

            obs = result.observation
            if result.done:
                break

        if env.state.done:
            break

    task = get_task(task_id)
    return grade_episode(results, final_calendar=obs.calendar, task=task)


def main():
    print("=" * 55)
    print("  OpenSched -- Benchmark Run")
    print("=" * 55)

    for task in TASKS:
        print(f"\n> [{task.difficulty.upper()}] {task.id}")
        print(f"  {task.description}")
        print(f"  Expected: {task.expected_outcome}")
        print()

        report = run_task(task.id)

        print(f"\n  Report:")
        for k, v in report.items():
            print(f"    {k}: {v}")
        print("  " + "-" * 40)


if __name__ == "__main__":
    main()
