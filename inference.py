"""
inference.py — OpenSched submission inference runner.

Runs all benchmark tasks by:
1. Resetting the environment via POST /reset
2. Calling an LLM (via OpenAI-compatible API) to generate scheduling actions
3. Submitting actions via POST /step
4. Logging structured output (START / STEP / END)
"""

import os
import json

import requests
from openai import OpenAI
from app.tasks import TASKS

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://sohamzade-opensched.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "opensched-agent")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ---------------------------------------------------------------------------
# OpenAI-compatible client (for LLM action generation)
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL.rstrip("/"),
    api_key=HF_TOKEN or "dummy",
)

# ---------------------------------------------------------------------------
# Helper: call environment endpoints
# ---------------------------------------------------------------------------

def call_reset(task_id: str) -> dict:
    """POST /reset with the given task_id. Returns the full reset response."""
    resp = requests.post(
        f"{API_BASE_URL.rstrip('/')}/reset",
        json={"task_id": task_id},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def call_step(action: dict) -> dict:
    """POST /step with a SchedulerAction payload. Returns step response."""
    resp = requests.post(
        f"{API_BASE_URL.rstrip('/')}/step",
        json=action,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Helper: ask the LLM for the next action
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI scheduling assistant. You receive the current calendar and an \
incoming meeting request. You must return EXACTLY ONE valid JSON object \
representing the next scheduling action. Do NOT include any explanation, \
markdown fencing, or extra text — only the raw JSON object.

The JSON must conform to this schema:
{
  "action_type": "<one of: schedule_new_meeting | reschedule_existing_meeting | reject_request | suggest_alternative_slot | finalize_schedule>",
  "meeting_title": "<string or null>",
  "start": "<HH:MM or null>",
  "end": "<HH:MM or null>",
  "new_start": "<HH:MM or null>",
  "new_end": "<HH:MM or null>",
  "reason": "<string or null>"
}

Rules:
- For schedule_new_meeting: provide meeting_title, start, end.
- For reschedule_existing_meeting: provide meeting_title (existing meeting), \
new_start, new_end.
- For reject_request: provide meeting_title and reason.
- Return ONLY the JSON object, nothing else.
"""


def call_model(observation: dict) -> dict:
    """Use the OpenAI client to generate the next scheduling action."""
    user_content = json.dumps(observation, indent=2)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fencing if the model wraps it
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    """Run one full episode for the given task_id. Returns total reward."""
    print(f"START task={task_id}")

    reset_data = call_reset(task_id)
    obs = reset_data["observation"]
    done = reset_data.get("done", False)

    total_reward = 0.0
    step_num = 0

    while not done:
        action = call_model(obs)
        step_num += 1

        print(f"STEP task={task_id} step={step_num} action={json.dumps(action)}")

        step_result = call_step(action)

        obs = step_result["observation"]
        reward = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        total_reward += reward

        if step_num >= 20:
            break

    print(f"END task={task_id} total_reward={total_reward:.2f}")
    return total_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for task in TASKS:
        run_task(task.id)


if __name__ == "__main__":
    main()
