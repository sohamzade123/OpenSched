"""
inference.py — Robust OpenSched submission inference runner.

This script runs benchmark tasks, handles API calls safely, and ensures
that scores conform to the strict (0, 1) range requirement.
"""

import os
import json
import requests
import traceback
from openai import OpenAI
from app.tasks import TASKS

# ---------------------------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------------------------

# Ensure the URL is clean (no trailing slash)
API_BASE_URL = os.getenv("API_BASE_URL", "https://sohamzade-opensched.hf.space").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "opensched-agent")
HF_TOKEN = os.getenv("HF_TOKEN")
TIMEOUT = 30  # Standard timeout for all API requests

# Initialize OpenAI-compatible client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy",
)

# ---------------------------------------------------------------------------
# Helper: Clamping Score
# ---------------------------------------------------------------------------

def clamp_score(x: float) -> float:
    """
    Ensures the score is strictly between 0 and 1.
    As per validator rules:
    - If score is <= 0, return 0.01
    - If score is >= 1, return 0.99
    - Handles invalid inputs by returning 0.5
    """
    try:
        x = float(x)
    except Exception:
        return 0.5

    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99
    return x

# ---------------------------------------------------------------------------
# Safe API Wrappers
# ---------------------------------------------------------------------------

def call_reset(task_id: str) -> dict:
    """POST /reset with the given task_id. Returns full response or empty dict."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"ERROR calling /reset for {task_id}: {e}")
        return {}


def call_step(action: dict) -> dict:
    """POST /step with a SchedulerAction payload. Returns response or empty dict."""
    try:
        # Action must be a dict
        if not isinstance(action, dict):
            print(f"ERROR: action is not a dict: {type(action)}")
            return {}
            
        resp = requests.post(
            f"{API_BASE_URL}/step",
            json=action,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"ERROR calling /step: {e}")
        return {}


def call_state() -> dict:
    """GET /state. Returns current world state or empty dict."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/state",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"ERROR calling /state: {e}")
        return {}


# ---------------------------------------------------------------------------
# Robust LLM Interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI scheduling assistant. You receive the current calendar and an \
incoming meeting request. You must return EXACTLY ONE valid JSON object \
representing the next scheduling action. Do NOT include any explanation, \
markdown fencing, or extra text — only the raw JSON object.

The JSON must conform to this schema:
{
  "action_type": "schedule_new_meeting" | "reschedule_existing_meeting" | "reject_request" | "suggest_alternative_slot" | "finalize_schedule",
  "meeting_title": string | null,
  "start": "HH:MM" | null,
  "end": "HH:MM" | null,
  "new_start": "HH:MM" | null,
  "new_end": "HH:MM" | null,
  "reason": string | null
}
"""

def call_model(observation: dict) -> dict:
    """Use the OpenAI client safely to generate the next scheduling action."""
    try:
        user_content = json.dumps(observation, indent=2)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            timeout=TIMEOUT
        )

        if not response.choices:
            print("ERROR: LLM returned no choices")
            return {"action_type": "finalize_schedule", "reason": "No choices from model"}

        raw = response.choices[0].message.content.strip()

        # Handle potential markdown fencing
        if raw.startswith("```"):
            # Strip first line and last line
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
            # Handle possible "json" tag
            if raw.startswith("json"):
                raw = raw[4:].strip()

        return json.loads(raw)
    except Exception as e:
        print(f"ERROR calling LLM or parsing JSON: {e}")
        # Return a fallback action to avoid breaking the loop
        return {"action_type": "finalize_schedule", "reason": "Fallback due to error"}


# ---------------------------------------------------------------------------
# Task Execution Logic
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    """Run one full episode for the given task_id. Returns clamped score."""
    print(f"START task={task_id}")

    total_reward = 0.0
    try:
        # Initialize the environment
        reset_data = call_reset(task_id)
        obs = reset_data.get("observation")
        
        if obs is None:
            print(f"ERROR: Could not get initial observation for {task_id}")
            return clamp_score(0.0)

        done = reset_data.get("done", False)
        step_num = 0
        max_steps = 20

        while not done and step_num < max_steps:
            step_num += 1
            
            # Predict action
            action = call_model(obs)
            print(f"STEP task={task_id} step={step_num} action={json.dumps(action)}")

            # Execute action
            step_result = call_step(action)
            
            if not step_result:
                print(f"ERROR: Step failed for task {task_id} at step {step_num}")
                break

            # Update state
            obs = step_result.get("observation", {})
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            total_reward += reward

    except Exception as e:
        print(f"ERROR: Unhandled exception in run_task for {task_id}: {e}")
        traceback.print_exc()

    # Final result for the task
    final_score = clamp_score(total_reward)
    print(f"END task={task_id} total_reward={final_score:.2f}")
    return final_score


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("OpenSched Inference Runner Start")
    
    results = {}
    for task in TASKS:
        try:
            score = run_task(task.id)
            results[task.id] = score
        except Exception as e:
            print(f"ERROR: Critical failure in main loop for task {task.id}: {e}")
            results[task.id] = clamp_score(0.0)

    print("\nBenchmark Summary:")
    for tid, score in results.items():
        print(f" - {tid}: {score:.4f}")
    print("Inference Runner Finished")


if __name__ == "__main__":
    main()
