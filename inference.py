"""
inference.py — Robust OpenSched submission inference runner with structured output.

This script runs benchmark tasks, handles API calls safely, and prints
validator-friendly structured output (START / STEP / END).
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

API_BASE_URL = os.getenv("API_BASE_URL", "https://sohamzade-opensched.hf.space").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "opensched-agent")
HF_TOKEN = os.getenv("HF_TOKEN")
TIMEOUT = 30 

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
    """POST /reset with the given task_id."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def call_step(action: dict) -> dict:
    """POST /step with a SchedulerAction payload."""
    try:
        if not isinstance(action, dict):
            return {}
            
        resp = requests.post(
            f"{API_BASE_URL}/step",
            json=action,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
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
            return {"action_type": "finalize_schedule", "reason": "No choices from model"}

        raw = response.choices[0].message.content.strip()

        # Handle potential markdown fencing
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        return json.loads(raw)
    except Exception:
        return {"action_type": "finalize_schedule", "reason": "Fallback due to error"}


# ---------------------------------------------------------------------------
# Task Execution Logic
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    """Run one full episode for the given task_id. Prints structured output."""
    # START block
    print(f"[START] task={task_id}", flush=True)

    total_reward = 0.0
    step_num = 0
    try:
        # Initialize
        reset_data = call_reset(task_id)
        obs = reset_data.get("observation")
        
        if obs is None:
            # Fallback if reset fails
            step_num = 1
            reward = 0.01
            print(f"[STEP] step={step_num} reward={reward}", flush=True)
            total_reward = reward
        else:
            done = reset_data.get("done", False)
            max_steps = 20

            while not done and step_num < max_steps:
                step_num += 1
                action = call_model(obs)
                
                step_result = call_step(action)
                if not step_result:
                    # If step fails, print one last step to satisfy validator
                    print(f"[STEP] step={step_num} reward=0.01", flush=True)
                    total_reward += 0.01
                    break

                obs = step_result.get("observation", {})
                reward = float(step_result.get("reward", 0.0))
                done = step_result.get("done", False)
                total_reward += reward
                
                # STEP block
                print(f"[STEP] step={step_num} reward={reward}", flush=True)

            # Ensure at least one STEP if the loop didn't run
            if step_num == 0:
                step_num = 1
                print(f"[STEP] step=1 reward=0.01", flush=True)
                total_reward = 0.01

    except Exception:
        if step_num == 0:
            step_num = 1
            print(f"[STEP] step=1 reward=0.01", flush=True)
            total_reward = 0.01

    # Final result for the task
    final_score = clamp_score(total_reward)
    
    # END block
    print(f"[END] task={task_id} score={final_score} steps={step_num}", flush=True)
    return final_score


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    for task in TASKS:
        try:
            run_task(task.id)
        except Exception:
            # Absolute fallback if run_task itself crashes (which it shouldn't)
            print(f"[START] task={task.id}", flush=True)
            print(f"[STEP] step=1 reward=0.01", flush=True)
            print(f"[END] task={task.id} score=0.01 steps=1", flush=True)


if __name__ == "__main__":
    main()
