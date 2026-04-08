---
title: OpenSched
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860 
---

# OpenSched

**An OpenEnv-style benchmark for evaluating AI agents on calendar scheduling and conflict resolution.**

OpenSched presents an AI agent with a realistic calendar environment containing existing meetings and an incoming meeting request. The agent must resolve time conflicts through scheduling, rescheduling, or rejecting — and is scored on how well it handles the task.

---

## Features

- Deterministic calendar scheduling benchmark
- Interactive **reset / step** environment for AI agents
- Conflict detection and meeting rescheduling utilities
- Priority-aware scheduling logic
- Built-in deterministic resolver for baseline performance
- Reward-based environment for evaluating agent behavior
- Episode grading with outcome scoring
- FastAPI server for easy external agent integration
- Unit-tested scheduling and environment logic

---

## Problem Statement

Calendar scheduling is deceptively hard. Real-world scheduling involves:

- **Time conflicts** between overlapping meetings
- **Priority trade-offs** (should a low-priority sync yield to a high-priority board call?)
- **Chain reactions** (moving one meeting may require moving others)
- **Constraint satisfaction** (attendee availability, work-hour boundaries)

Most calendar tools handle this with simple rules. But an intelligent agent should be able to **reason about trade-offs**, plan **multi-step rescheduling**, and produce **conflict-free calendars** — just like a skilled executive assistant would.

**OpenSched provides a controlled, deterministic environment to test whether an AI agent can do this.**

---

## Why This Matters

Scheduling is a proxy for a broader class of **constraint-satisfaction + planning** problems.

An agent that performs well on OpenSched demonstrates:

- Multi-step planning under constraints
- Priority-based decision making
- State tracking across environment interactions
- Structured action selection
- Goal completion under limited steps

This makes OpenSched useful for evaluating:

- **LLM agents**
- **RL agents**
- **Tool-using autonomous systems**
- **Planning and decision-making models**

---

## What Makes OpenSched Different?

Unlike traditional calendar apps, OpenSched is designed as an **AI benchmark**, not a productivity tool.

It evaluates whether an agent can:

- reason about trade-offs,
- resolve scheduling conflicts,
- take multi-step actions,
- and produce valid final plans under constraints.

This makes it useful for benchmarking **reasoning**, **planning**, and **decision-making** capabilities in realistic scheduling scenarios.

---

## How the Environment Works

OpenSched follows a standard **reset / step** loop:

```python
obs = env.reset(task_id="easy-conflict")

while not done:
    action = agent.decide(obs)
    obs, reward, done, info = env.step(action)

report = grade_episode(results)
```

Each episode is a self-contained scheduling problem. The agent sees the current calendar and a pending request, takes actions, and receives rewards.

---

## Example Task

### Incoming Request

```json
{
  "title": "Board Call",
  "duration_minutes": 60,
  "priority": "high",
  "preferred_start": "10:00"
}
```

### Existing Calendar

```json
[
  {
    "title": "Design Review",
    "start": "10:00",
    "end": "11:00",
    "priority": "low"
  }
]
```

### Expected Agent Behavior

- Detect the conflict
- Move `Design Review` to `11:00–12:00`
- Schedule `Board Call` at `10:00–11:00`
- Finalize with zero conflicts

This type of task tests whether the agent can make a **priority-aware scheduling decision** while maintaining a valid final calendar.

---

## Observation Space

At each step, the agent receives a `SchedulerObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `calendar` | `list[Meeting]` | Current scheduled meetings (title, start, end, priority, attendees) |
| `incoming_request` | `MeetingRequest` | The request to handle (title, duration, priority, preferred time) |
| `conflicts` | `int` | Number of pairwise conflicts in the calendar |
| `steps_taken` | `int` | Steps used so far |
| `message` | `str` | Human-readable status |

---

## Action Space

The agent submits a `SchedulerAction` with one of 5 action types:

| Action Type | Required Fields | Effect |
|-------------|----------------|--------|
| `schedule_new_meeting` | `start`, `end` | Add the incoming request to the calendar |
| `reschedule_existing_meeting` | `meeting_title`, `new_start`, `new_end` | Move an existing meeting |
| `reject_request` | — | Reject the incoming request |
| `suggest_alternative_slot` | — | Query available time slots |
| `finalize_schedule` | — | End the episode |

---

## Reward Logic

| Outcome | Reward |
|---------|--------|
| Scheduled without conflicts | **+1.0** |
| Rescheduled without conflicts | **+0.5** |
| Finalized with zero conflicts | **+1.0** |
| Rejected a request | **-0.5** |
| Action created conflicts | **-1.0** |
| Finalized with remaining conflicts | **-0.5** |

The reward structure encourages agents to:

- resolve conflicts efficiently,
- avoid invalid scheduling,
- preserve preferred meeting times where possible,
- and complete tasks successfully.

---

## Grading

Each episode is graded on a **0.0 to 1.0 outcome score** based on:

1. Whether the requested meeting was successfully scheduled
2. Whether the final calendar is conflict-free
3. Whether the preferred time was preserved

Additional metrics include:

- total reward
- success rate
- action breakdown
- final conflict count

---

## Benchmark Tasks

| ID | Difficulty | Scenario | Expected Outcome |
|----|-----------|----------|------------------|
| `easy-conflict` | Easy | High-priority request overlaps a low-priority meeting | Reschedule blocker and schedule at preferred time |
| `medium-reschedule` | Medium | Conflict solvable by moving one meeting to a free slot | Move Design Review to `11:00`, schedule request at `10:00` |
| `hard-chain-reaction` | Hard | Tight schedule requiring multiple meetings to be shuffled | Relocate 2 meetings and fit a 2-hour workshop at `09:00` |

These tasks are intentionally designed to test increasing levels of **planning complexity**.

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the benchmark

```bash
python inference.py
```

This runs all benchmark tasks using the built-in deterministic resolver.

### Run tests

```bash
pytest tests/ -v
```

---

## Run the API Server

Start the FastAPI app locally:

```bash
uvicorn app.main:app --reload
```

Then open:

- API: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`

---

## Try It Quickly

Once the API server is running, try these endpoints in Swagger UI:

### 1) Health check

```http
GET /health
```

### 2) List available tasks

```http
GET /tasks
```

### 3) Run a benchmark task automatically

```http
POST /run/easy-conflict
```

### 4) Use the interactive environment manually

```http
POST /reset
```

Request body:

```json
{
  "task_id": "easy-conflict"
}
```

Then step through actions using:

```http
POST /step
```

Example action:

```json
{
  "action_type": "reschedule_existing_meeting",
  "meeting_title": "Design Review",
  "new_start": "11:00",
  "new_end": "12:00"
}
```

Finally, end the episode with:

```json
{
  "action_type": "finalize_schedule"
}
```

---

## API Endpoints

### Informational

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all benchmark tasks |
| `GET` | `/tasks/{id}` | Get full task details |

### Auto-run (Built-in Resolver)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/run/{id}` | Run a task with the deterministic resolver and return the grade |

### Interactive (For External Agents)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset the environment with a task |
| `POST` | `/step` | Submit one `SchedulerAction` |
| `GET` | `/state` | Check the current environment state |

---

## Project Structure

```text
app/
  main.py        # FastAPI app with API routes
  models.py      # Pydantic models and validation
  env.py         # Scheduling environment (reset / step)
  scheduler.py   # Scheduling utilities + multi-step resolver
  rewards.py     # Per-step reward computation
  graders.py     # Episode grading and outcome scoring
  tasks.py       # Benchmark task definitions
  utils.py       # Shared helpers

tests/
  test_env.py        # Environment tests
  test_scheduler.py  # Scheduler utility tests
  test_tasks.py      # Task registry tests

inference.py      # Benchmark runner
openenv.yaml      # Benchmark configuration
Dockerfile        # Container packaging
README.md         # Project documentation
```

---

## Run with Docker

### Build the image

```bash
docker build -t opensched .
```

### Run the container

```bash
docker run -p 8000:8000 opensched
```

Then visit:

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

---

## Tech Stack

- **Python 3.11+**
- **FastAPI**
- **Pydantic v2**
- **pytest**
- **Docker**

---

## Future Improvements

- Add attendee availability constraints
- Support recurring meetings
- Add hidden constraints for harder evaluation
- Introduce randomized benchmark generation
- Add leaderboard and benchmarking dashboard
- Support multi-agent scheduling scenarios

---

## Ideal Use Cases

OpenSched is useful for:

- benchmarking LLM agents
- testing scheduling policies
- evaluating reasoning under constraints
- building AI assistants for calendar automation
- demonstrating planning and decision-making in hackathons or research demos

---

## License

This project is open for educational and benchmark experimentation purposes.



Submission refresh checkvalidator refresh
 
