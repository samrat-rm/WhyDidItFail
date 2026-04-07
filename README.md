---
title: WhyDidItFail Environment Server
emoji: 🔍
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# WhyDidItFail — ML Training Failure Diagnosis Environment

An OpenEnv environment where an AI agent must diagnose why a machine learning training run failed. The agent inspects logs, configs, and gradient statistics to identify the root cause and suggest a fix.

## Overview

Real ML engineers spend significant time debugging failed training runs. This environment simulates that workflow: the agent receives partial observability (it must decide what to inspect) and must reason sequentially from evidence to diagnosis.

**12 realistic failure modes** across 3 difficulty tiers:
- **Easy**: identify failure from training logs only (loss/accuracy curves)
- **Medium**: identify failure from logs + hyperparameter config
- **Hard**: identify failure from logs + config + gradient norm data, and provide a concrete fix

## Failure Modes

| Category | Failure Mode |
|---|---|
| Optimization | exploding gradients, vanishing gradients, learning rate too high/low |
| Regularization | overfitting, missing regularization |
| Architecture | dying relu, bad weight initialization |
| Configuration | optimizer misconfiguration, batch size too small, lr scheduler misconfiguration |

## Action Space

| Action | Description |
|---|---|
| `inspect_logs` | View training/validation loss and accuracy curves by epoch |
| `inspect_config` | View hyperparameter config (lr, optimizer, batch size, dropout, etc.) |
| `inspect_gradients` | View gradient norm statistics by layer and epoch |
| `submit_diagnosis` | Submit final diagnosis with label, suggested fix, and reasoning |

## Observation Space

Each step returns a `WhyDidItFailObservation` with:
- `task_description` — the current task objective
- `visible_data` — data returned by the last inspect action (JSON)
- `feedback` — partial progress hint (e.g. which sources still need inspection)
- `steps_taken` — step counter
- `reward` — step-level reward
- `done` — episode termination flag

## Reward Function

Rewards are provided throughout the episode, not just at completion:

| Component | Weight | Signal |
|---|---|---|
| Diagnosis score | 0.70 | Correct failure mode label (exact match = 0.40 base, fuzzy = 0.10 per category keyword) |
| Evidence score | 0.15 | Inspected required sources; penalizes missing or irrelevant inspections |
| Efficiency score | 0.15 | Minimal steps to diagnosis; decays for wasted actions |
| Fix bonus | +0.15 | Keyword match on suggested fix (capped at 1.0 total) |

Step-level rewards during inspection: +0.10 / +0.07 / +0.05 for each required source discovered (decaying). Re-inspection: −0.05. Irrelevant inspection: −0.03.

## Tasks

### Task 1 — Easy (`task_easy`)
- **Objective**: Identify the failure mode from training logs only
- **Required sources**: `logs`
- **Max steps**: 10
- **Failure modes**: exploding gradients, learning rate too high, overfitting, underfitting

### Task 2 — Medium (`task_medium`)
- **Objective**: Identify the failure mode from logs + hyperparameter config
- **Required sources**: `logs`, `config`
- **Max steps**: 15
- **Failure modes**: learning rate too low, missing regularization, batch size too small, optimizer misconfiguration

### Task 3 — Hard (`task_hard`)
- **Objective**: Identify failure mode from logs + config + gradients, and provide a concrete fix
- **Required sources**: `logs`, `config`, `gradients`
- **Max steps**: 20
- **Failure modes**: vanishing gradients, dying relu, bad weight initialization, lr scheduler misconfiguration

## Baseline Performance (Qwen/Qwen2.5-72B-Instruct)

| Task | Avg Score | Pass Rate |
|---|---|---|
| Easy | ~0.85 | ~80% |
| Medium | ~0.92 | ~100% |
| Hard | ~0.93 | ~100% |

## Setup

### Environment Variables

| Variable | Default | Required |
|---|---|---|
| `HF_TOKEN` | — | Yes (mandatory) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | No |
| `SERVER_URL` | `http://localhost:8000` | No |

### Running Locally

```bash
# Install dependencies
uv sync

# Start the environment server
uvicorn server.app:app --reload

# Run inference (in another terminal)
HF_TOKEN=your_token uv run python inference.py
```

### Docker

```bash
docker build -t whydiditfail-env:latest .
docker run -p 8000:8000 whydiditfail-env:latest
```

## Project Structure

```
WhyDidItFail/
├── inference.py                    # Baseline inference script
├── client.py                       # WhyDidItFailEnv client (WebSocket)
├── models.py                       # Action and Observation Pydantic models
├── openenv.yaml                    # OpenEnv manifest
├── Dockerfile                      # Container image
└── server/
    ├── WhyDidItFail_environment.py # Core environment logic (step/reset/state)
    ├── app.py                      # FastAPI server (HTTP + WebSocket)
    ├── scenarios.py                # 12 scenario definitions
    ├── graders.py                  # Programmatic grader
    └── llm_judge.py                # LLM-based reasoning quality judge
```

## OpenEnv Spec Compliance

- Typed `Action`, `Observation` Pydantic models ✓
- `step(action)` → `(observation, reward, done, info)` ✓
- `reset()` → initial observation ✓
- `state()` → current state ✓
- `openenv.yaml` with 3 tasks and grader definitions ✓
- Passes `openenv validate` ✓