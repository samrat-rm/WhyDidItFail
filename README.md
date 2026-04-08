---
title: WhyDidItFail Environment Server
emoji: 🔬
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/5ea68b2e-0005-4aab-93db-6206ef29ad91" />


# 🔬 WhyDidItFail — ML Training Failure Diagnosis Environment

> Every dev has been there. It's 2am. Your training run just died. The loss curve looks like a seismograph during an earthquake. You have no idea why. **WhyDidItFail** puts an AI agent in that exact seat — and makes it figure out what went wrong.

This is a real-world OpenEnv environment where an AI agent must **diagnose failed ML training runs** by inspecting logs, configs, and gradient statistics — then commit to a root cause and a fix. No handholding, no free answers. Just evidence, reasoning, and a score.

**Built as a training target for small models (7B–13B).** The action space is constrained, the reward signal is dense, and the failure modes are realistic enough to stress-test any agent that thinks it knows ML.

---

## How It Works — Episode Lifecycle

```
reset()
  └─► Agent receives task description + hint
        │
        ▼
   [inspect_logs]  ──► Observation: training curves (loss, acc per epoch)
        │                Reward: +0.10 (first required source)
        ▼
   [inspect_config] ──► Observation: hyperparams (lr, optimizer, dropout...)
        │                Reward: +0.07 (second required source)
        ▼
   [inspect_gradients] ► Observation: gradient norms by layer
        │                Reward: +0.05 (third required source)
        ▼
   [submit_diagnosis] ──► diagnosis + suggested_fix + reasoning
                          Reward: 0.0–1.0 (graded on correctness + evidence + efficiency)
                          done = True
```

> - Each action reveals a different slice of evidence
> - Agent must decide what to inspect and when to stop
> - Submitting too early or too late both cost points
> - Wrong inspection sources penalize the score


---

## Scenarios

### Easy — Logs only

| Scenario | Problem | Description |
|---|---|---|
| Exploding Gradients | Loss → NaN | Training loss goes NaN after epoch 2. Gradient norms spike to infinity. The model diverges catastrophically — a classic sign of learning rate being too high or missing gradient clipping. The agent must catch NaN in the logs and label it correctly. |
| Learning Rate Too High | Oscillating loss | Loss bounces wildly every epoch — goes down, shoots up, never converges. No NaN, just chaos. The optimizer is taking steps so large it overshoots the minimum repeatedly. Batch size is fine; the culprit is the LR. |
| Overfitting | Val loss climbing | Train loss hits near-zero by epoch 15. Val loss is diverging upward. The config already has regularization (dropout, weight decay) present — so this is true overfitting, not a missing regularization bug. The agent must distinguish these two. |
| Underfitting | Both losses stuck high | Train accuracy and val accuracy hover near random baseline (~10%) throughout training. No gap between them. The model isn't learning at all — too simple for the task, wrong architecture, or training stopped too early. |

### Medium — Logs + Config

| Scenario | Problem | Description |
|---|---|---|
| Learning Rate Too Low | Glacial convergence | Loss is decreasing, but imperceptibly — 0.001 per epoch. The config reveals `lr=1e-6`. The model is technically learning, but so slowly it would take thousands of epochs to converge. The agent needs both the log trend and the config LR to make this call. |
| Missing Regularization | Overfit without defense | Train loss low, val loss rising — looks like overfitting. But the config shows `weight_decay=0.0` and `dropout=0.0`. This isn't overfitting the model fighting the regularizer — it's the model memorizing because there's no regularizer at all. Label matters here. |
| Batch Size Too Small | Noisy gradient trajectory | Loss goes down on average but is extremely noisy — spikes and dips every epoch. Config shows `batch_size=2`. With tiny batches, gradient estimates are high-variance: each update is basically random. The agent must connect the noise pattern to the batch size config. |
| Optimizer Misconfiguration | SGD with no momentum | Loss curves look stuck or very slow. Config shows `optimizer=SGD, momentum=0.0`. SGD without momentum has no gradient averaging — it stalls on saddle points and flat regions. Modern SGD needs momentum to navigate loss landscapes effectively. |

### Hard — Logs + Config + Gradients

| Scenario | Problem | Description |
|---|---|---|
| Vanishing Gradients | Gradient decay toward inputs | Gradient norms decay exponentially from output to input layers (e.g. 1e-1 → 1e-8). Config shows sigmoid or tanh activation. These saturating activations crush gradients during backprop. The input layers learn nothing. Agent must read gradient norms by layer and connect to activation choice. |
| Dying ReLU | Zero gradients in hidden layers | Gradient norms in hidden layers are exactly 0.0 — not small, exactly zero. Config shows ReLU activation and a high learning rate. Neurons have permanently entered the "dead zone" where their pre-activation is always negative, so they never fire or update again. |
| Bad Weight Initialization | NaN from epoch 1 | Loss is NaN from the very first epoch — before training even begins meaningfully. Gradient norms are astronomically large (>10,000). Config shows an extreme weight initialization std (e.g. 100). Weights so large that the forward pass immediately overflows. |
| LR Scheduler Misconfiguration | Periodic loss spikes | Training goes fine, then suddenly loss spikes at a predictable interval — every N epochs. Config shows `lr_scheduler=StepLR, gamma=10.0`. Gamma > 1.0 means the scheduler is **increasing** the learning rate at each step, not decreasing it. A subtle config error with dramatic consequences. |

---

## Features

- **12 realistic failure modes** across 3 difficulty tiers — exploding gradients, overfitting, dying ReLU, bad weight initialization, and more
- **Partial observability** — the agent chooses what to inspect (logs, config, gradients) and must reason from incomplete evidence
- **Dense reward signal** — step-level rewards during inspection, not just at the end
- **Dual grading** — programmatic keyword scorer (85%) + LLM reasoning judge (15%)
- **Multi-component score** — diagnosis correctness, evidence coverage, efficiency, fix quality, and inspection order all contribute
- **WebSocket environment** — real-time interaction via FastAPI; supports concurrent sessions
- **Docker-ready** — one command to run the full environment server
- **Local agent** — smoke test the pipeline without any API key

---

## Grading — The Heart of the Environment ❤️

### Scoring Flow

```
submit_diagnosis received
        │
        ├─► Diagnosis Score     (was the label correct?)
        │       exact keyword match  → +0.40
        │       category/fuzzy match → +0.10 per keyword
        │       vague answer (<3 words) → −0.10
        │
        ├─► Evidence Score      (did the agent inspect the right sources?)
        │       +0.08 per required source inspected
        │       −0.10 per required source NOT inspected
        │       −0.02 per irrelevant source inspected
        │
        ├─► Evidence-Diagnosis Penalty  (had the clues, drew wrong conclusion?)
        │       all required sources + wrong diagnosis → −0.10
        │       some required sources + wrong diagnosis → −0.05
        │
        ├─► Efficiency Score    (did the agent act without waste?)
        │       minimum steps → +0.15
        │       extra steps   → −0.02 × (extra_steps^1.2)
        │       early submit  → −0.05 per missing step
        │
        ├─► Fix Score           (was the suggested fix actionable?)
        │       all fix keywords match → +0.15
        │       ≥60% match → +0.10
        │       ≥30% match → +0.05
        │       no fix provided → −0.05
        │
        └─► Ordering Bonus      (+0.05 if sources inspected in canonical order)
                                 logs → config → gradients
        
        Total = clamp(sum, 0.0, 1.0)
```


### Score Breakdown Table

| Score Type | Logic | Max Reward | Min Reward |
|---|---|---|---|
| Diagnosis | Keyword match on failure mode label | +0.70 | 0.00 |
| Evidence | Required sources inspected vs missing | +0.25 | −0.15 |
| Evidence-Diagnosis Penalty | Had evidence but wrong conclusion | 0.00 | −0.10 |
| Efficiency | Steps taken vs minimum needed | +0.15 | 0.00 |
| Fix | Keyword match on suggested fix | +0.15 | −0.05 |
| Ordering Bonus | Canonical inspection order | +0.05 | 0.00 |
| **Total** | Clamped to [0.0, 1.0] | **1.00** | **0.00** |

### Step-Level Rewards (during inspection)

| Action | Reward |
|---|---|
| First required source discovered | +0.10 |
| Second required source discovered | +0.07 |
| Third required source discovered | +0.05 |
| Irrelevant source inspected | −0.03 |
| Re-inspecting a source | −0.05 |

---

## LLM Judge

### What it does

The programmatic grader handles keyword matching — fast and deterministic. The **LLM Judge** runs after the episode ends and evaluates the quality of the agent's *reasoning*: did it actually cite evidence? Was the logic coherent? Did the fix make sense given the diagnosis?

### Judge Flow

```
submit_diagnosis
        │
        ├─► Programmatic grader  (keyword match → score)  85% weight
        │
        └─► LLM Judge           (reasoning quality → score)  15% weight
                │
                ├── diagnosis + suggested_fix + reasoning + scenario data
                │
                └── LLM evaluates:
                        - Did the agent cite specific numbers from the data?
                        - Is the reasoning internally consistent?
                        - Does the fix address the actual root cause?
                        │
                        └── Returns float 0.0–1.0

Final Score = 0.85 × keyword_score + 0.15 × judge_score
```


The judge uses the same model running inference (configurable via `MODEL_NAME`). It's deliberately lightweight — a single-turn evaluation with a structured prompt — so it doesn't dominate runtime.

---

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

---

## Baseline Performance (Qwen/Qwen2.5-72B-Instruct)

| Task | Avg Score | Pass Rate |
|---|---|---|
| Easy | 0.964 | 100% |
| Medium | 0.952 | 100% |
| Hard | 0.979 | 100% |

---

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python package manager
- [Docker](https://www.docker.com/) — for running the environment server
- A Hugging Face account with an API token ([get one here](https://huggingface.co/settings/tokens))

---

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

---

### 2. Install dependencies

```bash
git clone https://github.com/samrat-rm/WhyDidItFail
cd WhyDidItFail
uv sync
```

---

### 3. Configure environment variables

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token_here

# Optional overrides
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
SERVER_URL=http://localhost:8000
```

| Variable | Default | Required |
|---|---|---|
| `HF_TOKEN` | — | Yes |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | No |
| `SERVER_URL` | `http://localhost:8000` | No |

---

### 4. Start the environment server

The environment server runs in Docker. Build and start it:

```bash
# Build the image
docker build -t why_did_it_fail_env:latest .

# Run the server (exposes on port 8000)
docker run -p 8000:8000 why_did_it_fail_env:latest
```

The server is ready when you see `Uvicorn running on http://0.0.0.0:8000`.

---

### 5. Run inference

In a separate terminal:

```bash
uv run python inference.py

# Python 3 explicit
uv run python3 inference.py
```

Stdout will stream `[START]` / `[STEP]` / `[END]` lines per episode. Internal logs go to stderr.

---

### Local Agent — No API Key Required

To smoke test the full pipeline without calling an external LLM, you can run inference with a local model via [Ollama](https://ollama.com/).

**1. Install Ollama**

```bash
brew install ollama
```

**2. Pull a model**

```bash
# Recommended: a small instruction-tuned model
ollama pull qwen2.5:7b
```

**3. Start the Ollama server**

```bash
ollama serve
```

**4. Run inference against the local model**

```bash
USE_LOCAL=true uv run python inference.py
```

> The local agent follows a fixed inspection strategy and won't match frontier model scores, but it exercises the full pipeline — server, grader, judge, and stdout format — with no API calls or token costs.

---

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

---

## OpenEnv Spec Compliance

- Typed `Action`, `Observation` Pydantic models ✓
- `step(action)` → `(observation, reward, done, info)` ✓
- `reset()` → initial observation ✓
- `state()` → current state ✓
- `openenv.yaml` with 3 tasks and grader definitions ✓
- Passes `openenv validate` ✓


#### AI Usage Disclosure

This project was developed with the assistance of AI tools, including Claude and ChatGPT. These tools were used to support tasks such as code generation, documentation drafting, and problem-solving.

All AI-generated content has been carefully reviewed, tested, and validated before being included in this repository. I take full responsibility for the accuracy, functionality, and integrity of the code and documentation provided.

AI assistance was used as a productivity aid, but all final decisions, implementations, and reviews were performed by me to ensure quality and correctness.
