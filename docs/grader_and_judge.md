# Grader & LLM Judge — Internal Design

This document explains how WhyDidItFail scores an agent's performance after each episode.
Scoring has two layers: a **programmatic grader** that runs keyword matching, and an **LLM judge** that evaluates reasoning quality. The two scores are combined into a single final number.

---

## Overview

```
Agent submits diagnosis
        │
        ├─► Programmatic Grader  (server/graders.py)
        │         five sub-scores → keyword_score  [0.0–1.0]
        │
        └─► LLM Judge            (server/llm_judge.py)
                  three criteria → judge_score     [0.0–1.0]

Final Score = 0.85 × keyword_score + 0.15 × judge_score
```

The grader is fast and deterministic — it runs synchronously inside the environment's `step()` call. The judge is a separate LLM call that runs **after** the episode ends (the WebSocket is already closed by then), so it never adds latency to the agent's action loop.

---

## Part 1 — Programmatic Grader

**File:** `server/graders.py`  
**Entry point:** `grade(diagnosis, suggested_fix, scenario, steps_taken, inspection_order)`

The grader produces five sub-scores and sums them. The result is clamped to `[0.0, 1.0]`.

```
Total = diagnosis_score + evidence_diagnosis_penalty
      + evidence_score + efficiency_score + fix_score + ordering_bonus
```

### 1.1 Diagnosis Score — up to +0.70

This is the biggest sub-score. It checks whether the agent named the correct failure mode.

Every scenario has a `correct_diagnosis` field (e.g. `"exploding_gradients"`). There are two keyword maps in the grader:

**Exact keywords** — strong signal, worth +0.40 each:
```
"exploding_gradients" → ["exploding gradients", "exploding"]
"overfitting"         → ["overfitting", "overfit"]
"dying_relu"          → ["dying relu", "dead relu"]
...
```

**Category keywords** — weaker signal, worth +0.10 each:
```
"exploding_gradients" → ["nan", "gradient", "overflow", "diverge"]
"overfitting"         → ["generalization", "val loss", "memoriz"]
...
```

The agent's diagnosis string is lowercased and scanned for both sets. Matches accumulate. The result is capped at 0.70.

**Vague answer penalty:** If no exact keyword matched and the diagnosis is fewer than 3 words, −0.10 is applied. This discourages one-word guesses.

**Practical meaning:** A correct exact label ("exploding gradients" or "exploding") earns +0.40 right away. Hitting a few category keywords on top of that can bring it up toward the 0.70 cap. A wrong diagnosis with some related vocabulary still earns partial credit, but cannot exceed 0.70.

---

### 1.2 Evidence-Diagnosis Penalty — 0.00 to −0.10

This penalty fires when the agent had access to the right data but still got the diagnosis wrong. It distinguishes between an agent that guessed blind versus one that inspected evidence and reasoned poorly.

| Situation | Penalty |
|---|---|
| All required sources inspected, diagnosis wrong | −0.10 |
| Some required sources inspected, diagnosis wrong | −0.05 |
| No required sources inspected, diagnosis wrong | 0.00 |
| Diagnosis correct (any case) | 0.00 |

The evidence score (next section) already penalises skipping required sources. This penalty stacks on top when the agent collected the evidence but drew the wrong conclusion — a clear reasoning failure.

---

### 1.3 Evidence Score — up to +0.25 (floored at −0.15)

This sub-score measures whether the agent inspected the right data sources before submitting.

Each scenario defines which sources are required. Easy scenarios require only `logs`. Medium requires `logs + config`. Hard requires `logs + config + gradients`.

```
+0.08  per required source inspected
−0.10  per required source NOT inspected
−0.02  per irrelevant source inspected
```

The raw sum is clamped to `[−0.15, +0.25]`.

**Example — hard scenario, agent inspected logs + config but skipped gradients:**
- logs required and inspected: +0.08
- config required and inspected: +0.08
- gradients required but missing: −0.10
- Total: +0.06

**Example — easy scenario, agent inspected logs + gradients (gradients not required):**
- logs required and inspected: +0.08
- gradients not required: −0.02
- Total: +0.06

The −0.02 for irrelevant sources is intentionally mild — some exploration is acceptable. Missing a required source is much more costly.

---

### 1.4 Efficiency Score — up to +0.15

This sub-score rewards acting without waste. The minimum number of steps for any episode is: `len(required_sources) + 1` (inspect all required sources, then submit).

```
If steps_taken == min_steps:  score = 0.15  (full reward)
If steps_taken > min_steps:   score = max(0.0, 0.15 − 0.02 × extra_steps^1.2)
If steps_taken < min_steps:   score = max(0.0, 0.15 − 0.05 × missing_steps)
```

The exponent `1.2` makes the penalty accelerate as the agent wastes more steps — each additional wasted step costs slightly more than the previous one.

Early submission (fewer steps than the minimum) is also penalised because the agent almost certainly skipped a required source — but that case is already caught harder by the evidence score.

**If `steps_taken > max_steps`** (which is `len(required_sources) × 3 + 2`), the grader immediately returns `0.0` and skips all sub-scores. This is the hard ceiling.

---

### 1.5 Fix Score — up to +0.15 (or −0.05)

This sub-score checks whether the agent's suggested fix is actionable and correct.

Each scenario has a `correct_fix` string (e.g. `"enable gradient clipping (clip_grad_norm=1.0)"`). The grader tokenises that string, strips stop words (`to`, `a`, `the`, `and`, `or`, `use`, `set`, `by`), and keeps content words longer than 2 characters.

It then counts how many of those content words appear anywhere in the agent's `suggested_fix` string.

| Match ratio | Score |
|---|---|
| 100% of keywords matched | +0.15 |
| ≥ 60% matched | +0.10 |
| ≥ 30% matched | +0.05 |
| < 30% matched | 0.00 |
| No fix provided | −0.05 |

Omitting the fix always costs points. Even a partially correct fix earns more than silence.

---

### 1.6 Ordering Bonus — +0.05

A small bonus for inspecting required sources in the canonical order: `logs → config → gradients`.

The grader extracts the subsequence of inspected sources that are required (ignoring any irrelevant ones inspected in between), and checks whether that subsequence matches the canonical order for the scenario.

For an easy scenario (only `logs` required) the bonus is trivially earned. For hard scenarios (all three required) the agent must visit them in order.

This rewards structured investigation — the same order a human engineer would follow.

---

### 1.7 Maximum Achievable Scores

| Sub-score | Max | Min |
|---|---|---|
| Diagnosis | +0.70 | 0.00 |
| Evidence-Diagnosis Penalty | 0.00 | −0.10 |
| Evidence | +0.25 | −0.15 |
| Efficiency | +0.15 | 0.00 |
| Fix | +0.15 | −0.05 |
| Ordering Bonus | +0.05 | 0.00 |
| **Total (clamped)** | **1.00** | **0.00** |

The theoretical maximum without fix is `0.70 + 0.25 + 0.15 + 0.05 = 1.15`, which is clamped to `1.00`. The fix and ordering bonus are therefore "free" once the other scores are high — they can push a near-perfect run to a clean `1.00`.

---

## Part 2 — Step-Level Rewards

The grader described above runs only at `submit_diagnosis`. But the environment also emits a small reward on every inspection step, giving the agent an in-episode signal.

These step rewards come from `_inspect_reward()` in the environment, not from `graders.py`.

| Situation | Step Reward |
|---|---|
| First required source discovered | +0.10 |
| Second required source discovered | +0.07 |
| Third required source discovered | +0.05 |
| Irrelevant source inspected | −0.03 |
| Re-inspecting a source already seen | −0.05 |

The decaying reward schedule (+0.10 → +0.07 → +0.05) reflects that each additional required source is slightly less surprising than the first. It also gives a larger signal for discovering the first clue, which is usually the most diagnostic piece of evidence.

These step rewards are reported in `[STEP]` lines during inference but they are **not** included in the final episode score — `grade()` computes the terminal score independently.

---

## Part 3 — LLM Judge

**File:** `server/llm_judge.py`  
**Entry point:** `judge(client, model, diagnosis, reasoning, suggested_fix, scenario, inspection_order)`

### 3.1 What It Evaluates

The programmatic grader is blind to *how* the agent arrived at its answer. An agent could get lucky with the right keyword and still have incoherent reasoning. The LLM judge evaluates the *quality of the agent's reasoning* across three criteria:

| Criterion | Question it asks | Max |
|---|---|---|
| `evidence_grounding` | Does the reasoning cite specific values from the data the agent actually saw? | 5 |
| `causal_chain` | Does it logically connect that evidence to the diagnosed failure mode? | 5 |
| `fix_rationale` | Is the fix directly justified by the evidence and diagnosis? | 5 |

Raw score range: 0–15. Normalised to 0.0–1.0 by dividing by 15.

---

### 3.2 When It Runs

The judge runs **after** the episode ends — specifically after `submit_diagnosis` returns and the WebSocket connection is closed. This is intentional: the WebSocket session is single-use, so the judge call can't interfere with the agent's action loop.

```
Agent calls submit_diagnosis
        │
        └─► environment returns final_reward, done=True
                │
                └─► WebSocket closes
                        │
                        └─► inference.py calls judge()
                                │
                                └─► one LLM call → score
                                        │
                                        └─► Final Score = 0.85 × keyword + 0.15 × judge
```

---

### 3.3 What the Judge Sees

The judge prompt is built from:
1. The agent's `diagnosis`, `suggested_fix`, and `reasoning` strings
2. The **data the agent actually inspected** — reconstructed from `inspection_order` and the scenario's source data

The judge is given only what the agent had access to, not the full scenario. This prevents the judge from penalising an agent for not citing data it never saw.

```python
seen = {}
if "logs" in inspection_order:
    seen["training_logs"] = scenario["logs"]
if "config" in inspection_order:
    seen["config"] = scenario["config"]
if "gradients" in inspection_order:
    seen["gradient_norms"] = scenario["gradient_norms"]
```

---

### 3.4 The Judge Prompt

The judge receives a single-turn user message (no system prompt). It is asked to return a JSON object with integer scores:

```
{"evidence_grounding": <int>, "causal_chain": <int>, "fix_rationale": <int>}
```

Temperature is set to `0.0` for determinism. `max_tokens=64` is enough for the JSON response and prevents runaway output.

---

### 3.5 Fallback Behaviour

If the agent omits the `reasoning` field, the judge returns `None` immediately (no API call made). The caller treats `None` as "judge unavailable" and uses keyword score at full weight:

```
judge_score is None  →  Final Score = 1.0 × keyword_score
judge_score is float →  Final Score = 0.85 × keyword_score + 0.15 × judge_score
```

If the LLM call itself fails (network error, parse error, etc.), the judge also returns `None` and logs the exception to stderr. The episode score is never blocked waiting on the judge.

---

### 3.6 Weight Rationale

The 85/15 split keeps the judge in a supporting role. The programmatic grader is:
- **Fast** — no extra API call during the episode
- **Deterministic** — same input always produces same score
- **Directly tied to the correct answer** — keyword matching is unambiguous

The judge adds 15% for reasoning quality. This is enough to meaningfully separate agents that cite evidence from those that guess correctly without explaining why, but not enough to override a solid keyword score with a harsh reasoning judgment.

---

## Part 4 — Combined Final Score

```
Final Score = clamp(0.85 × keyword_score + 0.15 × judge_score, 0.0, 1.0)
```

Both the keyword score and the judge score are already in `[0.0, 1.0]`, so the final score is also in that range without needing clamping in practice. The clamp is a safeguard.

**Example — perfect run:**
- keyword_score = 1.00 (correct label, all sources, minimum steps, full fix, correct order)
- judge_score = 1.00 (cites specific numbers, clean causal chain, fix matches evidence)
- Final = 0.85 × 1.00 + 0.15 × 1.00 = **1.00**

**Example — correct label, poor reasoning:**
- keyword_score = 0.90 (correct label, good evidence, slightly wasteful)
- judge_score = 0.40 (vague reasoning, no specific numbers cited)
- Final = 0.85 × 0.90 + 0.15 × 0.40 = 0.765 + 0.060 = **0.825**

**Example — reasoning provided but LLM call fails:**
- keyword_score = 0.90
- judge_score = None
- Final = 1.00 × 0.90 = **0.90**

---

## Summary

| Component | File | Weight | What it measures |
|---|---|---|---|
| Diagnosis Score | graders.py | Up to 0.70 of keyword_score | Correct failure mode label |
| Evidence-Diagnosis Penalty | graders.py | Up to −0.10 of keyword_score | Reasoning failure despite having evidence |
| Evidence Score | graders.py | Up to 0.25 of keyword_score | Right data sources inspected |
| Efficiency Score | graders.py | Up to 0.15 of keyword_score | Steps taken vs minimum |
| Fix Score | graders.py | Up to 0.15 of keyword_score | Actionable and correct fix |
| Ordering Bonus | graders.py | +0.05 of keyword_score | Canonical inspection order |
| LLM Judge | llm_judge.py | 15% of final score | Reasoning quality (evidence citation, causal logic, fix rationale) |