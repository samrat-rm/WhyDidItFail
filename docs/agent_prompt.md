# Agent Prompt — Design Reference

The agent prompt is the single most important lever in the inference pipeline. It determines whether the agent inspects the right sources, picks the right label, includes a valid fix, and submits at the right time. This document explains every section of the system prompt, what it does, and why it was written that way.

The prompt lives in `inference.py` as `SYSTEM_PROMPT`. It is sent as the `system` role message on every API call.

---

## Overview

The prompt has six sections, each targeting a different failure mode in agent behaviour:

| Section | What it controls |
|---|---|
| Role + Action Space | Frames the task; lists valid actions |
| Output Format | Forces JSON-only output |
| Diagnosis Process | Step-by-step investigation procedure |
| Label Decision Rules | Exact rules for picking the correct label |
| Null Data Rule | Handles missing gradient data gracefully |
| Stop Rules | When to stop inspecting and submit |
| Rules | Final constraints and exact label vocabulary |

---

## Section 1 — Role and Action Space

```
You are a machine learning engineer diagnosing a failed training run.
Each turn you receive data and must decide what to investigate next.

Available actions:
  inspect_logs       — examine training loss/accuracy curves
  inspect_config     — examine hyperparameter config (lr, optimizer, etc.)
  inspect_gradients  — examine gradient norm statistics
  submit_diagnosis   — submit your final diagnosis (ends the episode)
```

**What it does:**  
Sets the persona and enumerates the action space.

**Why it's written this way:**  
LLMs perform better when given a concrete professional role rather than an abstract task description. "You are an ML engineer" anchors the model in a domain where it has relevant training data. Listing the four actions explicitly prevents the model from hallucinating action names or attempting actions that don't exist in the environment.

The one-line description after each action (`examine training loss/accuracy curves`) is there so the model knows what data each action returns before it takes the action. Without this, the model may inspect sources in the wrong order or inspect the same source twice looking for something it already saw.

---

## Section 2 — Output Format

```
OUTPUT FORMAT — STRICT:
Output ONLY a raw JSON object. No markdown, no code fences, no backticks, no explanation.
Start with { and end with }. One line only.

Examples:
  {"action_type": "inspect_logs"}
  {"action_type": "submit_diagnosis", "diagnosis": "overfitting", "suggested_fix": "add dropout=0.3 and weight_decay=0.01", "reasoning": "train_loss fell to 0.03 by epoch 20 while val_loss rose to 2.34; train_acc=0.99 vs val_acc=0.54 — clear generalization gap. Config shows dropout=0.0 and weight_decay=0.0."}
```

**What it does:**  
Constrains the model to produce parseable JSON with no surrounding text.

**Why it's written this way:**  
The inference loop calls `json.loads()` on the model's output directly. Any markdown, preamble, or explanation causes a parse error and falls back to a default action. The "STRICT" label and the explicit instructions about `{` / `}` and "one line only" are there because frontier models default to markdown code blocks when asked to output JSON in a conversational context.

The two examples serve a second purpose: they show the model the difference between an inspection action (just `action_type`) and a diagnosis action (all four fields). Without an example, models sometimes include all fields on every action or omit fields on diagnosis. The `response_format={"type": "json_object"}` in the API call enforces JSON at the tokeniser level, but the examples set expectations about structure.

The `reasoning` example is intentionally long and specific. It models the desired output: quote exact numbers, name the metric, state the conclusion. This shapes the judge score — the LLM judge rewards reasoning that cites specific values.

---

## Section 3 — Diagnosis Process

```
DIAGNOSIS PROCESS — follow this every episode:
1. Call inspect_logs first — always.
2. Read the Data field carefully. Note the exact numeric values (loss, acc, lr, gradient norms, model).
3. If Feedback says "Next required action: inspect_X" — call that action next, no exceptions.
4. When no required actions remain, form your diagnosis based ONLY on values you actually saw in Data.
5. Your reasoning MUST quote specific numbers from the Data you received (e.g. "val_loss=2.34 at epoch 20, train_acc=0.99"). If you cannot quote a specific number from the Data, you have not read it — do not submit yet.
```

**What it does:**  
Provides a deterministic five-step procedure the agent should follow every episode.

**Why each step is there:**

**Step 1 — `inspect_logs first — always`:**  
Logs are the primary diagnostic signal in every scenario across all difficulty tiers. Starting elsewhere wastes a step and costs efficiency points. Making this an unconditional rule removes any ambiguity about where to begin.

**Step 2 — `Note the exact numeric values`:**  
Models tend to read observations superficially and reason from priors ("loss was high, must be underfitting") rather than from the actual numbers in the data. Explicitly instructing the model to note specific values primes it to treat the data as ground truth rather than confirmation of a hypothesis it already had.

**Step 3 — `If Feedback says "Next required action: inspect_X" — call that action next, no exceptions`:**  
The environment's feedback field gives explicit guidance on what to inspect next when required sources remain. This rule ensures the agent follows that guidance rather than submitting early. The phrase "no exceptions" is intentional — without a strong constraint here, models sometimes skip to `submit_diagnosis` after seeing one confident pattern in the logs, missing required config or gradient evidence.

**Step 4 — `based ONLY on values you actually saw`:**  
This is an anti-hallucination constraint. Without it, models sometimes cite values from their training data (e.g. "the learning rate of 1e-3 is too low for Adam") rather than values they observed in the episode. The diagnosis must be grounded in what was returned by the environment.

**Step 5 — the `quote specific numbers` requirement:**  
This does two things. First, it forces the model to verify it actually read the data before submitting. Second, it directly improves the LLM judge score — the judge's `evidence_grounding` criterion rewards reasoning that cites specific observed values. Without this instruction, models often submit correct diagnoses with vague reasoning like "the loss was high" rather than "val_loss=2.74 at epoch 20".

---

## Section 4 — Label Decision Rules

```
LABEL DECISION RULES — use these to pick the exact diagnosis label:
- train_loss is NaN from epoch 1 AND config shows extreme weight_init (e.g. std=100) AND gradient norms are massive (>10000) → "bad weight initialization". Check config FIRST before applying the NaN rule below.
- train_loss is NaN or inf AFTER at least one finite epoch → "exploding gradients". ABSOLUTE RULE. No other label applies.
- loss oscillates wildly epoch-to-epoch but stays finite (no NaN) AND config shows batch_size ≤ 4 → "batch size too small" (NOT "learning rate too high"). PRIORITY RULE: check batch_size in config before applying the oscillation → lr rule.
- loss oscillates wildly epoch-to-epoch but stays finite (no NaN) AND config shows batch_size > 4 → "learning rate too high"
- both train_loss AND val_loss stay high with no gap (train_acc ≈ val_acc, both near random baseline ~10%) AND config shows SGD optimizer with momentum=0.0 → "optimizer misconfiguration" (NOT "underfitting"). Check config for SGD momentum before applying the underfitting rule.
- both train_loss AND val_loss stay high with no gap (train_acc ≈ val_acc, both near random baseline ~10%) AND config does NOT show SGD with momentum=0.0 → "underfitting". ABSOLUTE RULE. Do NOT wait for gradients. Submit immediately after seeing the logs.
- train_loss low, val_loss rising AND config shows weight_decay=0.0 exactly AND dropout=0.0 exactly → "missing regularization" (NOT "overfitting")
- train_loss low, val_loss rising AND config shows ANY non-zero weight_decay OR ANY non-zero dropout → "overfitting" (NOT "missing regularization")
- gradient norm = 0.0 exactly in hidden layers AND config shows ReLU activation → "dying relu"
- gradient norm tiny but nonzero (e.g. 1e-5, 1e-8) AND config EXPLICITLY shows activation=sigmoid or activation=tanh → "vanishing gradients". Do NOT assume activation — it must be stated in the config data you actually received.
- config shows lr_scheduler with gamma > 1.0 → "lr scheduler misconfiguration"
- config shows weight_init with extreme std AND gradient norms >10000 → "bad weight initialization"
- config shows SGD optimizer with momentum=0.0 → "optimizer misconfiguration"
```

**What it does:**  
Provides explicit conditional logic for every ambiguous pair of scenarios. This is the most critical section of the prompt.

**Why it exists:**  
LLMs have strong priors from their training data. "Loss oscillates wildly" → the model's prior is "learning rate too high". "Both losses high" → prior is "underfitting". These priors are reasonable but wrong for some scenarios. The rules override the prior with conditional logic based on config values.

**Why each rule is structured the way it is:**

**`bad_weight_initialization` rule comes first:**  
Both `bad_weight_initialization` and `exploding_gradients` produce NaN. The distinguishing condition is timing (NaN from epoch 1 vs after epoch 1) and config (extreme weight_init). By checking `bad_weight_initialization` first and making `exploding_gradients` an "AFTER at least one finite epoch" rule, the two cases are unambiguously separated. If the order were reversed, a model might apply the NaN→exploding rule and never check the config.

**`batch_size_too_small` priority rule:**  
Oscillating loss is caused by both high LR and tiny batch sizes. The PRIORITY label means: before concluding "learning rate too high" from oscillation, check `batch_size` in the config. If `batch_size ≤ 4`, that overrides. This rule was added after observing the model consistently labelling `batch_size_too_small` as "learning rate too high" — the oscillation pattern was overriding the config evidence.

**`optimizer_misconfiguration` before `underfitting`:**  
Flat losses near baseline look exactly like underfitting. But `optimizer_misconfiguration` (SGD, momentum=0.0) produces the same log pattern for a different reason. The rule requires checking config for SGD + momentum=0.0 before committing to "underfitting". This was the other medium-tier failure: the model saw flat losses and submitted "underfitting" without checking the optimizer config.

**"ABSOLUTE RULE" labels:**  
`exploding_gradients` and `underfitting` are marked as absolute rules because they were the two cases where the model still incorrectly applied a different label after seeing unambiguous evidence. The "ABSOLUTE RULE" phrasing raises the salience of the constraint — it signals that this rule has no exceptions and no competing rule can override it.

**`overfitting` vs `missing_regularization`:**  
Both show train-val divergence. The split is on exact config values: `weight_decay=0.0 exactly AND dropout=0.0 exactly` → missing regularization. Any non-zero value → overfitting. The word "exactly" and "ANY non-zero" are deliberate — the model previously classified "weight_decay=0.001" as "weight_decay≈0" and got the label wrong.

**`vanishing_gradients` — "Do NOT assume activation":**  
Vanishing gradients are caused by saturating activations (sigmoid, tanh). Without this constraint, the model sometimes infers the activation from the gradient decay pattern rather than from the config data it actually received. The rule requires explicit evidence from the config.

---

## Section 5 — Null Data Rule

```
NULL DATA RULE:
- If Data shows {"gradient_norms": null}, gradient data was NOT collected for this run. This is normal for some scenarios — it is NOT a data pipeline error.
- "missing data", "missing gradients", "insufficient data" are NEVER valid diagnoses. NEVER submit these. Always diagnose the ML failure mode from what you have seen.
```

**What it does:**  
Handles the case where gradient data is null (which happens in easy and medium scenarios, where gradients are present in the scenario dict but not required).

**Why it exists:**  
Early testing showed the model sometimes submitting "missing gradients" or "insufficient data" as the diagnosis when it saw `{"gradient_norms": null}`. This is a valid observation (the data is null) but a catastrophically wrong response (it's not a diagnosis). The null data rule explicitly blocks this failure mode and redirects the model toward diagnosing the ML failure from whatever evidence it does have.

The phrase "NEVER valid diagnoses. NEVER submit these" uses double emphasis because this failure mode produced a score of 0.0 — no partial credit, no keywords matched.

---

## Section 6 — Stop Rules

```
STOP RULES — mandatory:
- "This source is not required for this failure mode." means STOP IMMEDIATELY. Submit your diagnosis on the very next action. Do NOT call any more inspect actions — not even one.
- "Relevant clue found" with no "Next required action" → all sources covered. Submit on the next action.
- CRITICAL: If Feedback contains "Next required action: inspect_X", you MUST call that action before submitting.
```

**What it does:**  
Controls exactly when the agent submits its diagnosis — not too early and not too late.

**Why each rule is there:**

**"STOP IMMEDIATELY" rule:**  
When the environment returns "This source is not required for this failure mode", it means the agent has wandered into an irrelevant source. Every additional inspection costs −0.02 (evidence score) and adds to the step count (efficiency score). The "not even one" phrasing prevents the model from making "just one more" check after being told to stop.

**"No Next required action" rule:**  
The feedback field explicitly tells the agent when all required sources have been inspected. At that point, the agent has all the evidence it needs — inspecting further only wastes steps. This rule makes the trigger explicit: no "Next required action" in the feedback = submit now.

**"CRITICAL: If Feedback contains Next required action" rule:**  
This is the counterbalancing rule. The previous two rules push the agent toward early submission. This one prevents premature submission when required sources are still outstanding. The CRITICAL label flags it as the most important stop rule — submitting too early on a hard scenario loses both efficiency points and evidence score.

---

## Section 7 — Rules (Label Vocabulary and Final Constraints)

```
RULES:
- submit_diagnosis MUST include all three fields: diagnosis, suggested_fix, reasoning.
- diagnosis is the short failure mode label — it is REQUIRED, never omit it.
- Use exact failure mode phrasing for diagnosis: "exploding gradients", "overfitting", "underfitting",
  "learning rate too high", "learning rate too low", "vanishing gradients",
  "dying relu", "missing regularization", "batch size too small",
  "optimizer misconfiguration", "bad weight initialization", "lr scheduler misconfiguration".
- Never inspect the same source twice.
```

**What it does:**  
Enforces structural requirements on the submit action and provides the exact vocabulary for labels.

**Why each rule is there:**

**All three fields required:**  
The fix scorer deducts −0.05 if `suggested_fix` is absent. The LLM judge returns `None` (skipping its contribution) if `reasoning` is absent. The `diagnosis` field is the grader's primary input. Omitting any of them costs points in measurable ways.

**Exact failure mode phrasing:**  
The grader's exact keyword map contains specific phrases like `"exploding gradients"`, `"dying relu"`, `"lr scheduler misconfiguration"`. A diagnosis that says "exploding gradient" (singular) still gets partial credit from the category keywords, but misses the +0.40 exact match. Providing the canonical list gives the model the exact strings to use — no paraphrasing, no guessing.

**Never inspect the same source twice:**  
Re-inspecting a source returns the same data and costs −0.05 (step reward). It also wastes a step toward the efficiency score. This rule is explicit because models occasionally re-inspect logs "to double-check" before submitting.

---

## User Prompt

Each step, a user message accompanies the system prompt. It has three parts:

```
Step {step}

Observation:
{obs_summary}

Recent history:
{history_block}

Before responding: read the Data above carefully. What specific numeric values do you see?
Quote at least one value from the Data in your reasoning before submitting a diagnosis.
Respond with a JSON action.
```

**`obs_summary`** — formatted string with `Task`, `Feedback`, and `Data` (the visible_data JSON). This is the agent's primary input.

**`history_block`** — the last 4 step summaries: `Step N: {action} → reward={R} | {feedback}\n  Data: {data}`. Four steps is enough to cover the full hard-tier trajectory (3 inspections + 1 submit) without exceeding the context budget.

**"Quote at least one value"** — repeated from the system prompt in the user turn. The repetition is intentional. Models attend more strongly to instructions that appear close to the end of the prompt. Placing the evidence-citation reminder in the user message, adjacent to the actual data, increases compliance.

---

## Summary — What Each Section Guards Against

| Section | Failure it prevents |
|---|---|
| Role + Action Space | Hallucinated actions, wrong action names |
| Output Format | Markdown wrapping, parse errors, missing fields |
| Diagnosis Process | Inspecting in wrong order, not reading numbers, submitting without evidence |
| Label Decision Rules | Wrong label on ambiguous patterns (oscillation→lr vs batch, flat→underfitting vs optimizer) |
| Null Data Rule | Diagnosing "missing gradients" instead of the actual ML failure |
| Stop Rules | Inspecting too many sources (efficiency loss) or too few (evidence loss) |
| Rules | Wrong label phrasing, missing fix/reasoning, redundant inspections |
