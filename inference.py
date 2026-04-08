"""
Inference Script — WhyDidItFail

Environment variables:
    HF_TOKEN     required
    API_BASE_URL default: https://router.huggingface.co/v1
    MODEL_NAME   default: Qwen/Qwen2.5-72B-Instruct
    SERVER_URL   default: http://localhost:8000

Stdout format:
    [START]   task=<name> env=whydiditfail model=<model>          (per episode)
    [STEP]    step=<n> action=<action_type> reward=<float> done=<bool> error=<null|msg>
    [END]     success=<bool> steps=<n> reward=<float>             (per episode)
    [END]     score=<float>                                        (final overall)

All reward and score values are strictly in (0.10, 0.90).
"""

import asyncio
import json
import os
import textwrap
from typing import List

from websockets.exceptions import ConnectionClosedError

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from client import WhyDidItFailEnv
from server.llm_judge import judge as llm_judge
from models import WhyDidItFailAction
from server.scenarios import SCENARIOS

IMAGE_NAME       = os.getenv("IMAGE_NAME", "")
SERVER_URL       = os.getenv("SERVER_URL", "http://localhost:8000")
HF_TOKEN         = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
API_KEY          = HF_TOKEN or os.getenv("API_KEY")
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
USE_LOCAL        = os.getenv("USE_LOCAL", "false").lower() == "true"
MAX_STEPS        = 8
TEMPERATURE      = 0.3
MAX_TOKENS       = 256
SUCCESS_THRESHOLD = 0.5

EASY_SCENARIOS   = [k for k, v in SCENARIOS.items() if v["difficulty"] == "easy"][:1]
MEDIUM_SCENARIOS = [k for k, v in SCENARIOS.items() if v["difficulty"] == "medium"][:1]
HARD_SCENARIOS   = [k for k, v in SCENARIOS.items() if v["difficulty"] == "hard"][:1]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a machine learning engineer diagnosing a failed training run.
    Each turn you receive data and must decide what to investigate next.

    Available actions:
      inspect_logs       — examine training loss/accuracy curves
      inspect_config     — examine hyperparameter config (lr, optimizer, etc.)
      inspect_gradients  — examine gradient norm statistics
      submit_diagnosis   — submit your final diagnosis (ends the episode)

    OUTPUT FORMAT — STRICT:
    Output ONLY a raw JSON object. No markdown, no code fences, no backticks, no explanation.
    Start with { and end with }. One line only.

    Examples:
      {"action_type": "inspect_logs"}
      {"action_type": "submit_diagnosis", "diagnosis": "overfitting", "suggested_fix": "add dropout=0.3 and weight_decay=0.01", "reasoning": "train_loss fell to 0.03 by epoch 20 while val_loss rose to 2.34; train_acc=0.99 vs val_acc=0.54 — clear generalization gap. Config shows dropout=0.0 and weight_decay=0.0."}

    DIAGNOSIS PROCESS — follow this every episode:
    1. Call inspect_logs first — always.
    2. Read the Data field carefully. Note the exact numeric values (loss, acc, lr, gradient norms, model).
    3. If Feedback says "Next required action: inspect_X" — call that action next, no exceptions.
    4. When no required actions remain, form your diagnosis based ONLY on values you actually saw in Data.
    5. Your reasoning MUST quote specific numbers from the Data you received (e.g. "val_loss=2.34 at epoch 20, train_acc=0.99"). If you cannot quote a specific number from the Data, you have not read it — do not submit yet.

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

    NULL DATA RULE:
    - If Data shows {"gradient_norms": null}, gradient data was NOT collected for this run. This is normal for some scenarios — it is NOT a data pipeline error.
    - "missing data", "missing gradients", "insufficient data" are NEVER valid diagnoses. NEVER submit these. Always diagnose the ML failure mode from what you have seen.

    STOP RULES — mandatory:
    - "This source is not required for this failure mode." means STOP IMMEDIATELY. Submit your diagnosis on the very next action. Do NOT call any more inspect actions — not even one.
    - "Relevant clue found" with no "Next required action" → all sources covered. Submit on the next action.
    - CRITICAL: If Feedback contains "Next required action: inspect_X", you MUST call that action before submitting.

    RULES:
    - submit_diagnosis MUST include all three fields: diagnosis, suggested_fix, reasoning.
    - diagnosis is the short failure mode label — it is REQUIRED, never omit it.
    - Use exact failure mode phrasing for diagnosis: "exploding gradients", "overfitting", "underfitting",
      "learning rate too high", "learning rate too low", "vanishing gradients",
      "dying relu", "missing regularization", "batch size too small",
      "optimizer misconfiguration", "bad weight initialization", "lr scheduler misconfiguration".
    - Never inspect the same source twice.
""").strip()



def _user_prompt(step: int, obs_summary: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step {step}

        Observation:
        {obs_summary}

        Recent history:
        {history_block}

        Before responding: read the Data above carefully. What specific numeric values do you see?
        Quote at least one value from the Data in your reasoning before submitting a diagnosis.
        Respond with a JSON action.
    """).strip()


def _summarize(obs) -> str:
    lines = [
        f"Task: {obs.task_description}",
        f"Feedback: {obs.feedback}",
    ]
    if obs.visible_data:
        lines.append(f"Data:\n{json.dumps(obs.visible_data, indent=2)}")
    return "\n".join(lines)


def _get_action(client: OpenAI, step: int, obs_summary: str, history: List[str]) -> WhyDidItFailAction:
    if USE_LOCAL:
        from local_agent import get_action as _local_get_action
        prompt = f"{SYSTEM_PROMPT}\n\n{_user_prompt(step, obs_summary, history)}"
        return _local_get_action(step, prompt)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _user_prompt(step, obs_summary, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        filtered = {k: v for k, v in data.items() if k in WhyDidItFailAction.model_fields}
        return WhyDidItFailAction(**filtered)
    except Exception as exc:
        # print(f"  [DEBUG] parse error: {exc}", file=sys.stderr, flush=True)
        if step <= 2:
            return WhyDidItFailAction(action_type="inspect_logs", diagnosis=None, suggested_fix=None,reasoning=None)
        return WhyDidItFailAction(action_type="submit_diagnosis", diagnosis="unknown", suggested_fix=None,reasoning=None)

async def _make_env() -> WhyDidItFailEnv:
    return (
        await WhyDidItFailEnv.from_docker_image(IMAGE_NAME)
        if IMAGE_NAME
        else WhyDidItFailEnv(base_url=SERVER_URL)
    )


async def run_episode(
    env: WhyDidItFailEnv,
    client: OpenAI,
    scenario_key: str,
    task_name: str,
    effective_model: str,
) -> tuple[dict, WhyDidItFailEnv]:
    """Run one full episode for a specific scenario. Returns (result dict, env).
    env may be a fresh reconnected instance if the WebSocket dropped between episodes."""
    try:
        result = await env.reset(scenario_key=scenario_key)
    except ConnectionClosedError:
        # print(f"  [WARN]    scenario={scenario_key} reconnecting WebSocket...", file=sys.stderr, flush=True)
        env = await _make_env()
        result = await env.reset(scenario_key=scenario_key)

    print(f"[START] task={task_name} env=whydiditfail model={effective_model}", flush=True)

    obs      = result.observation
    history: List[str] = []
    rewards: List[float] = []
    inspection_order: List[str] = []
    submit_action: WhyDidItFailAction | None = None
    score    = 0.10
    success  = False

    try:
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = _get_action(client, step, _summarize(obs), history)
            try:
                result = await env.step(action)
            except ConnectionClosedError as e:
                print(f"[STEP] step={step} action={action.action_type} reward=0.10 done=true error={e}", flush=True)
                break
            obs    = result.observation
            reward = round(max(0.10, min(0.90, result.reward or 0.10)), 2)
            done   = result.done
            if action.action_type in ("inspect_logs", "inspect_config", "inspect_gradients"):
                source = action.action_type.replace("inspect_", "")
                if source not in inspection_order:
                    inspection_order.append(source)

            if action.action_type == "submit_diagnosis":
                submit_action = action  # judge runs after loop — WebSocket is closed by then

            rewards.append(reward)
            data_seen = json.dumps(obs.visible_data) if obs.visible_data else "{}"
            history.append(f"Step {step}: {action.action_type} → reward={reward:.2f} | {obs.feedback}\n  Data: {data_seen}")
            print(f"[STEP] step={step} action={action.action_type} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

            if done:
                break

        # WebSocket is closed — safe to call the judge now
        keyword_score = max(0.10, min(0.90, rewards[-1])) if rewards else 0.10
        judge_score: float | None = None
        if submit_action is not None:
            judge_score = llm_judge(
                client=client,
                model=MODEL_NAME,
                diagnosis=submit_action.diagnosis or "",
                reasoning=submit_action.reasoning,
                suggested_fix=submit_action.suggested_fix,
                scenario=SCENARIOS[scenario_key],
                inspection_order=inspection_order,
            )
        if judge_score is None:
            score = round(max(0.10, min(0.90, keyword_score)), 2)
            # print(f"  [JUDGE]   scenario={scenario_key} keyword={keyword_score:.2f} reasoning=n/a total={score:.2f}", file=sys.stderr, flush=True)
        else:
            score = round(max(0.10, min(0.90, 0.85 * keyword_score + 0.15 * judge_score)), 2)
            # print(f"  [JUDGE]   scenario={scenario_key} keyword={keyword_score:.3f} reasoning={judge_score:.3f} total={score:.3f}", file=sys.stderr, flush=True)

        success = score >= SUCCESS_THRESHOLD

    finally:
        steps_taken = len(rewards)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.10"
        print(f"[END] success={str(success).lower()} steps={steps_taken} rewards={rewards_str}", flush=True)

    return {"scenario_key": scenario_key, "score": score, "steps": steps_taken, "success": success}, env


async def run_task(task_name: str, scenario_keys: List[str], env: WhyDidItFailEnv, client: OpenAI) -> List[float]:
    if not scenario_keys:
        # print(f"  [INFO]    task={task_name} — no scenarios defined yet", flush=True)
        return []

    if USE_LOCAL:
        try:
            from local_agent import LOCAL_MODEL
            effective_model = LOCAL_MODEL
        except Exception:
            effective_model = "local_model"
    else:
        effective_model = MODEL_NAME

    results = []
    for key in scenario_keys:
        res, env = await run_episode(env, client, key, task_name, effective_model)
        results.append(res)

    scores = [r["score"] for r in results]
    task_score = round(max(0.10, min(0.90, sum(scores) / len(scores))), 2) if scores else 0.10
    print(f"[END] score={task_score:.2f}", flush=True)
    return scores


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await _make_env()

    try:
        scores = []
        scores += await run_task("task_easy",   EASY_SCENARIOS,   env, client)
        scores += await run_task("task_medium", MEDIUM_SCENARIOS, env, client)
        scores += await run_task("task_hard",   HARD_SCENARIOS,   env, client)
        pass  # scoring is handled by the yaml grader, not stdout
    finally:
        try:
            await env.close()
        except Exception:
            # print(f"  [DEBUG]   env.close() error: {e}", file=sys.stderr, flush=True)
            pass

if __name__ == "__main__":
    asyncio.run(main())