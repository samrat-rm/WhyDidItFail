"""
Inference Script — WhyDidItFail
===================================
MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN / API_KEY  Your Hugging Face / API key.

TASKS
    Task 1 (easy)   — identify failure mode from logs only
    Task 2 (medium) — identify failure mode from logs + config      [coming soon]
    Task 3 (hard)   — identify failure mode + provide correct fix   [coming soon]

STDOUT FORMAT
    [START]   task=<task_name> scenarios=<n> model=<model_name>
    [EPISODE] scenario=<key> step=<n> action=<json> reward=<0.00> done=<bool>
    [RESULT]  scenario=<key> score=<0.000> steps=<n> success=<bool>
    [SUMMARY] task=<task_name> avg_score=<0.000> pass_rate=<0.00>
"""

import asyncio
import json
import os
import textwrap
from typing import List

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from client import WhyDidItFailEnv
from models import WhyDidItFailAction
from server.scenarios import SCENARIOS

IMAGE_NAME       = os.getenv("IMAGE_NAME", "")
SERVER_URL       = os.getenv("SERVER_URL", "http://localhost:8000")
API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
MAX_STEPS        = 8
TEMPERATURE      = 0.3
MAX_TOKENS       = 256
SUCCESS_THRESHOLD = 0.5

# ── scenario lists by difficulty ─────────────────────────────────────────────

EASY_SCENARIOS   = [k for k, v in SCENARIOS.items() if v["difficulty"] == "easy"]
MEDIUM_SCENARIOS = [k for k, v in SCENARIOS.items() if v["difficulty"] == "medium"]
HARD_SCENARIOS   = [k for k, v in SCENARIOS.items() if v["difficulty"] == "hard"]

# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a machine learning engineer diagnosing a failed training run.
    Each turn you receive data and must decide what to investigate next.

    Available actions:
      inspect_logs       — examine training loss/accuracy curves
      inspect_config     — examine hyperparameter config (lr, optimizer, etc.)
      inspect_gradients  — examine gradient norm statistics
      submit_diagnosis   — submit your final diagnosis (ends the episode)

    Respond with a JSON object on a single line. Examples:
        {"action_type": "inspect_logs"}
        {"action_type": "submit_diagnosis", "diagnosis": "exploding gradients"}
        {"action_type": "submit_diagnosis", "diagnosis": "overfitting", "suggested_fix": "add dropout=0.3"}

    Be efficient — inspect only what you need. Submit when confident.
    The diagnosis should be a short phrase describing the failure mode.
""").strip()


def _user_prompt(step: int, obs_summary: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step {step}

        Observation:
        {obs_summary}

        Recent history:
        {history_block}

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
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _user_prompt(step, obs_summary, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return WhyDidItFailAction(**json.loads(text))
    except Exception as exc:
        print(f"  [DEBUG] parse error: {exc}", flush=True)
        if step <= 2:
            return WhyDidItFailAction(action_type="inspect_logs", diagnosis=None, suggested_fix=None)
        return WhyDidItFailAction(action_type="submit_diagnosis", diagnosis="unknown", suggested_fix=None)

# ── episode runner ────────────────────────────────────────────────────────────

async def run_episode(env: WhyDidItFailEnv, client: OpenAI, scenario_key: str) -> dict:
    """Run one full episode for a specific scenario. Returns result dict."""
    result   = await env.reset(scenario_key=scenario_key)
    obs      = result.observation
    history: List[str] = []
    rewards: List[float] = []

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        action   = _get_action(client, step, _summarize(obs), history)
        result   = await env.step(action)
        obs      = result.observation
        reward   = result.reward or 0.0
        done     = result.done
        act_str  = action.model_dump_json(exclude_none=True)

        rewards.append(reward)
        history.append(f"Step {step}: {act_str} → reward={reward:.2f} | {obs.feedback}")
        print(f"  [EPISODE] scenario={scenario_key} step={step} action={act_str} reward={reward:.2f} done={str(done).lower()}", flush=True)

        if done:
            break

    # Final score = reward on submit_diagnosis (last reward)
    score   = rewards[-1] if rewards else 0.0
    success = score >= SUCCESS_THRESHOLD
    return {"scenario_key": scenario_key, "score": score, "steps": len(rewards), "success": success}


# ── task runners ──────────────────────────────────────────────────────────────

async def run_task(task_name: str, scenario_keys: List[str], env: WhyDidItFailEnv, client: OpenAI) -> None:
    if not scenario_keys:
        print(f"[SUMMARY] task={task_name} — no scenarios defined yet", flush=True)
        return

    print(f"\n[START] task={task_name} scenarios={len(scenario_keys)} model={MODEL_NAME}", flush=True)

    results = []
    for key in scenario_keys:
        res = await run_episode(env, client, key)
        results.append(res)
        print(f"[RESULT] scenario={res['scenario_key']} score={res['score']:.3f} steps={res['steps']} success={str(res['success']).lower()}", flush=True)

    avg_score = sum(r["score"] for r in results) / len(results)
    pass_rate = sum(1 for r in results if r["success"]) / len(results)
    print(f"[SUMMARY] task={task_name} avg_score={avg_score:.3f} pass_rate={pass_rate:.2f}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = (
        await WhyDidItFailEnv.from_docker_image(IMAGE_NAME)
        if IMAGE_NAME
        else WhyDidItFailEnv(base_url=SERVER_URL)
    )

    try:
        await run_task("easy",   EASY_SCENARIOS,   env, client)
        await run_task("medium", MEDIUM_SCENARIOS, env, client)
        await run_task("hard",   HARD_SCENARIOS,   env, client)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())