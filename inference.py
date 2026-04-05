"""
Inference Script — WhyDidItFail
===================================
MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN / API_KEY  Your Hugging Face / API key.

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from client import WhyDidItFailEnv
from models import WhyDidItFailAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("WHYDIDITFAIL_TASK", "whydiditfail")
BENCHMARK = os.getenv("WHYDIDITFAIL_BENCHMARK", "whydiditfail")
MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.5  # reward >= 0.5 counts as success

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a machine learning engineer diagnosing a failed training run.
    Each turn you will receive data from the training run and must decide what to investigate next.

    Available actions:
    - inspect_logs       : examine training loss curves
    - inspect_config     : examine hyperparameter config (lr, optimizer, etc.)
    - inspect_gradients  : examine gradient statistics
    - submit_diagnosis   : submit your final diagnosis (ends the episode)

    You must respond with a JSON object on a single line. Examples:
        {"action_type": "inspect_logs"}
        {"action_type": "inspect_config"}
        {"action_type": "submit_diagnosis", "diagnosis": "exploding gradients"}

    Only submit_diagnosis when you are confident. The diagnosis should describe the failure mode
    in plain terms (e.g. "exploding gradients", "overfitting", "vanishing gradients").
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, observation_summary: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}

        Current observation:
        {observation_summary}

        History:
        {history_block}

        Respond with a JSON action.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, observation_summary: str, history: List[str]) -> WhyDidItFailAction:
    user_prompt = build_user_prompt(step, observation_summary, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        return WhyDidItFailAction(**data)
    except Exception as exc:
        print(f"[DEBUG] Model request/parse failed: {exc}", flush=True)
        # Fallback: inspect logs if early, otherwise give up and submit empty diagnosis
        if step <= 2:
            return WhyDidItFailAction(action_type="inspect_logs")
        return WhyDidItFailAction(action_type="submit_diagnosis", diagnosis="unknown")


def summarize_observation(obs) -> str:
    lines = [
        f"Task: {obs.task_description}",
        f"Feedback: {obs.feedback}",
        f"Available actions: {', '.join(obs.available_actions)}",
    ]
    if obs.visible_data:
        lines.append(f"Data: {json.dumps(obs.visible_data, indent=2)}")
    return "\n".join(lines)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await WhyDidItFailEnv.from_docker_image(IMAGE_NAME or "")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_summary = summarize_observation(obs)
            action = get_model_action(client, step, obs_summary, history)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            action_str = action.model_dump_json(exclude_none=True)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action_str} -> reward={reward:.2f} feedback={obs.feedback!r}")

            if done:
                break

        score = max(rewards) if rewards else 0.0  # final diagnosis reward is what matters
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())