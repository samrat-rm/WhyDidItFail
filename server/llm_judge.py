"""
LLM Judge — reasoning quality scorer for WhyDidItFail.

Called from inference.py after submit_diagnosis.
Uses the same OpenAI-compatible client and model as the agent.
Returns a normalized score in [0.0, 1.0] representing reasoning quality.
Returns 0.0 silently if reasoning is absent or the call fails.

Scoring criteria (0–5 each, total 0–15 → normalized to 0.0–1.0):
  evidence_grounding  — does the reasoning cite specific observed values?
  causal_chain        — does it connect evidence to the failure mode logically?
  fix_rationale       — is the fix justified by the evidence?

Final score in inference.py:
  total = 0.85 * keyword_score + 0.15 * judge_score  → always in [0.0, 1.0]
"""

import json

from openai import OpenAI


def _build_prompt(
    diagnosis: str,
    suggested_fix: str | None,
    reasoning: str,
    scenario: dict,
    inspection_order: list[str],
) -> str:
    seen: dict = {}
    if "logs" in inspection_order:
        seen["training_logs"] = scenario.get("logs", [])
    if "config" in inspection_order:
        seen["config"] = scenario.get("config", {})
    if "gradients" in inspection_order:
        seen["gradient_norms"] = scenario.get("gradient_norms", None)

    return f"""You are evaluating the reasoning of an ML debugging agent.

Agent submission:
  Diagnosis:     {diagnosis}
  Suggested fix: {suggested_fix or "none provided"}
  Reasoning:     {reasoning}

Data the agent had access to:
{json.dumps(seen, indent=2)}

Score the reasoning (integers only):
  evidence_grounding (0-5): Does the reasoning cite specific values from the data above?
  causal_chain       (0-5): Does it logically connect that evidence to the diagnosed failure mode?
  fix_rationale      (0-5): Is the fix directly justified by the evidence and diagnosis?

Respond with JSON only, no explanation:
{{"evidence_grounding": <int>, "causal_chain": <int>, "fix_rationale": <int>}}"""


def judge(
    client: OpenAI,
    model: str,
    diagnosis: str,
    reasoning: str | None,
    suggested_fix: str | None,
    scenario: dict,
    inspection_order: list[str],
) -> float | None:
    """Score reasoning quality. Returns 0.0–1.0, or None if unavailable/failed.

    None signals the caller to skip judge weighting entirely and use the
    keyword score at full weight (1.0) rather than 0.85.
    """
    if not reasoning or not reasoning.strip():
        return None

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": _build_prompt(
                    diagnosis, suggested_fix, reasoning, scenario, inspection_order
                )},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)

        raw = (
            data.get("evidence_grounding", 0)
            + data.get("causal_chain", 0)
            + data.get("fix_rationale", 0)
        )
        # normalize: raw 0–15 → 0.0–1.0
        return round(max(0, min(15, raw)) / 15, 4)

    except Exception as exc:
        print(f"  [JUDGE] failed: {exc}", flush=True)
        return None