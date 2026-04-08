"""
Local LLM agent via Ollama — for testing only.
Called from inference.py when USE_LOCAL=true.
"""

import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

from models import WhyDidItFailAction

# LOCAL_MODEL = "tinyllama"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "phi3")
LOCAL_URL   = os.getenv("LOCAL_URL", "http://127.0.0.1:11434/api/generate")


def _call(prompt: str) -> str:
    res = requests.post(
        LOCAL_URL,
        json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False, "format": "json"},
    )
    return res.json()["response"]


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from the response.

    Handles three common small-model output patterns:
      - Pure JSON object:        {"action_type": ...}
      - JSON wrapped in prose:   Sure! Here: {"action_type": ...}
      - JSON wrapped in a list:  [{"action_type": ...}]
    Uses bracket counting so nested objects don't break the extraction.
    """
    # Try parsing the whole text first (clean output case)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    # Fall back: find first { and walk balanced braces
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text!r}")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError(f"Unbalanced braces in response: {text!r}")


_KNOWN_FIELDS = set(WhyDidItFailAction.model_fields)


def get_action(step: int, prompt: str) -> WhyDidItFailAction:
    """Call the local LLM and parse the response into a WhyDidItFailAction."""
    text = ""
    try:
        text = _call(prompt).strip()
        data = _extract_json(text)
        filtered = {k: v for k, v in data.items() if k in _KNOWN_FIELDS}

        # phi3 sometimes echoes the feedback signal instead of an action, e.g.:
        # {"feedback": "...", "source_to_investigate": "inspect_config"}
        # Recover action_type from source_to_investigate when possible.
        if "action_type" not in filtered or filtered.get("action_type") is None:
            _valid_actions = {"inspect_logs", "inspect_config", "inspect_gradients", "submit_diagnosis"}
            src = data.get("source_to_investigate", "")
            if isinstance(src, str) and src in _valid_actions:
                filtered["action_type"] = src
            else:
                raise ValueError(f"action_type missing in parsed output: {data}")

        return WhyDidItFailAction(**filtered)
    except Exception as exc:
        print(f"  [LOCAL] parse failed (step {step}): {exc} | raw: {text!r}", flush=True)
        # Step-based progression: avoid re-inspecting the same source.
        from typing import cast, Literal
        _fallback = ["inspect_logs", "inspect_config", "inspect_gradients", "submit_diagnosis"]
        action_type = cast(
            Literal["inspect_logs", "inspect_config", "inspect_gradients", "submit_diagnosis"],
            _fallback[min(step - 1, len(_fallback) - 1)],
        )
        diagnosis = "unknown" if action_type == "submit_diagnosis" else None
        return WhyDidItFailAction(action_type=action_type, diagnosis=diagnosis, suggested_fix=None, reasoning=None)