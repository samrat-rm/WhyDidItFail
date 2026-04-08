"""
Quick state verification script.

Runs a single hard episode (vanishing_gradients — requires all 3 sources)
and calls env.state() after each action. Prints a diff of what changed.

Usage:
    uv run python test_state.py
    SERVER_URL=http://localhost:8001 uv run python test_state.py
"""

import asyncio
import os

from client import WhyDidItFailEnv
from models import WhyDidItFailAction


SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
SCENARIO    = "vanishing_gradients"   # hard — needs logs + config + gradients


def _fmt(state) -> dict:
    return {
        "episode_id":       state.episode_id,
        "step_count":       state.step_count,
        "scenario_key":     state.scenario_key,
        "difficulty":       state.difficulty,
        "inspection_order": state.inspection_order,
        "required_sources": state.required_sources,
        "max_steps":        state.max_steps,
    }


def _check(label: str, state, expected: dict):
    s = _fmt(state)
    failures = []
    for k, v in expected.items():
        if s[k] != v:
            failures.append(f"  FAIL  {k}: expected {v!r}, got {s[k]!r}")
    if failures:
        print(f"\n[{label}] FAILED")
        for f in failures:
            print(f)
    else:
        print(f"[{label}] OK  — {s}")


async def main():
    env = WhyDidItFailEnv(base_url=SERVER_URL)

    print(f"Connecting to {SERVER_URL} ...")

    # ── 1. State before reset ─────────────────────────────────────────────────
    state = await env.state()
    _check("before reset", state, {
        "scenario_key":     None,
        "difficulty":       None,
        "inspection_order": [],
        "required_sources": [],
        "max_steps":        0,
    })

    # ── 2. State after reset ──────────────────────────────────────────────────
    await env.reset(scenario_key=SCENARIO)
    state = await env.state()
    _check("after reset", state, {
        "scenario_key":     SCENARIO,
        "difficulty":       "hard",
        "inspection_order": [],
        "required_sources": ["logs", "config", "gradients"],
        "max_steps":        11,   # 3 required × 3 + 2
    })

    # ── 3. After inspect_logs ─────────────────────────────────────────────────
    await env.step(WhyDidItFailAction(action_type="inspect_logs",diagnosis=None,suggested_fix=None,reasoning=None))
    state = await env.state()
    _check("after inspect_logs", state, {
        "step_count":       1,
        "inspection_order": ["logs"],
        "required_sources": ["logs", "config", "gradients"],
    })

    # ── 4. After inspect_config ───────────────────────────────────────────────
    await env.step(WhyDidItFailAction(action_type="inspect_config", diagnosis=None, suggested_fix=None, reasoning=None))
    state = await env.state()
    _check("after inspect_config", state, {
        "step_count":       2,
        "inspection_order": ["logs", "config"],
    })

    # ── 5. After inspect_gradients ────────────────────────────────────────────
    await env.step(WhyDidItFailAction(action_type="inspect_gradients", diagnosis=None, suggested_fix=None, reasoning=None))
    state = await env.state()
    _check("after inspect_gradients", state, {
        "step_count":       3,
        "inspection_order": ["logs", "config", "gradients"],
    })

    # ── 6. After re-inspection (should not duplicate) ─────────────────────────
    await env.step(WhyDidItFailAction(action_type="inspect_logs", diagnosis=None, suggested_fix=None, reasoning=None))
    state = await env.state()
    _check("after re-inspect logs", state, {
        "step_count":       4,
        "inspection_order": ["logs", "config", "gradients"],   # no duplicate
    })

    # ── 7. After submit — episode done ────────────────────────────────────────
    await env.step(WhyDidItFailAction(
        action_type="submit_diagnosis",
        diagnosis="vanishing gradients",
        suggested_fix="switch activation to relu and add batch normalization",
        reasoning="gradient norms decay from 0.21 at output to 1e-8 at layer_1; config shows activation=sigmoid",
    ))
    state = await env.state()
    _check("after submit", state, {
        "step_count":       5,
        "scenario_key":     SCENARIO,
        "inspection_order": ["logs", "config", "gradients"],
    })

    await env.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())