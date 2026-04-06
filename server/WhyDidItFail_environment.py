# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""WhyDidItFail Environment Implementation."""

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import WhyDidItFailAction, WhyDidItFailObservation
from server.scenarios import SCENARIOS
from server.graders import grade


class WhyDidItFailEnvironment(Environment):
    """Diagnostic environment where the agent investigates a failed ML training run."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.scenario: dict | None = None
        self.inspected: set[str] = set()

    @property
    def state(self) -> State:
        return self._state

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> WhyDidItFailObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self.inspected = set()

        scenario_key = kwargs.get("scenario_key")
        if scenario_key and scenario_key in SCENARIOS:
            self.scenario = SCENARIOS[scenario_key]
        else:
            if seed is not None:
                random.seed(seed)
            self.scenario = random.choice(list(SCENARIOS.values()))
        return WhyDidItFailObservation(
            task_description=(
                "A training run has failed. Diagnose the root cause.\n"
                f"Difficulty: {self.scenario['difficulty']}. "
                "Available actions: inspect_logs, inspect_config, inspect_gradients, submit_diagnosis."
            ),
            visible_data={"hint": "Start by inspecting the training logs."},
            available_actions=["inspect_logs", "inspect_config", "inspect_gradients", "submit_diagnosis"],
            steps_taken=0,
            reward=0.0,
            done=False,
            feedback="Investigation started.",
        )

    def step(self, action: WhyDidItFailAction, timeout_s: Optional[float] = None, **kwargs: Any) -> WhyDidItFailObservation:
        if self.scenario is None:
            raise RuntimeError("Environment must be reset before calling step.")

        self._state.step_count += 1
        required: list[str] = self.scenario.get("required_sources", ["logs"])

        if action.action_type == "inspect_logs":
            step_reward = self._inspect_reward("logs", required)
            self.inspected.add("logs")
            return WhyDidItFailObservation(
                task_description="Continue your investigation.",
                visible_data={"training_logs": self.scenario["logs"]},
                available_actions=["inspect_logs", "inspect_config", "inspect_gradients", "submit_diagnosis"],
                steps_taken=self._state.step_count,
                reward=step_reward,
                done=False,
                feedback=self._inspect_feedback("logs", required, step_reward),
            )

        elif action.action_type == "inspect_config":
            step_reward = self._inspect_reward("config", required)
            self.inspected.add("config")
            return WhyDidItFailObservation(
                task_description="Continue your investigation.",
                visible_data={"config": self.scenario["config"]},
                available_actions=["inspect_logs", "inspect_config", "inspect_gradients", "submit_diagnosis"],
                steps_taken=self._state.step_count,
                reward=step_reward,
                done=False,
                feedback=self._inspect_feedback("config", required, step_reward),
            )

        elif action.action_type == "inspect_gradients":
            step_reward = self._inspect_reward("gradients", required)
            self.inspected.add("gradients")
            return WhyDidItFailObservation(
                task_description="Continue your investigation.",
                visible_data={"gradient_norms": self.scenario["gradient_norms"]},
                available_actions=["inspect_logs", "inspect_config", "inspect_gradients", "submit_diagnosis"],
                steps_taken=self._state.step_count,
                reward=step_reward,
                done=False,
                feedback=self._inspect_feedback("gradients", required, step_reward),
            )

        elif action.action_type == "submit_diagnosis":
            final_reward, feedback = self._grade(action)
            return WhyDidItFailObservation(
                task_description="Diagnosis submitted.",
                visible_data={},
                available_actions=[],
                steps_taken=self._state.step_count,
                reward=final_reward,
                done=True,
                feedback=feedback,
            )

        else:
            return WhyDidItFailObservation(
                task_description="Continue your investigation.",
                visible_data={},
                available_actions=["inspect_logs", "inspect_config", "inspect_gradients", "submit_diagnosis"],
                steps_taken=self._state.step_count,
                reward=-0.05,
                done=False,
                feedback=f"Unknown action '{action.action_type}'. No reward.",
            )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _inspect_reward(self, source: str, required: list[str]) -> float:
        """Return step reward for an inspect action."""
        if source in self.inspected:
            return -0.05   # redundant inspection
        if source in required:
            return +0.05   # useful evidence
        return -0.05       # irrelevant source

    def _inspect_feedback(self, source: str, required: list[str], reward: float) -> str:
        label = {"logs": "training logs", "config": "hyperparameter config", "gradients": "gradient statistics"}[source]
        if source in self.inspected:
            return f"You already examined the {label}. No new information gained."
        if reward > 0:
            return f"You examined the {label}. This looks relevant."
        return f"You examined the {label}. This may not be relevant to the failure."

    def _grade(self, action: WhyDidItFailAction) -> tuple[float, str]:
        """Delegate to the unified grade() function and return (reward, feedback)."""
        assert self.scenario is not None
        diagnosis    = (action.diagnosis or "").strip().lower()
        suggested_fix = (action.suggested_fix or "").strip().lower() or None
        difficulty   = self.scenario["difficulty"]

        reward = grade(
            diagnosis=diagnosis,
            suggested_fix=suggested_fix,
            scenario=self.scenario,
            steps_taken=self._state.step_count,
            inspected=self.inspected,
            difficulty=difficulty,
        )

        if reward >= 0.80:
            feedback = f"Excellent diagnosis! Score: {reward:.2f}"
        elif reward >= 0.50:
            feedback = f"Partially correct. Score: {reward:.2f}. Actual failure: '{self.scenario['correct_diagnosis']}'."
        else:
            feedback = f"Incorrect diagnosis. Score: {reward:.2f}. Actual failure: '{self.scenario['correct_diagnosis']}'."

        return reward, feedback