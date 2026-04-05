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

try:
    from ..models import WhyDidItFailAction, WhyDidItFailObservation
    from ..server.scenarios import SCENARIOS
except ImportError:
    from models import WhyDidItFailAction, WhyDidItFailObservation
    from server.scenarios import SCENARIOS


class WhyDidItFailEnvironment(Environment):
    """Diagnostic environment where the agent investigates a failed training run."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.scenario = None
        self.inspected = set()   # tracks what the agent has already looked at

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> WhyDidItFailObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.scenario = random.choice(list(SCENARIOS.values()))
        self.inspected = set()
        return WhyDidItFailObservation(
            task_description="A training run has failed. Diagnose the problem.",
            visible_data={"hint": "Use inspect_logs or inspect_config to begin."},
            available_actions=["inspect_logs","inspect_config",
                               "inspect_gradients","submit_diagnosis"],
            steps_taken=0, reward=0.0, done=False,
            feedback="Investigation started."
        )

    def step(self, action: WhyDidItFailAction, timeout_s: Optional[float] = None, **kwargs: Any) -> WhyDidItFailObservation:
        if self.scenario is None:
            raise RuntimeError("Environment must be reset before calling step.")
        self._state.step_count += 1

        if action.action_type == "inspect_logs":
            self.inspected.add("logs")
            visible = {"training_logs": self.scenario["logs"]}
            feedback = "You examined the training logs."

        elif action.action_type == "inspect_config":
            self.inspected.add("config")
            visible = {"config": self.scenario["config"]}
            feedback = "You examined the hyperparameter config."

        elif action.action_type == "inspect_gradients":
            self.inspected.add("gradients")
            visible = {"gradient_norms": self.scenario["gradient_norms"]}
            feedback = "You examined gradient statistics."

        elif action.action_type == "submit_diagnosis":
            reward, feedback, done = self.grade(action)
            return WhyDidItFailObservation(
                task_description="Diagnosis submitted.",
                visible_data={}, available_actions=[],
                steps_taken=self._state.step_count,
                reward=reward, done=True, feedback=feedback
            )

        else:
            visible = {}
            feedback = f"Unknown action '{action.action_type}'."

        return WhyDidItFailObservation(
            task_description="Continue your investigation.",
            visible_data=visible,
            available_actions=["inspect_logs","inspect_config",
                               "inspect_gradients","submit_diagnosis"],
            steps_taken=self._state.step_count,
            reward=0.0, done=False, feedback=feedback
        )

    def grade(self, action: WhyDidItFailAction) -> tuple[float, str, bool]:
        """Score a submit_diagnosis action against the current scenario."""
        if self.scenario is None:
            raise RuntimeError("Environment must be reset before calling grade.")
        diagnosis = (action.diagnosis or "").strip().lower()
        correct_diagnosis = self.scenario["correct_diagnosis"].strip().lower()
        correct_fix = (self.scenario.get("correct_fix") or "").strip().lower()
        suggested_fix = (action.suggested_fix or "").strip().lower()

        diagnosis_correct = diagnosis == correct_diagnosis
        fix_correct = suggested_fix == correct_fix if correct_fix else True

        if diagnosis_correct and fix_correct:
            reward = 1.0
            feedback = "Correct diagnosis and fix!"
        elif diagnosis_correct:
            reward = 0.5
            feedback = f"Correct diagnosis, but the suggested fix was wrong. Expected: '{self.scenario.get('correct_fix')}'."
        else:
            reward = 0.0
            feedback = f"Incorrect diagnosis. The actual failure mode was '{self.scenario['correct_diagnosis']}'."

        return reward, feedback, True