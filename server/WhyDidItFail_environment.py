import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import WhyDidItFailAction, WhyDidItFailObservation
from server.scenarios import SCENARIOS
from server.graders import grade


class WhyDidItFailEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.scenario: dict | None = None
        self.inspection_order: list[str] = []  # first-visit order; doubles as membership check
        self.max_steps: int = 0

    @property
    def state(self) -> State:
        return self._state

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> WhyDidItFailObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self.inspection_order = []

        scenario_key = kwargs.get("scenario_key")

        if scenario_key and scenario_key in SCENARIOS:
            self.scenario = SCENARIOS[scenario_key]
        else:
            if seed is not None:
                random.seed(seed)
            self.scenario = random.choice(list(SCENARIOS.values()))

        required_sources = self.scenario.get("required_sources", ["logs"])
        self.max_steps = len(required_sources) * 3 + 2

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

        # Hard step limit — terminate immediately, grade() will return 0.0.
        if self._state.step_count > self.max_steps and action.action_type != "submit_diagnosis":
            return WhyDidItFailObservation(
                task_description="Step limit reached. Episode terminated.",
                visible_data={},
                available_actions=[],
                steps_taken=self._state.step_count,
                reward=0.0,
                done=True,
                feedback=(
                    f"Step limit ({self.max_steps}) reached without a diagnosis. "
                    f"Score: 0.00. Actual failure: '{self.scenario['correct_diagnosis']}'."
                ),
            )
        required: list[str] = self.scenario.get("required_sources", ["logs"])

        if action.action_type == "inspect_logs":
            step_reward = self._inspect_reward("logs", required)
            if "logs" not in self.inspection_order:
                self.inspection_order.append("logs")
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
            if "config" not in self.inspection_order:
                self.inspection_order.append("config")
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
            if "gradients" not in self.inspection_order:
                self.inspection_order.append("gradients")
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

    # Rewards decay as more required sources are discovered — first clue is worth most.
    _REQUIRED_STEP_REWARDS = [0.10, 0.07, 0.05]

    def _inspect_reward(self, source: str, required: list[str]) -> float:
        """Return step reward for an inspect action.

        Required sources:   progressive — +0.10 / +0.07 / +0.05 for 1st/2nd/3rd discovery.
        Irrelevant sources: -0.03 (mild; some exploration is acceptable).
        Re-inspection:      -0.05 (waste).
        """
        if source in self.inspection_order:
            return -0.05   # redundant inspection

        if source in required:
            n_found = sum(1 for s in self.inspection_order if s in required)
            idx = min(n_found, len(self._REQUIRED_STEP_REWARDS) - 1)
            return self._REQUIRED_STEP_REWARDS[idx]

        return -0.03       # irrelevant source

    def _inspect_feedback(self, source: str, required: list[str], reward: float) -> str:
        label = {"logs": "training logs", "config": "hyperparameter config", "gradients": "gradient statistics"}[source]
        if source in self.inspection_order:
            return f"You already examined the {label}. No new information gained."
        if source in required:
            remaining_sources = [s for s in required if s not in self.inspection_order and s != source]
            msg = f"You examined the {label}. Relevant clue found (+{reward:.2f})."
            if remaining_sources:
                next_source = f"inspect_{remaining_sources[0]}"
                msg += f" {len(remaining_sources)} required source(s) still unexamined. Next required action: {next_source}."
            return msg
        return f"You examined the {label}. This source is not required for this failure mode."

    def _grade(self, action: WhyDidItFailAction) -> tuple[float, str]:
        """Delegate to the unified grade() function and return (reward, feedback)."""
        assert self.scenario is not None
        diagnosis     = (action.diagnosis or "").strip().lower()
        suggested_fix = (action.suggested_fix or "").strip().lower() or None
        difficulty    = self.scenario["difficulty"]

        reward = grade(
            diagnosis=diagnosis,
            suggested_fix=suggested_fix,
            scenario=self.scenario,
            steps_taken=self._state.step_count,
            inspection_order=self.inspection_order,
            difficulty=difficulty,
        )

        if reward >= 0.80:
            feedback = f"Excellent diagnosis! Score: {reward:.2f}"
        elif reward >= 0.50:
            feedback = f"Partially correct. Score: {reward:.2f}. Actual failure: '{self.scenario['correct_diagnosis']}'."
        else:
            feedback = f"Incorrect diagnosis. Score: {reward:.2f}. Actual failure: '{self.scenario['correct_diagnosis']}'."

        return reward, feedback