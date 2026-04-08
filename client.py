# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""WhyDidItFail Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import WhyDidItFailAction, WhyDidItFailObservation, WhyDidItFailState


class WhyDidItFailEnv(EnvClient[WhyDidItFailAction, WhyDidItFailObservation, WhyDidItFailState]):
    """
    Client for the WhyDidItFail Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each instance has its own dedicated session.

    Example:
        >>> with WhyDidItFailEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.task_description)
        ...
        ...     action = WhyDidItFailAction(action_type="inspect_logs")
        ...     result = env.step(action)
        ...     print(result.observation.visible_data)
        ...
        ...     action = WhyDidItFailAction(
        ...         action_type="submit_diagnosis",
        ...         diagnosis="exploding gradients"
        ...     )
        ...     result = env.step(action)
        ...     print(result.observation.feedback, result.reward)
    """

    def _step_payload(self, action: WhyDidItFailAction) -> Dict:
        """Convert WhyDidItFailAction to JSON payload."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[WhyDidItFailObservation]:
        """Parse server response into StepResult[WhyDidItFailObservation]."""
        obs_data = payload.get("observation", {})
        observation = WhyDidItFailObservation(
            task_description=obs_data.get("task_description", ""),
            visible_data=obs_data.get("visible_data", {}),
            available_actions=obs_data.get("available_actions", []),
            steps_taken=obs_data.get("steps_taken", 0),
            reward=obs_data.get("reward", 0.0),
            done=obs_data.get("done", False),
            feedback=obs_data.get("feedback", ""),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> WhyDidItFailState:
        """Parse server response into WhyDidItFailState."""
        return WhyDidItFailState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scenario_key=payload.get("scenario_key"),
            difficulty=payload.get("difficulty"),
            inspection_order=payload.get("inspection_order", []),
            required_sources=payload.get("required_sources", []),
            max_steps=payload.get("max_steps", 0),
        )