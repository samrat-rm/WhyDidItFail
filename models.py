from typing import Any, Dict, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class WhyDidItFailAction(Action):
    """Agent's diagnostic action."""
    action_type: Literal["inspect_logs", "inspect_config", "inspect_gradients", "submit_diagnosis"] = Field(
        ..., description="One of: inspect_logs | inspect_config | inspect_gradients | submit_diagnosis"
    )
    diagnosis: str | None = Field(None, description=
        "Required when action_type=submit_diagnosis. Its the agent's conclusion about what is wrong.")
    suggested_fix: str | None = Field(None, description=
        "Required when action_type=submit_diagnosis. Exact fix to apply.")
    reasoning: str | None = Field(None, description=
        "Required when action_type=submit_diagnosis. Explain what evidence led to this diagnosis.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")


class WhyDidItFailState(State):
    """Full episode state exposed via GET /state and WSStateMessage."""
    scenario_key: str | None = Field(None, description=
        "Key of the active scenario (e.g. 'exploding_gradients'). None before reset.")
    difficulty: str | None = Field(None, description=
        "Difficulty tier of the active scenario: easy, medium, or hard.")
    inspection_order: list[str] = Field(default_factory=list, description=
        "Sources inspected so far this episode, in the order they were first visited.")
    required_sources: list[str] = Field(default_factory=list, description=
        "Sources the agent must inspect before submitting a valid diagnosis.")
    max_steps: int = Field(0, description=
        "Hard step ceiling for this episode. Exceeding it terminates with score 0.")


class WhyDidItFailObservation(Observation):
    """What the agent sees after each action."""
    task_description: str = Field(..., description=
        "The problem the agent must diagnose.")
    visible_data: dict = Field(..., description=
        "Data returned by the last action (logs, config, gradients, etc.).")
    available_actions: list[str] = Field(..., description=
        "Which action_types are valid on this step.")
    steps_taken: int = Field(..., description=
        "Number of actions taken so far in this episode.")
    reward: float = Field(default=0.10, description=    # type: ignore[override]
        "Score for the current step. 0.90 = max.")
    done: bool = Field(default=False, description=
        "True when the episode has ended.")
    feedback: str = Field(..., description=
        "Partial progress hint from the environment.")    