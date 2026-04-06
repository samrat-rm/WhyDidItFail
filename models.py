from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class WhyDidItFailAction(Action):
    """Agent's diagnostic action."""
    action_type: str = Field(..., description=
        "One of: inspect_logs | inspect_config | inspect_gradients | submit_diagnosis")
    diagnosis: str | None = Field(None, description=
        "Required when action_type=submit_diagnosis. Its the agent's conclusion about what is wrong.")
    suggested_fix: str | None = Field(None, description=
        "Required when action_type=submit_diagnosis. Exact fix to apply.")
    reasoning: str | None = Field(None, description=
        "Required when action_type=submit_diagnosis. Explain what evidence led to this diagnosis.")


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
    reward: float = Field(default=0.0, description=    # type: ignore[override]
        "Score for the current step. 1.0 = solved.")  
    done: bool = Field(default=False, description=
        "True when the episode has ended.")
    feedback: str = Field(..., description=
        "Partial progress hint from the environment.")    