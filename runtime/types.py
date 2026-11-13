"""Internal contracts. External inputs are validated at their entry boundary."""
from dataclasses import dataclass, field
from enum import Enum


class ToolStatus(str, Enum):
    OK = 'ok'
    RETRYABLE_ERROR = 'retryable_error'
    FATAL_ERROR = 'fatal_error'
    BLOCKED = 'blocked'


class RunStatus(str, Enum):
    USER_ENDED = 'user_ended'
    ENV_ENDED = 'env_ended'
    COMPLETED = 'completed'
    BUDGET_EXCEEDED = 'budget_exceeded'
    LOOP_BROKEN = 'loop_broken'
    MODEL_ERROR = 'model_error'
    ENV_ERROR = 'env_error'


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    side_effect: bool = False
    id_args: list[str] = field(default_factory=list)

    def wire(self):
        return {'type': 'function', 'function': {'name': self.name,
                'description': self.description, 'parameters': self.parameters}}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    status: ToolStatus
    content: str
    error_code: str | None = None
    recovery_hint: str | None = None
    data: dict | None = None


@dataclass
class Usage:
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    cached_input_tokens: int | None = 0
    estimated_cost_cny: float | None = 0.0
    cost_upper_bound_cny: float | None = 0.0

    def add(self, other):
        for key in self.__dataclass_fields__:
            left, right = getattr(self, key), getattr(other, key)
            setattr(self, key, left + right if left is not None and right is not None else None)


@dataclass
class ModelReply:
    message: dict
    usage: Usage
    raw_response: dict


@dataclass
class RunBudget:
    max_steps: int = 30
    max_model_calls: int = 80
    max_total_tokens: int = 400000
    max_wall_seconds: float = 600


@dataclass
class AgentState:
    run_id: str
    messages: list[dict] = field(default_factory=list)
    step: int = 0
    model_calls: int = 0
    budget_tokens: int = 0  # Known tokens plus request-size reserves for uncertain attempts.
    usage_total: Usage = field(default_factory=Usage)
    hook_state: dict = field(default_factory=dict)


@dataclass
class RunResult:
    status: RunStatus
    state: AgentState
    reason: str | None = None
