from typing import Literal, Protocol
from runtime.types import ModelReply, ToolCall, ToolResult, ToolSpec


class ModelClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict], *, timeout_s: float) -> ModelReply: ...


class ToolProvider(Protocol):
    def list_tools(self) -> list[ToolSpec]: ...
    def execute(self, call: ToolCall) -> ToolResult: ...


class Environment(Protocol):
    mode: Literal['dialogue', 'single_shot']
    done: bool

    def reset(self) -> str: ...
    def respond(self, text: str) -> str: ...
    def final_reward(self) -> float | None: ...
