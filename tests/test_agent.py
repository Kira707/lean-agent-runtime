from copy import deepcopy
import json
import pytest
from runtime.agent import run
from runtime.types import AgentState, ModelReply, RunBudget, RunStatus, ToolResult, ToolStatus, Usage


def tool(name='lookup', args='{}', id='a'):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': id, 'type': 'function', 'function': {'name': name, 'arguments': args}}]}


class Model:
    def __init__(self, replies):
        self.replies, self.requests = iter(replies), []

    def complete(self, messages, tools, *, timeout_s):
        self.requests.append(deepcopy(messages))
        reply = next(self.replies)
        return ModelReply(reply, Usage(), deepcopy(reply))


class Env:
    mode = 'dialogue'
    done = False
    def __init__(self, finish=False):
        self.calls, self.turns, self.finish = [], [], finish
    def reset(self): return 'hello'
    def list_tools(self): return []
    def execute(self, call):
        self.calls.append(call)
        self.done = self.finish
        return ToolResult(ToolStatus.OK, 'Error: missing argument' if not call.arguments else 'ok')
    def respond(self, text):
        self.turns.append(text)
        self.done = True
        return '###STOP###'
    def final_reward(self): return None


def test_single_shot_end_is_not_reward():
    env = Env(); env.mode = 'single_shot'
    result = run(Model([{'role':'assistant','content':'done'}]), env, env, system='policy')
    assert result.status == RunStatus.COMPLETED and env.final_reward() is None


def test_dialogue_stop_is_recorded_without_another_model_call():
    env, model = Env(), Model([{'role':'assistant','content':'done'}])
    result = run(model, env, env, system='policy')
    assert result.status == RunStatus.USER_ENDED
    assert result.state.messages[-1]['content'] == '###STOP###' and len(model.requests) == 1


def test_first_tool_only_and_raw_unchanged():
    message = tool(args='{"x":1}')
    message['tool_calls'] += tool(args='BROKEN', id='b')['tool_calls']
    env = Env(True)
    result = run(Model([message]), env, env, system='p')
    assert len(env.calls) == 1 and len(result.state.messages[2]['tool_calls']) == 1
    assert len(message['tool_calls']) == 2


def test_tool_terminal_does_not_contact_user():
    env = Env(True)
    result = run(Model([tool()]), env, env, system='p')
    assert result.status == RunStatus.ENV_ENDED and env.turns == []


def test_schema_invalid_object_continues_but_bad_json_stops():
    env = Env()
    result = run(Model([tool(), {'role':'assistant','content':'done'}]), env, env, system='p')
    assert result.state.messages[3]['content'] == 'Error: missing argument'
    assert result.status == RunStatus.USER_ENDED
    result = run(Model([tool(args='not json')]), Env(), Env(), system='p')
    assert result.status == RunStatus.MODEL_ERROR


@pytest.mark.parametrize('state,budget,clock', [
    (AgentState('a', step=1), RunBudget(max_steps=1), lambda:0),
    (AgentState('a', model_calls=1), RunBudget(max_model_calls=1), lambda:0),
    (AgentState('a', usage_total=Usage(input_tokens=10)), RunBudget(max_total_tokens=10), lambda:0),
    (AgentState('a'), RunBudget(max_wall_seconds=1), lambda:1)])
def test_each_budget_blocks_new_request(state, budget, clock):
    model, env = Model([]), Env()
    result = run(model, env, env, system='p', state=state, budget=budget, clock=clock, started=0)
    assert result.status == RunStatus.BUDGET_EXCEEDED and model.requests == []


def test_state_isolation_and_extra_fields_preserved():
    model = Model([tool(), {'role':'assistant','content':'done','reasoning_content':'extra'},
                   {'role':'assistant','content':'again'}])
    first_env, second_env = Env(), Env()
    first = run(model, first_env, first_env, system='p')
    second = run(model, second_env, second_env, system='p')
    first.state.hook_state['x'] = 1
    assert first.state.messages[-2]['reasoning_content'] == 'extra'
    assert second.state.step == 1 and second.state.hook_state == {}
    assert len(second.state.messages) == 4
