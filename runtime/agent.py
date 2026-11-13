"""Model -> first tool or user -> observation, retaining reference messages."""
from copy import deepcopy
import json
import time

from runtime.types import AgentState, RunBudget, RunResult, RunStatus, ToolCall


def budget_reason(state, budget, elapsed):
    if state.step >= budget.max_steps:
        return 'max_steps'
    if state.model_calls >= budget.max_model_calls:
        return 'max_model_calls'
    usage = state.usage_total
    tokens = (usage.input_tokens + usage.output_tokens
              if usage.input_tokens is not None and usage.output_tokens is not None else state.budget_tokens)
    if tokens >= budget.max_total_tokens:
        return 'max_total_tokens'
    if elapsed >= budget.max_wall_seconds:
        return 'max_wall_seconds'
    return None


def selected_call(message):
    calls = message.get('tool_calls')
    if not calls:
        return None
    call = calls[0]
    arguments = json.loads(call['function']['arguments'])
    if not isinstance(arguments, dict):
        raise ValueError('selected tool arguments must be a JSON object')
    if not isinstance(call['id'], str) or not isinstance(call['function']['name'], str):
        raise ValueError('selected tool id and name must be strings')
    return ToolCall(call['id'], call['function']['name'], arguments)


def run(model, env, provider, *, system, state=None, budget=None,
        emit=lambda *args, **kwargs: None, clock=time.monotonic, started=None,
        request_timeout=60, before_respond=None, max_text_retries=2):
    """Client counts all attempts/usage; environment grading stays outside Runtime."""
    state = state if state is not None else AgentState('local')
    budget = budget or RunBudget()
    started = clock() if started is None else started
    if not state.messages:
        state.messages = [{'role': 'system', 'content': system},
                          {'role': 'user', 'content': env.reset()}]
    tools = [tool.wire() for tool in provider.list_tools()]
    text_retries = 0
    h = None
    if before_respond is not None:
        h = state.hook_state.setdefault('payment', {'blocked_messages':0,'blocked_facts':0,
            'rewrite_requests':0,'extra_agent_calls':0,'extra_steps':0,'fail_open_messages':0})
    while True:
        reason = budget_reason(state, budget, clock() - started)
        if reason:
            return RunResult(RunStatus.BUDGET_EXCEEDED, state, reason)
        if h is not None and text_retries:
            h['extra_agent_calls'] += 1
        reply = model.complete(deepcopy(state.messages), tools,
            timeout_s=min(request_timeout, budget.max_wall_seconds - (clock() - started)))
        state.step += 1
        if h is not None and text_retries:
            h['extra_steps'] += 1
        message = deepcopy(reply.message)
        # Parse only the selected call. The ignored tail can contain malformed JSON.
        try:
            call = selected_call(message)
        except (KeyError, TypeError, IndexError, ValueError) as error:
            emit('model_error', error=str(error))
            return RunResult(RunStatus.MODEL_ERROR, state, str(error))
        if call:
            message['tool_calls'] = message['tool_calls'][:1]
            result = provider.execute(call)
            state.messages.extend([message, {'role': 'tool', 'tool_call_id': call.id,
                'name': call.name, 'content': result.content}])
            emit('tool_call', call=call, result=result)
            if env.done:
                return RunResult(RunStatus.ENV_ENDED, state)
        elif env.mode == 'dialogue':
            feedback = before_respond(state.messages,message.get('content')) if before_respond else None
            if feedback and text_retries < max_text_retries:
                state.messages.extend([message,{'role':'user','content':feedback['text']}])
                text_retries += 1
                h['blocked_messages'] += 1
                h['blocked_facts'] += len(feedback['problems'])
                h['rewrite_requests'] += 1
                emit('h_block',message_index=len(state.messages)-1,feedback_message_index=len(state.messages),
                    text=message.get('content'),problems=feedback['problems'],rewrite_attempt=text_retries)
                continue
            if feedback:
                h['fail_open_messages'] += 1
                emit('h_fail_open',message_index=len(state.messages)+1,text=message.get('content'),
                     problems=feedback['problems'],rewrite_attempts=text_retries)
            elif text_retries:
                emit('h_rewrite_accepted',message_index=len(state.messages)+1,text=message.get('content'),
                     rewrite_attempts=text_retries)
            observation = env.respond(message.get('content'))
            state.messages.extend([message, {'role': 'user', 'content': observation}])
            emit('user_turn', text=message.get('content'), observation=observation)
            text_retries = 0
            if env.done:
                return RunResult(RunStatus.USER_ENDED, state)
        else:
            state.messages.append(message)
            return RunResult(RunStatus.COMPLETED, state)
