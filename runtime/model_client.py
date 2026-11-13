"""Explicit OpenAI-compatible transport; shared state includes user simulation."""
from dataclasses import asdict
from datetime import datetime
import json
import os
import re
import time
from threading import Lock
import httpx

from m0.recording import cost_from_usage
from runtime.types import ModelReply, Usage


class BudgetExceeded(RuntimeError):
    pass


class ModelOutputError(ValueError):
    pass


class AccountQuotaExceeded(RuntimeError):
    def __init__(self, message, reset_at):
        super().__init__(message)
        self.reset_at = reset_at


class RequestPacer:
    """One start-to-start interval shared by Agent and user simulator."""
    def __init__(self, interval, *, clock=time.monotonic, sleep=time.sleep):
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self.last = None
        self.lock = Lock()

    def wait(self):
        with self.lock:
            delay = 0 if self.last is None else max(0, self.interval-(self.clock()-self.last))
            if delay: self.sleep(delay)
            self.last = self.clock()


def parse_reply(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get('choices'), list) or not raw['choices']:
        raise ModelOutputError('response.choices must be a nonempty list')
    if not isinstance(raw['choices'][0], dict):
        raise ModelOutputError('response.choices[0] must be an object')
    message = raw['choices'][0].get('message')
    if not isinstance(message, dict) or message.get('role') != 'assistant':
        raise ModelOutputError('response.choices[0].message must be an assistant object')
    calls = message.get('tool_calls')
    if calls is not None and not isinstance(calls, list):
        raise ModelOutputError('message.tool_calls must be a list or null')
    if not calls and not isinstance(message.get('content'), str):
        raise ModelOutputError('message.content must be text when no tool is called')
    return message


class OpenAIClient:
    def __init__(self, route, prices, state, budget, *, actor='agent', emit=lambda *a, **k: None,
                 transport=None, started=None, clock=time.monotonic, max_retries=0,
                 max_output_tokens=4096, guard=lambda: None, pace=lambda: None,
                 retry_wait_seconds=0, sleep=time.sleep):
        self.route, self.prices, self.state, self.budget = route, prices, state, budget
        self.actor, self.emit, self.clock, self.guard = actor, emit, clock, guard
        self.started = clock() if started is None else started
        self.phase, self.max_retries, self.max_output_tokens = 'task', max_retries, max_output_tokens
        self.pace, self.retry_wait_seconds, self.sleep = pace, retry_wait_seconds, sleep
        self.http = httpx.Client(transport=transport)

    def complete(self, messages, tools, *, timeout_s):
        body = {'model': self.route['model'], 'temperature': self.route['temperature'],
                'messages': messages, 'max_tokens': self.max_output_tokens}
        if tools:
            body['tools'] = tools
        for attempt in range(self.max_retries + 1):
            remaining = self.budget.max_wall_seconds - (self.clock() - self.started)
            if remaining <= 0 or self.state.model_calls >= self.budget.max_model_calls:
                raise BudgetExceeded('model call or wall-clock budget')
            if self.state.budget_tokens >= self.budget.max_total_tokens:
                raise BudgetExceeded('token budget')
            self.guard()
            self.pace()
            remaining = self.budget.max_wall_seconds - (self.clock() - self.started)
            if remaining <= 0: raise BudgetExceeded('wall-clock budget after pacing')
            self.state.model_calls += 1
            start = self.clock()
            request = body | {'api_base': self.route['api_base'], 'timeout_s': min(timeout_s, remaining)}
            try:
                response = self.http.post(self.route['api_base'].rstrip('/') + '/chat/completions',
                    json=body, headers={'Authorization': 'Bearer ' + os.environ[self.route['api_key_env']]},
                    timeout=min(timeout_s, remaining))
                response.raise_for_status()
            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as error:
                # Failed attempts keep unknown usage. A configured retry is part of
                # this call; after it, the batch stops when any cost remains unknown.
                # httpcore HTTPConnection._connect precedes HTTP11Connection's
                # send_request_headers/body. ConnectError has no inference request.
                untransmitted = isinstance(error, httpx.ConnectError)
                failed_usage = Usage() if untransmitted else Usage(None, None, None, None, None)
                self.state.usage_total.add(failed_usage)
                if not untransmitted:
                    # A scheduling reserve, never reported as measured token usage.
                    self.state.budget_tokens += len(json.dumps(body,ensure_ascii=False).encode('utf-8')) + self.max_output_tokens
                error_response = {}
                if isinstance(error, httpx.HTTPStatusError):
                    error_response = {'http_status': error.response.status_code,
                        'response_text': error.response.text,
                        'response_headers': {key: value for key, value in error.response.headers.items()
                            if key in ('x-request-id', 'request-id', 'x-tt-logid',
                                       'x-ark-request-id', 'traceparent', 'content-type')}}
                    if error.response.status_code == 429:
                        try:
                            payload = error.response.json()
                        except json.JSONDecodeError:
                            payload = {}
                        api_error = payload.get('error',{}) if isinstance(payload,dict) else {}
                        if isinstance(api_error,dict) and api_error.get('code')=='AccountQuotaExceeded':
                            match = re.search(r'reset at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4})',api_error['message'])
                            error_response['quota_reset_at'] = datetime.strptime(match[1],'%Y-%m-%d %H:%M:%S %z').isoformat() if match else None
                            error_response['api_error_code'] = 'AccountQuotaExceeded'
                self.emit('model_call', actor=self.actor, phase=self.phase, attempt=attempt,
                    request=request, response=None, usage=asdict(failed_usage) if untransmitted else None,
                    billing_basis='request_not_transmitted' if untransmitted else 'unknown', error_type=type(error).__name__,
                    error=str(error), wall_ms=round((self.clock()-start)*1000), **error_response)
                if error_response.get('api_error_code')=='AccountQuotaExceeded':
                    raise AccountQuotaExceeded(api_error['message'],error_response['quota_reset_at']) from error
                retry = isinstance(error, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)) or (
                    isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (429,500,502,503,504))
                if retry and attempt < self.max_retries:
                    if self.clock()-self.started+self.retry_wait_seconds >= self.budget.max_wall_seconds:
                        raise BudgetExceeded('retry wait exceeds wall-clock budget') from error
                    self.emit('retry_wait',actor=self.actor,seconds=self.retry_wait_seconds,
                              next_attempt=attempt+1,error_type=type(error).__name__)
                    self.sleep(self.retry_wait_seconds)
                    continue
                raise
            try:
                raw = response.json()
            except json.JSONDecodeError as error:
                self.state.usage_total.add(Usage(None, None, None, None, None))
                self.emit('model_call', actor=self.actor, phase=self.phase, attempt=attempt,
                    request=request, response=None, response_text=response.text, usage=None,
                    error_type='ModelOutputError', error='response is not JSON',
                    wall_ms=round((self.clock()-start)*1000))
                raise ModelOutputError('response is not JSON') from error
            cost = cost_from_usage((raw.get('usage') or {}) if isinstance(raw, dict) else {}, self.prices)
            usage = Usage(**{key: cost[key] for key in Usage.__dataclass_fields__})
            self.state.usage_total.add(usage)
            self.state.budget_tokens += (usage.input_tokens+usage.output_tokens
                if usage.input_tokens is not None and usage.output_tokens is not None
                else len(json.dumps(body,ensure_ascii=False).encode('utf-8'))+self.max_output_tokens)
            self.emit('model_call', actor=self.actor, phase=self.phase, attempt=attempt,
                request=request, response=raw, usage=asdict(usage),
                cost_unknown_reason=cost['cost_unknown_reason'], wall_ms=round((self.clock()-start)*1000))
            return ModelReply(parse_reply(raw), usage, raw)

    def close(self):
        self.http.close()
