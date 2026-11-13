from copy import deepcopy
import json
from pathlib import Path
import httpx
import pytest
from runtime.model_client import OpenAIClient, ModelOutputError, RequestPacer, AccountQuotaExceeded
from runtime.types import AgentState, RunBudget

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/m0_pilot.json').read_text())
RAW = {'choices':[{'message':{'role':'assistant','content':'ok'}}],
       'usage':{'prompt_tokens':10,'completion_tokens':2,'prompt_tokens_details':{'cached_tokens':5}}}


def client(handler, monkeypatch, **kwargs):
    monkeypatch.setenv('ARK_AGENT_API_KEY', 'fake-agent-key')
    return OpenAIClient(CONFIG['agent'], CONFIG['prices'], AgentState('test'), RunBudget(),
                        transport=httpx.MockTransport(handler), **kwargs)


def test_route_messages_tools_and_credentials_not_recorded(monkeypatch):
    seen, events = [], []
    def handler(request):
        seen.append(request)
        return httpx.Response(200,json=RAW)
    c = client(handler,monkeypatch,emit=lambda *a,**k:events.append(k))
    messages = [{'role':'user','content':'hi','extra':'preserve'}]
    c.complete(messages, [{'type':'function'}], timeout_s=20)
    assert str(seen[0].url) == CONFIG['agent']['api_base']+'/chat/completions'
    assert json.loads(seen[0].content)['messages'] == messages
    assert seen[0].headers['Authorization'] == 'Bearer fake-agent-key'
    assert 'fake-agent-key' not in json.dumps(events)


def test_real_m0_response_fields_cache_and_unknown_usage(monkeypatch):
    records = [json.loads(l) for l in (ROOT/'docs/results/pilot_20261029_193917/calls.jsonl').read_text().splitlines()]
    raw = next(r['response'] for r in records if r['actor']=='agent' and r['response']['choices'][0]['message'].get('tool_calls'))
    c=client(lambda r:httpx.Response(200,json=raw),monkeypatch)
    reply=c.complete([],[],timeout_s=1)
    assert reply.message == raw['choices'][0]['message']
    assert reply.usage.cached_input_tokens is not None
    unknown=client(lambda r:httpx.Response(200,json={'choices':RAW['choices']}),monkeypatch)
    assert unknown.complete([],[],timeout_s=1).usage.estimated_cost_cny is None


def test_output_boundary_preserves_arguments_for_runtime(monkeypatch):
    raw=deepcopy(RAW); raw['choices'][0]['message']={'role':'assistant','tool_calls':[
        {'id':'a','function':{'name':'lookup','arguments':'not json'}}]}
    c=client(lambda r:httpx.Response(200,json=raw),monkeypatch)
    assert c.complete([],[],timeout_s=1).message['tool_calls'][0]['function']['arguments']=='not json'
    c=client(lambda r:httpx.Response(200,json={'choices':[]}),monkeypatch)
    with pytest.raises(ModelOutputError): c.complete([],[],timeout_s=1)
    for malformed in ([], {'choices':[None]}):
        c=client(lambda r:httpx.Response(200,json=malformed),monkeypatch)
        with pytest.raises(ModelOutputError): c.complete([],[],timeout_s=1)


def test_timeout_retry_counts_attempts_and_keeps_unknown_cost(monkeypatch):
    attempts=[]; events=[]
    def handler(request):
        attempts.append(1)
        if len(attempts)==1: raise httpx.ReadTimeout('timeout',request=request)
        return httpx.Response(200,json=RAW)
    c=client(handler,monkeypatch,max_retries=1,emit=lambda event,**k:events.append({'event':event,**k}))
    c.complete([],[],timeout_s=1)
    assert c.state.model_calls==2 and sum(e['event']=='model_call' for e in events)==2
    assert sum(e['event']=='retry_wait' for e in events)==1
    assert c.state.usage_total.estimated_cost_cny is None


@pytest.mark.parametrize('status,retries,expected', [(429,2,3),(401,2,1),(400,2,1)])
def test_only_429_http_errors_retry(monkeypatch,status,retries,expected):
    c=client(lambda r:httpx.Response(status,json={'error':'test'}),monkeypatch,max_retries=retries)
    with pytest.raises(httpx.HTTPStatusError): c.complete([],[],timeout_s=1)
    assert c.state.model_calls==expected


def test_timeout_clipped_and_m0_cost_function_reused(monkeypatch):
    timeouts=[]
    def handler(r):
        timeouts.append(r.extensions['timeout']['read'])
        return httpx.Response(200,json=RAW)
    c=client(handler,monkeypatch,clock=lambda:598,started=0)
    reply=c.complete([],[],timeout_s=60)
    from m0.recording import cost_from_usage
    assert timeouts==[2]
    assert reply.usage.estimated_cost_cny==cost_from_usage(RAW['usage'],CONFIG['prices'])['estimated_cost_cny']


def test_connect_error_before_http_request_has_known_zero_inference_cost(monkeypatch):
    events=[]
    def handler(request): raise httpx.ConnectError('TLS handshake failed',request=request)
    c=client(handler,monkeypatch,emit=lambda *a,**k:events.append(k))
    with pytest.raises(httpx.ConnectError): c.complete([],[],timeout_s=1)
    assert c.state.model_calls==1 and c.state.usage_total.estimated_cost_cny==0
    assert events[0]['billing_basis']=='request_not_transmitted'


def test_http_error_preserves_original_body_and_request_identifiers(monkeypatch):
    events=[]
    body='{"error":{"code":"ExampleInternalError","message":"failed","request_id":"body-id"}}'
    c=client(lambda r:httpx.Response(500,text=body,
        headers={'x-request-id':'header-id','x-tt-logid':'log-id'}),monkeypatch,
        emit=lambda *a,**k:events.append(k))
    with pytest.raises(httpx.HTTPStatusError): c.complete([],[],timeout_s=1)
    assert events[0]['http_status']==500 and events[0]['response_text']==body
    assert events[0]['response_headers']['x-request-id']=='header-id'
    assert events[0]['response_headers']['x-tt-logid']=='log-id'
    assert events[0]['usage'] is None and c.state.model_calls==1


def test_shared_pacer_separates_model_roles_without_real_sleep():
    now=[0.0];waits=[]
    def sleep(seconds): waits.append(seconds);now[0]+=seconds
    pacer=RequestPacer(3,clock=lambda:now[0],sleep=sleep)
    pacer.wait();pacer.wait();pacer.wait()
    assert waits==[3,3] and now[0]==6


def test_transient_500_waits_sixty_seconds_then_success(monkeypatch):
    now=[0.0];waits=[];attempts=[]
    def sleep(seconds): waits.append(seconds);now[0]+=seconds
    def handler(request):
        attempts.append(1)
        return httpx.Response(500,json={'error':'transient'}) if len(attempts)==1 else httpx.Response(200,json=RAW)
    c=client(handler,monkeypatch,max_retries=2,retry_wait_seconds=60,clock=lambda:now[0],started=0,sleep=sleep)
    reply=c.complete([],[],timeout_s=60)
    assert waits==[60] and c.state.model_calls==2 and reply.message['content']=='ok'
    assert c.state.usage_total.estimated_cost_cny is None and c.state.budget_tokens>0


def test_connect_retry_waits_but_permanent_http_error_does_not(monkeypatch):
    waits=[];attempts=[]
    def handler(request):
        attempts.append(1)
        if len(attempts)==1: raise httpx.ConnectError('TLS',request=request)
        return httpx.Response(200,json=RAW)
    c=client(handler,monkeypatch,max_retries=1,retry_wait_seconds=60,sleep=waits.append)
    c.complete([],[],timeout_s=60)
    assert waits==[60] and c.state.usage_total.estimated_cost_cny is not None
    c=client(lambda r:httpx.Response(401,json={'error':'auth'}),monkeypatch,max_retries=2,retry_wait_seconds=60,sleep=waits.append)
    with pytest.raises(httpx.HTTPStatusError):c.complete([],[],timeout_s=60)
    assert waits==[60] and c.state.model_calls==1


def test_five_hour_account_quota_stops_without_minute_retry(monkeypatch):
    waits=[];events=[]
    body={'error':{'code':'AccountQuotaExceeded','message':'Quota exceeded. It will reset at 2026-11-01 00:51:01 +0800 CST.'}}
    c=client(lambda r:httpx.Response(429,json=body),monkeypatch,max_retries=2,
        retry_wait_seconds=60,sleep=waits.append,emit=lambda *a,**k:events.append(k))
    with pytest.raises(AccountQuotaExceeded) as exc:c.complete([],[],timeout_s=60)
    assert c.state.model_calls==1 and waits==[]
    assert exc.value.reset_at=='2026-11-01T00:51:01+08:00'
    assert events[0]['api_error_code']=='AccountQuotaExceeded'
