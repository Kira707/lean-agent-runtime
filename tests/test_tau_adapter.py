from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from litellm import ModelResponse
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.types import Action
from runtime.agent import run
from runtime.types import ModelReply, RunStatus, ToolCall, Usage
from domains.tau.adapter import TauAdapter

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'docs/results/pilot_20261029_193917/calls.jsonl'


class User:
    def __init__(self, responses=('hello', '###STOP###')): self.responses = iter(responses)
    def reset(self, instruction=None): return next(self.responses)
    def step(self, content): return next(self.responses)
    def get_total_cost(self): return 0


def env(index=4, responses=('hello', '###STOP###')):
    with patch('tau_bench.envs.base.load_user', return_value=User(responses)):
        return MockRetailDomainEnv(task_split='dev', task_index=index)


class Replay:
    def __init__(self, responses): self.responses, self.requests = iter(responses), []
    def complete(self, messages, tools, *, timeout_s):
        self.requests.append(deepcopy(messages))
        raw = next(self.responses)
        return ModelReply(raw['choices'][0]['message'], Usage(), raw)


def test_saved_real_responses_match_reference_actions_messages_and_observations():
    records = [json.loads(s) for s in PILOT.read_text().splitlines()]
    records = [r for r in records if r['task'] == 'retail/dev/4' and r['phase'] != 'initialization']
    agent = [r['response'] for r in records if r['actor'] == 'agent']
    users = [r['response']['choices'][0]['message']['content'] for r in records if r['actor'] == 'user']
    reference_env = env(responses=users)
    iterator = iter(agent)
    def complete(**kwargs):
        response = ModelResponse(**next(iterator))
        response._hidden_params['response_cost'] = 0
        return response
    with patch('tau_bench.agents.tool_calling_agent.completion', complete):
        expected = ToolCallingAgent(reference_env.tools_info, reference_env.wiki, 'fake', 'openai').solve(reference_env, 4)
    actual_env = env(responses=users)
    adapter = TauAdapter(actual_env, 4)
    actual = run(Replay(agent), adapter, adapter, system=actual_env.wiki)
    assert actual.state.messages == expected.messages
    assert [a.model_dump() for a in actual_env.actions] == [a.model_dump() for a in reference_env.actions]
    assert adapter.final_reward() == expected.reward
    assert adapter.list_tools() and [s.wire() for s in adapter.list_tools()] == actual_env.tools_info


def test_same_read_write_sequence_has_same_observations_and_state():
    left, right = env(), env()
    adapter = TauAdapter(right, 4)
    left.reset(task_index=4); adapter.reset()
    order_id = next(k for k, v in left.data['orders'].items() if v['status'] == 'pending')
    actions = [Action(name='get_order_details', kwargs={'order_id':order_id}),
               Action(name='cancel_pending_order', kwargs={'order_id':order_id,'reason':'no longer needed'})]
    for i, action in enumerate(actions):
        expected = left.step(action)
        observed = adapter.execute(ToolCall(str(i), action.name, action.kwargs))
        assert observed.content == expected.observation and adapter.done == expected.done
    assert left.get_data_hash() == right.get_data_hash()


def test_schema_invalid_error_and_schema_invalid_accepted_match_reference():
    left, right = env(), env()
    adapter = TauAdapter(right, 4)
    left.reset(task_index=4); adapter.reset()
    order_id = next(k for k,v in left.data['orders'].items() if v['status']=='pending')
    args = dict(order_id=order_id, address1='a',address2='',city='c',state='s',country='USA',zip=12345)
    for name, kwargs in [('get_order_details', {}), ('modify_pending_order_address', args), ('unknown', {})]:
        expected = left.step(Action(name=name, kwargs=kwargs))
        actual = adapter.execute(ToolCall('x', name, kwargs))
        assert actual.content == expected.observation
    assert right.data['orders'][order_id]['address']['zip'] == 12345


def test_user_and_tool_termination_keep_reward_and_position():
    for name in ('respond', 'transfer_to_human_agents'):
        left, right = env(), env()
        adapter = TauAdapter(right, 4)
        left.reset(task_index=4); adapter.reset()
        args = {'content':'done'} if name=='respond' else {'summary':'done'}
        expected = left.step(Action(name=name, kwargs=args))
        observed = adapter.respond('done') if name=='respond' else adapter.execute(ToolCall('t', name,args)).content
        assert expected.done == adapter.done and expected.observation == observed
        assert expected.reward == adapter.final_reward()


def test_pre_score_evidence_excludes_gold_replay_and_is_immutable():
    adapter = TauAdapter(env(), 4); adapter.reset()
    actual_hash = adapter.env.get_data_hash()
    adapter.respond('done')
    evidence = deepcopy(adapter.evidence)
    assert evidence['data_hash'] == actual_hash and len(evidence['actions']) == 1
    assert len(adapter.env.actions) > len(evidence['actions'])
    adapter.final_reward(); adapter.close()
    assert adapter.evidence == evidence


def test_hidden_task_information_never_enters_model_request():
    raw = {'choices':[{'message':{'role':'assistant','content':'done'}}]}
    model = Replay([raw]); adapter = TauAdapter(env(), 4)
    run(model, adapter, adapter, system=adapter.env.wiki)
    assert model.requests == [[{'role':'system','content':adapter.env.wiki}, {'role':'user','content':'hello'}]]
    assert adapter.task_index == 4
