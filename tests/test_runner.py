from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from unittest.mock import patch
import httpx
import pytest

from eval.runner import BatchBudget, USER_CLIENT, execute_job, run_batch, plan
from runtime.types import RunBudget, Usage, ModelReply
from runtime.store import RunStore
from tau_bench.envs.retail.env import MockRetailDomainEnv

CONFIG=json.loads(__import__('pathlib').Path('configs/m1_baseline.json').read_text())


class FakeClient:
    def __init__(self,route,prices,state,budget,*,actor,emit,guard,**kwargs):
        self.state,self.actor,self.emit,self.guard=state,actor,emit,guard
        self.phase='task'
    def complete(self,messages,tools,*,timeout_s):
        self.guard();self.state.model_calls+=1
        usage=Usage(1,1,0,.01,.01);self.state.usage_total.add(usage)
        self.emit('model_call',actor=self.actor,phase=self.phase,usage=asdict(usage),request={'messages':messages})
        return ModelReply({'role':'assistant','content':self.state.run_id},usage,{})
    def close(self): pass


class User:
    def reset(self,instruction=None):
        USER_CLIENT.get().complete([],[],timeout_s=1)
        return 'hello'
    def step(self,text):
        USER_CLIENT.get().complete([],[],timeout_s=1)
        return '###STOP###'
    def get_total_cost(self): return None


def make_env(task):
    return MockRetailDomainEnv(task_split=task['split'],task_index=task['index'])


def test_concurrent_real_run_boundaries_isolate_state_and_records(tmp_path):
    jobs=plan({'development':[{'domain':'retail','split':'dev','index':4}]},trials=2)
    batch,store=BatchBudget(5),RunStore(tmp_path/'runs.sqlite')
    with patch('tau_bench.envs.base.load_user',side_effect=lambda **k:User()),ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(execute_job,j,CONFIG,RunBudget(),batch,tmp_path,store,make_env,FakeClient) for j in jobs]
        rows=[f.result() for f in futures]
    for row in rows:
        assert row['model_calls']==3 and row['usage']['input_tokens']==3
        messages=json.loads((tmp_path/row['run_id']/'messages.json').read_text())
        assert messages[-2]['content']==row['run_id']
    assert batch.spent==pytest.approx(.06) and len(store.all())==2


def test_named_model_error_is_saved_and_next_job_can_complete(tmp_path):
    class Broken(FakeClient):
        def complete(self,*a,**k): raise httpx.ReadTimeout('test timeout')
    jobs=plan({'development':[{'domain':'retail','split':'dev','index':4}]},trials=2)
    batch,store=BatchBudget(5),RunStore(tmp_path/'runs.sqlite')
    with patch('tau_bench.envs.base.load_user',side_effect=lambda **k:User()):
        first=execute_job(jobs[0],CONFIG,RunBudget(),batch,tmp_path,store,make_env,Broken)
        second=execute_job(jobs[1],CONFIG,RunBudget(),batch,tmp_path,store,make_env,FakeClient)
    assert first['status']=='model_error' and first['reward'] is None
    assert second['status']=='user_ended'


@pytest.mark.parametrize('unknown',[False,True])
def test_batch_cost_or_unknown_stops_unstarted_jobs(tmp_path,unknown):
    jobs=plan({'development':[{'domain':'retail','split':'dev','index':4}]},trials=3)
    def worker(job,config,budget,batch,folder,store):
        batch.record('model_call', {'usage':None if unknown else {'estimated_cost_cny':10}})
        row=job|{'status':'user_ended','reward':1,'trace_path':'fake'};store.save(row)
        return row
    strict_config=CONFIG|{'unknown_request_reserve_cny':0}
    rows=run_batch(jobs,strict_config,RunBudget(),tmp_path,worker=worker)
    assert sum(r['status']=='not_started' for r in rows)==2


def test_planned_running_finished_records_and_program_errors_visible(tmp_path):
    jobs=plan({'development':[{'domain':'retail','split':'dev','index':4}]},trials=2)
    def bug(*args): raise KeyError('programming error')
    with pytest.raises(KeyError): run_batch(jobs,CONFIG,RunBudget(),tmp_path,worker=bug)
    assert len(RunStore(tmp_path/'runs.sqlite').all())==2


def test_resume_preserves_completed_errors_and_only_starts_unstarted(tmp_path):
    jobs=plan({'development':[{'domain':'retail','split':'dev','index':4}]},trials=2)
    store=RunStore(tmp_path/'runs.sqlite')
    first=jobs[0]|{'status':'model_error','reward':None,'trace_path':'old','usage':{'estimated_cost_cny':0}}
    store.save(first)
    def worker(job,config,budget,batch,folder,store):
        row=job|{'status':'user_ended','reward':1,'trace_path':'new'};store.save(row);return row
    rows=run_batch(jobs,CONFIG,RunBudget(),tmp_path,worker=worker,resume=True)
    assert rows[0]==first and rows[1]['status']=='user_ended'


def test_unknown_fee_reserve_allows_retry_budget_without_fabricating_fee():
    batch=BatchBudget(2.5,unknown_reserve=1)
    batch.record('model_call',{'usage':None});batch.check()
    batch.record('model_call',{'usage':{'estimated_cost_cny':.2}})
    assert batch.spent==.2 and batch.reserved==1 and batch.unknown_calls==1
    batch.record('model_call',{'usage':None});batch.check()
    batch.record('model_call',{'usage':None})
    from runtime.model_client import BudgetExceeded
    with pytest.raises(BudgetExceeded):batch.check()


def test_account_quota_pauses_before_next_task(tmp_path):
    jobs=plan({'development':[{'domain':'retail','split':'dev','index':4}]},trials=3)
    def worker(job,config,budget,batch,folder,store):
        batch.record('model_call',{'usage':None,'api_error_code':'AccountQuotaExceeded',
                                  'quota_reset_at':'2026-11-01T00:51:01+08:00'})
        row=job|{'status':'model_error','reward':None,'trace_path':'fake'};store.save(row);return row
    rows=run_batch(jobs,CONFIG,RunBudget(),tmp_path,worker=worker)
    assert sum(r['status']=='not_started' for r in rows)==2
    assert (tmp_path/'quota_pause.json').exists()
