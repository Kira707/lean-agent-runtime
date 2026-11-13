"""Development-only runner. User completion binding is batch-wide/context-local."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Lock
import time
from unittest.mock import patch
import httpx

from runtime.agent import run
from runtime.model_client import OpenAIClient, BudgetExceeded, ModelOutputError, RequestPacer, AccountQuotaExceeded
from runtime.store import RunStore
from runtime.tracer import Tracer
from runtime.types import AgentState, RunStatus

USER_CLIENT = ContextVar('user_model_client')


def simulator_completion(**kwargs):
    from litellm import ModelResponse
    client = USER_CLIENT.get()
    reply = client.complete(kwargs['messages'], kwargs.get('tools') or [], timeout_s=60)
    response = ModelResponse(**reply.raw_response)
    response._hidden_params['response_cost'] = None  # Only our price table is used.
    return response


class BatchBudget:
    def __init__(self, cap, unknown_reserve=0):
        self.cap, self.spent, self.unknown = float(cap), 0.0, False
        self.lock = Lock()
        self.unknown_reserve, self.reserved, self.unknown_calls = unknown_reserve, 0.0, 0
        self.quota_exhausted, self.quota_reset_at = False, None

    def check(self):
        with self.lock:
            if self.quota_exhausted:
                ready=(self.quota_reset_at is not None and datetime.now(timezone.utc)>=
                       datetime.fromisoformat(self.quota_reset_at)+timedelta(seconds=5))
                if not ready:raise BudgetExceeded(f'Account quota unavailable until {self.quota_reset_at}')
                self.quota_exhausted, self.quota_reset_at = False, None
            if self.unknown or self.spent+self.reserved >= self.cap:
                raise BudgetExceeded('batch cost cap or unknown billed usage')

    def record(self, event, data):
        if event == 'model_call':
            if data.get('api_error_code')=='AccountQuotaExceeded':
                self.quota_exhausted, self.quota_reset_at = True, data.get('quota_reset_at')
            usage = data.get('usage')
            cost = usage.get('estimated_cost_cny') if usage else None
            with self.lock:
                if cost is None:
                    self.unknown_calls += 1
                    self.reserved += self.unknown_reserve
                    self.unknown = self.unknown_reserve == 0
                else:
                    self.spent += cost


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')


def plan(manifest, trials=3, variant='baseline'):
    return [{'run_id':f"{variant}_r{r:02d}_{t['split']}_{t['index']:03d}",
             'variant':variant,'trial':r,'task_ref':t,
             'task':f"retail/{t['split']}/{t['index']}"}
            for r in range(1,trials+1) for t in manifest['development']]


def execute_job(job, config, budget, batch, folder, store, make_env=None, make_client=OpenAIClient):
    from domains.tau.adapter import TauAdapter
    from tau_bench.envs import get_env
    state, started = AgentState(job['run_id']), time.monotonic()
    path = folder/job['run_id']; path.mkdir(exist_ok=True)
    trace = Tracer(path/'trace.jsonl',state.run_id)
    def emit(event, **data):
        trace(event, **data); batch.record(event,data)
        if event=='retry_wait':
            print(f"{state.run_id} {data['actor']} waits {data['seconds']}s before retry {data['next_attempt']}",flush=True)
    emit('run_start',job=job,config=config)
    clients = {actor:make_client(config[actor],config['prices'],state,budget,actor=actor,
        emit=emit,started=started,max_retries=config.get('max_retries',0),
        retry_wait_seconds=config.get('retry_wait_seconds',0),
        max_output_tokens=config['max_output_tokens'],guard=batch.check,
        pace=getattr(batch,'pacer',None).wait if hasattr(batch,'pacer') else lambda:None)
        for actor in ('agent','user')}
    clients['user'].phase = 'initialization'
    token = USER_CLIENT.set(clients['user'])
    adapter, row = None, job | {'status':'running','reward':None,'trace_path':str(trace.path)}
    store.save(row)
    try:
        task = job['task_ref']
        if make_env is None:
            env = get_env('retail','llm',config['user']['model'],task['split'],
                config['user']['provider'],task['index'],user_temperature=config['user']['temperature'])
        else:
            env = make_env(task)
        clients['user'].phase = 'task'
        adapter = TauAdapter(env,task['index'])
        from functools import partial
        from domains.tau.response_hooks import response_feedback
        harness = config.get('harness',{})
        payment, retail = harness.get('payment_fact_check',False), harness.get('retail_fact_check',False)
        result = run(clients['agent'],adapter,adapter,system=env.wiki,state=state,budget=budget,
                     emit=emit,started=started,request_timeout=config['request_timeout_seconds'],
                     before_respond=partial(response_feedback,payment=payment,retail=retail) if payment or retail else None,
                     max_text_retries=harness.get('max_text_retries',2))
        row.update(status=result.status.value,reason=result.reason,reward=adapter.final_reward())
    except AccountQuotaExceeded as error:
        row.update(status=RunStatus.MODEL_ERROR.value,reason=str(error),
                   error_type='AccountQuotaExceeded',quota_reset_at=error.reset_at)
    except BudgetExceeded as error:
        row.update(status=RunStatus.BUDGET_EXCEEDED.value,reason=str(error))
    except (httpx.HTTPError, ModelOutputError, json.JSONDecodeError) as error:
        row.update(status=RunStatus.MODEL_ERROR.value,reason=str(error),error_type=type(error).__name__)
    except (OSError, ConnectionError) as error:
        row.update(status=RunStatus.ENV_ERROR.value,reason=str(error),error_type=type(error).__name__)
    finally:
        if adapter is not None:
            adapter.close()
            dump(path/'evaluation.json',{'task':adapter.env.task.model_dump(),
                'actual':adapter.evidence,'reward_info':adapter.reward_info})
        USER_CLIENT.reset(token)
        for client in clients.values(): client.close()
    row.update(usage=asdict(state.usage_total),steps=state.step,model_calls=state.model_calls,
               budget_tokens=state.budget_tokens,
               harness=state.hook_state.get('payment'),
               wall_ms=round((time.monotonic()-started)*1000))
    dump(path/'messages.json',state.messages)
    emit('run_end',**row)
    store.save(row)
    dump(path/'run.json',row)
    return row


def run_batch(jobs, config, budget, folder, *, worker=execute_job, resume=False):
    import tau_bench.envs.user as user_module
    folder = Path(folder).resolve(); folder.mkdir(parents=True,exist_ok=True)
    store, batch = RunStore(folder/'runs.sqlite'), BatchBudget(config['cost_cap_cny'],config.get('unknown_request_reserve_cny',0))
    batch.pacer = RequestPacer(config.get('min_request_interval_seconds',0))
    for job in jobs:
        if not resume or store.get(job['run_id']) is None:
            store.save(job | {'status':'not_started','reward':None,
                             'trace_path':str(folder/job['run_id']/'trace.jsonl')})
    if resume:
        completed=[r for r in store.all() if r['status']!='not_started']
        if any(r['status']=='running' for r in completed):
            raise ValueError('Resolve interrupted running state before resume')
        for row in completed:
            trace=folder/row['run_id']/'trace.jsonl'
            if trace.exists():
                for line in trace.read_text(encoding='utf-8').splitlines():
                    event=json.loads(line)
                    if event['event']=='model_call':
                        if event.get('error_type')=='ConnectError' and event.get('usage') is None:
                            event=event|{'usage':{'estimated_cost_cny':0}}  # Prior confirmed pre-request failure.
                        batch.record('model_call',event)
            else:
                batch.record('model_call',{'usage':row['usage']})
        # Previous quota events remain in the cost ledger; they do not re-pause a
        # batch explicitly resumed after the service reset.
        batch.quota_exhausted, batch.quota_reset_at = False, None
    remaining=[j for j in jobs if store.get(j['run_id'])['status']=='not_started']
    pending, iterator, stopped = {}, iter(remaining), False
    with patch.object(user_module,'completion',simulator_completion), ThreadPoolExecutor(config['concurrency']) as pool:
        while True:
            while not stopped and len(pending)<config['concurrency']:
                if batch.quota_exhausted:
                    print(f'Account quota exhausted; batch paused until {batch.quota_reset_at}',flush=True)
                    stopped=True;break
                if batch.unknown or batch.spent+batch.reserved>=batch.cap:
                    stopped=True; break
                job=next(iterator,None)
                if job is None: break
                pending[pool.submit(worker,job,config,budget,batch,folder,store)] = job
            if not pending: break
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future); row=future.result()
                print(f"{job['run_id']} {row['status']} reward={row['reward']} known_cny={batch.spent:.4f} reserve_cny={batch.reserved:.1f}",flush=True)
            dump(folder/'runs.json',store.all())
            dump(folder/'budget.json',{'known_equivalent_cny':batch.spent,
                'unknown_request_count':batch.unknown_calls,'scheduling_reserve_cny':batch.reserved,
                'scheduling_cap_cny':batch.cap,'reserve_is_not_measured_cost':True})
            if batch.quota_exhausted:
                dump(folder/'quota_pause.json',{'quota_reset_at':batch.quota_reset_at,
                    'reason':'AccountQuotaExceeded','resume_only_unstarted':True})
    dump(folder/'runs.json',store.all())
    return store.all()
