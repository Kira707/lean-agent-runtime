"""Local development API: fixed development tasks, one model worker."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Literal
from unittest.mock import patch
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from eval.runner import execute_job, BatchBudget, simulator_completion
from runtime.config import load_config
from runtime.model_client import RequestPacer
from runtime.store import RunStore

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'outputs/service'


@asynccontextmanager
async def lifespan(application):
    # Resource lifetime pattern: https://fastapi.tiangolo.com/advanced/events/
    import tau_bench.envs.user as user_module
    load_dotenv(ROOT/'.env')
    application.state.configs={v:load_config(ROOT/f'configs/m1_{v}.json') for v in
                               ('baseline','payment_guard','fact_guard','all_guards')}
    config,_=application.state.configs['baseline']
    for role in ('agent','user'):
        if not os.environ.get(config[role]['api_key_env']):raise ValueError(f'Missing {role} API credential')
    application.state.store=RunStore(RESULTS/'runs.sqlite')
    application.state.batch=BatchBudget(config['cost_cap_cny'],config['unknown_request_reserve_cny'])
    application.state.batch.pacer=RequestPacer(config['min_request_interval_seconds'])
    application.state.executor=ThreadPoolExecutor(max_workers=1)
    application.state.futures={}
    manifest=json.loads((ROOT/'configs/task_manifest.json').read_text())
    application.state.allowed={(t['domain'],t['split'],t['index']) for t in manifest['development']}
    with patch.object(user_module,'completion',simulator_completion):
        try:yield
        finally:application.state.executor.shutdown(wait=True)


app=FastAPI(title='Lean Agent Runtime',lifespan=lifespan)


class TaskRef(BaseModel):
    model_config=ConfigDict(extra='forbid')
    domain: Literal['retail']='retail'
    split: Literal['dev','train']
    index: int=Field(ge=0)


class RunRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    task_ref: TaskRef
    variant: Literal['baseline','payment_guard','fact_guard','all_guards']='payment_guard'


def run_service_job(job):
    config,budget=app.state.configs[job['variant']]
    return execute_job(job,config,budget,app.state.batch,RESULTS,app.state.store)


@app.post('/runs',status_code=202)
def create_run(request:RunRequest,idempotency_key:str|None=Header(default=None)):
    t=request.task_ref
    if (t.domain,t.split,t.index) not in app.state.allowed:
        raise HTTPException(422,'Task must belong to the frozen development pool')
    run_id=uuid4().hex
    job={'run_id':run_id,'variant':request.variant,'trial':1,'task_ref':t.model_dump(),
         'task':f'{t.domain}/{t.split}/{t.index}'}
    row=job|{'status':'queued','reward':None,'trace_path':str(RESULTS/run_id/'trace.jsonl')}
    actual,created,conflict=app.state.store.reserve(idempotency_key or uuid4().hex,request.model_dump(),row)
    if conflict:raise HTTPException(409,'Idempotency-Key already belongs to a different request')
    if created:app.state.futures[actual]=app.state.executor.submit(run_service_job,job)
    return {'run_id':actual,'status':app.state.store.get(actual)['status']}


@app.get('/runs/{run_id}')
def get_run(run_id:str):
    row=app.state.store.get(run_id)
    if row is None:raise HTTPException(404,'Unknown run')
    future=app.state.futures.get(run_id)
    if future is not None and future.done() and future.exception() is not None:
        error=future.exception()
        raise HTTPException(500,f'Worker failed: {type(error).__name__}: {error}')
    return row


@app.get('/runs/{run_id}/trace')
def get_trace(run_id:str):
    row=get_run(run_id);path=Path(row['trace_path'])
    if row['status']=='queued' and not path.exists():return []
    # The worker may be appending a large event; expose complete JSONL lines only.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines(keepends=True)
            if line.endswith('\n')]
