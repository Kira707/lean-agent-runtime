from concurrent.futures import ThreadPoolExecutor
from threading import Event
import json
import importlib
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

module=importlib.import_module('service.app')
BODY={'task_ref':{'domain':'retail','split':'dev','index':4},'variant':'payment_guard'}


@pytest.fixture
def api(tmp_path,monkeypatch):
    calls=[];finished=Event()
    monkeypatch.setattr(module,'RESULTS',tmp_path)
    monkeypatch.setenv('ARK_AGENT_API_KEY','fake');monkeypatch.setenv('ARK_USER_API_KEY','fake')
    def worker(job):
        calls.append(job);path=tmp_path/job['run_id']/'trace.jsonl';path.parent.mkdir()
        path.write_text(json.dumps({'event':'run_end','run_id':job['run_id']})+'\n')
        module.app.state.store.save(job|{'status':'user_ended','reward':1,'trace_path':str(path)})
        finished.set()
    monkeypatch.setattr(module,'run_service_job',worker)
    with TestClient(module.app) as client:yield client,calls,finished


def test_submit_then_read_result_and_trace(api):
    client,calls,finished=api
    response=client.post('/runs',json=BODY)
    assert response.status_code==202 and finished.wait(2)
    rid=response.json()['run_id']
    assert client.get('/runs/'+rid).json()['reward']==1
    assert client.get('/runs/'+rid+'/trace').json()==[{'event':'run_end','run_id':rid}]


def test_reject_unapproved_task_scope_and_unknown_run(api):
    client,calls,_=api
    for task in ({'domain':'retail','split':'test','index':4},
                 {'domain':'airline','split':'dev','index':4},
                 {'domain':'retail','split':'train','index':0}):
        assert client.post('/runs',json=BODY|{'task_ref':task}).status_code==422
    assert client.get('/runs/missing').status_code==404
    assert client.get('/runs/missing/trace').status_code==404 and calls==[]


def test_idempotency_returns_same_run_and_conflict_on_changed_request(api):
    client,calls,finished=api;headers={'Idempotency-Key':'same'}
    first=client.post('/runs',json=BODY,headers=headers);assert finished.wait(2)
    again=client.post('/runs',json=BODY,headers=headers)
    assert first.json()['run_id']==again.json()['run_id'] and len(calls)==1
    assert client.post('/runs',json=BODY|{'variant':'baseline'},headers=headers).status_code==409


def test_concurrent_same_key_schedules_exactly_one_run(api):
    client,calls,finished=api
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(client.post,'/runs',json=BODY,headers={'Idempotency-Key':'concurrent'}) for _ in range(2)]
        responses=[f.result() for f in futures]
    assert all(r.status_code==202 for r in responses) and finished.wait(2)
    assert len({r.json()['run_id'] for r in responses})==1 and len(calls)==1
