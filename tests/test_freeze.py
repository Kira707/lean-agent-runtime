from copy import deepcopy
import pytest
from eval.freeze import paired_plan, compare_transport, validate_freeze, digest, F1_VARIANTS
from pathlib import Path
import json


def test_unapproved_freeze_and_different_transport_are_rejected(monkeypatch,tmp_path):
    import eval.freeze as module
    import scripts.f1 as entry
    monkeypatch.setattr(entry,'FREEZE',tmp_path/'missing.json')
    with pytest.raises(FileNotFoundError):entry.execute(live=True)
    alternative=tmp_path/'multi.json';alternative.write_text('{"approved":false}')
    with pytest.raises(ValueError,match='approval'):
        entry.execute(live=True,freeze_path=alternative)
    with pytest.raises(ValueError,match='approval'):validate_freeze(Path('.'),{'approved':False})
    baseline=json.loads(Path('configs/m1_baseline.json').read_text())
    candidate=json.loads(Path('configs/m1_payment_guard.json').read_text())
    compare_transport({'baseline':baseline,'payment_guard':candidate})
    candidate=deepcopy(candidate);candidate['retry_wait_seconds']=1
    with pytest.raises(ValueError,match='retry_wait_seconds'):
        compare_transport({'baseline':baseline,'payment_guard':candidate})
    candidate=json.loads(Path('configs/m1_payment_guard.json').read_text())
    refs=[{'domain':'retail','split':'test','index':i} for i in range(50)]
    core={'configs':{'baseline':baseline,'payment_guard':candidate,
        'all_guards':json.loads(Path('configs/m1_all_guards.json').read_text())},'tasks':refs,'implementation_commit':'tested'}
    frozen={'approved':True,'core':core,'protocol_hash':digest(core),'jobs':paired_plan(refs,digest(core))}
    monkeypatch.setattr(module,'git',lambda *args:'changed source')
    with pytest.raises(ValueError,match='Implementation/config changed'):validate_freeze(Path('.'),frozen)
    monkeypatch.setattr(module,'git',lambda *args:'')
    core['configs']=deepcopy(core['configs'])
    for config in core['configs'].values():config['request_timeout_seconds']=61
    frozen['protocol_hash']=digest(core);frozen['jobs']=paired_plan(refs,digest(core))
    with pytest.raises(ValueError,match='Frozen configs differ'):validate_freeze(Path('.'),frozen)


def test_formal_plan_is_fixed_450_unique_and_rotates_group_order():
    refs=[{'domain':'retail','split':'test','index':i} for i in range(50)]
    jobs=paired_plan(refs,'frozen');assert len(jobs)==len({r['run_id'] for r in jobs})==450
    assert jobs==paired_plan(refs,'frozen')
    for ref in refs:
        for variant in F1_VARIANTS:
            subset=[j for j in jobs if j['task_ref']==ref and j['variant']==variant]
            assert {j['trial'] for j in subset}=={1,2,3}
    assert jobs[0]['variant']!=jobs[3]['variant']


def test_three_groups_have_equal_transport_and_exclude_standalone_h2():
    refs=[{'domain':'retail','split':'test','index':i} for i in range(50)]
    variants=F1_VARIANTS
    configs={v:json.loads(Path(f'configs/m1_{v}.json').read_text()) for v in variants}
    compare_transport(configs)
    jobs=paired_plan(refs,'pending',variants)
    assert len(jobs)==len({j['run_id'] for j in jobs})==450
    assert {j['variant'] for j in jobs}=={'baseline','payment_guard','all_guards'}
    for ref in refs:
        for variant in variants:
            assert {j['trial'] for j in jobs if j['task_ref']==ref and j['variant']==variant}=={1,2,3}
    old_four=deepcopy(configs)
    old_four['fact_guard']=json.loads(Path('configs/m1_fact_guard.json').read_text())
    with pytest.raises(ValueError,match='no standalone H2'):
        validate_freeze(Path('.'),{'approved':True,'core':{'configs':old_four}})
    configs['all_guards']['min_request_interval_seconds']=0
    with pytest.raises(ValueError,match='all_guards'):
        compare_transport(configs)


def test_approved_f1_cap_15_keeps_all_other_frozen_settings(monkeypatch):
    import eval.freeze as module
    frozen=json.loads(Path('configs/f1_multi_freeze.json').read_text())
    frozen['approved']=True
    for config in frozen['core']['configs'].values():config['cost_cap_cny']='15'
    def fingerprint():
        frozen['protocol_hash']=digest(frozen['core'])
        frozen['jobs']=paired_plan(frozen['core']['tasks'],frozen['protocol_hash'],tuple(frozen['core']['configs']))
    monkeypatch.setattr(module,'git',lambda *args:'')
    fingerprint();validate_freeze(Path('.'),frozen)
    for config in frozen['core']['configs'].values():config['cost_cap_cny']='16'
    fingerprint()
    with pytest.raises(ValueError,match='<=15'):validate_freeze(Path('.'),frozen)
    for config in frozen['core']['configs'].values():
        config['cost_cap_cny']='15';config['agent']['temperature']=0.1
    fingerprint()
    with pytest.raises(ValueError,match='Frozen configs differ'):validate_freeze(Path('.'),frozen)
