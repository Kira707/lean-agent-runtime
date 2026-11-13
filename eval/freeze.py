"""Freeze preparation reads manifest IDs only; execution requires approval."""
import hashlib
import json
from pathlib import Path
from m0.protocol import git, verify_base, validate_manifest
from runtime.config import load_config

TRANSPORT=('agent','user','user_strategy','budget','concurrency','request_timeout_seconds',
           'max_output_tokens','max_retries','retry_wait_seconds','min_request_interval_seconds',
           'unknown_request_reserve_cny','cost_cap_cny','prices')
CODE_PATHS=('runtime','domains','eval','scripts','service','m0','patches','pyproject.toml',
            'requirements.lock.txt','configs/m1_baseline.json','configs/m1_payment_guard.json',
            'configs/m1_fact_guard.json','configs/m1_all_guards.json',
            'configs/user_simulator.lock.json','configs/task_manifest.json')
VARIANT_CONFIGS={v:f'configs/m1_{v}.json' for v in
                 ('baseline','payment_guard','fact_guard','all_guards')}
F1_VARIANTS=('baseline','payment_guard','all_guards')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def paired_plan(tasks, protocol_hash='', variants=F1_VARIANTS):
    if len(tasks)!=50 or len({t['index'] for t in tasks})!=50 or any(
            t['domain']!='retail' or t['split']!='test' for t in tasks):
        raise ValueError('F1 requires the fifty fixed retail test references')
    jobs=[]
    for trial in (1,2,3):
        for position,t in enumerate(tasks):
            offset=(trial+position)%len(variants)
            order=variants[offset:]+variants[:offset]
            for variant in order:
                jobs.append({'run_id':f"{variant}_r{trial:02d}_test_{t['index']:03d}",
                    'variant':variant,'trial':trial,'task_ref':t,'task':f"retail/test/{t['index']}",
                    'config_hash':digest({'protocol':protocol_hash,'variant':variant})})
    return jobs


def compare_transport(configs):
    baseline=configs['baseline']
    for variant,config in configs.items():
        different=[key for key in TRANSPORT if baseline.get(key)!=config.get(key)]
        if different:raise ValueError(f'Transport/retry settings differ in {variant}: {different}')


def make_freeze(root):
    root=Path(root)
    configs={v:load_config(root/VARIANT_CONFIGS[v])[0] for v in F1_VARIANTS}
    compare_transport(configs)
    manifest=json.loads((root/'configs/task_manifest.json').read_text());validate_manifest(manifest)
    code_commit=git(root,'rev-parse','HEAD').strip()
    if git(root,'diff',code_commit,'--',*CODE_PATHS).strip():
        raise ValueError('Commit tested source/config changes before freezing')
    core={'implementation_commit':code_commit,'configs':configs,'tasks':manifest['test'],
        'tau':verify_base(root/'vendor/tau-bench',root/'patches/tau_user_simulator.patch'),
        'policy_sha256':hashlib.sha256((root/'vendor/tau-bench/tau_bench/envs/retail/wiki.md').read_bytes()).hexdigest(),
        'audit':{'shared_extractor':True,'sample_size':10,'seed':20261031,'variant':'all_guards',
                 'method':'Read card-related H trajectories directly, without fact extractor; report with F1'},
        'account_quota_policy':'Pause after current failure; resume unstarted jobs after provider reset plus 5 seconds'}
    fingerprint=digest(core)
    return {'approved':False,'core':core,'protocol_hash':fingerprint,
            'jobs':paired_plan(core['tasks'],fingerprint,tuple(configs))}


def validate_freeze(root, frozen):
    root=Path(root)
    if frozen.get('approved') is not True:raise ValueError('F1 freeze requires explicit user approval')
    core=frozen['core'];compare_transport(core['configs'])
    batch_cap=core['configs']['baseline']['cost_cap_cny']
    if not 0<float(batch_cap)<=15:
        raise ValueError('Approved F1 batch equivalent cost cap must be <=15 CNY')
    if set(core['configs'])!=set(F1_VARIANTS):
        raise ValueError('F1 requires exactly Baseline, H1 and H1+H2; no standalone H2')
    if digest(core)!=frozen['protocol_hash'] or paired_plan(core['tasks'],frozen['protocol_hash'],tuple(core['configs']))!=frozen['jobs']:
        raise ValueError('Frozen inputs or job order changed')
    if git(root,'diff',core['implementation_commit'],'--',*CODE_PATHS).strip():
        raise ValueError('Implementation/config changed after freeze')
    current={v:load_config(root/VARIANT_CONFIGS[v])[0] for v in core['configs']}
    # F1 approval may override the batch cap; development defaults stay at 9.5.
    for config in current.values():config['cost_cap_cny']=batch_cap
    if current!=core['configs']:raise ValueError('Frozen configs differ from current configs')
    manifest=json.loads((root/'configs/task_manifest.json').read_text())
    if manifest['test']!=core['tasks']:raise ValueError('Frozen task references differ')
    if verify_base(root/'vendor/tau-bench',root/'patches/tau_user_simulator.patch')!=core['tau']:
        raise ValueError('Tau fingerprint differs')
    policy=(root/'vendor/tau-bench/tau_bench/envs/retail/wiki.md').read_bytes()
    if hashlib.sha256(policy).hexdigest()!=core['policy_sha256']:raise ValueError('Frozen policy differs')
