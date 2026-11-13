"""Explicit development baseline entry point; test and airline have no route."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from m0.protocol import git, verify_base, validate_manifest
from runtime.config import load_config
from eval.runner import dump, plan, run_batch

ROOT=Path(__file__).resolve().parents[1]


def main(live=False, smoke=False, resume=None, config_path='configs/m1_baseline.json', tasks=None):
    if not live: raise ValueError('Paid development run requires --live')
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    config,budget=load_config(ROOT/config_path)
    for actor in ('agent','user'):
        if not os.environ.get(config[actor]['api_key_env']): raise ValueError(f'Missing {actor} credential')
    base=verify_base(ROOT/'vendor/tau-bench',ROOT/'patches/tau_user_simulator.patch')
    manifest=json.loads((ROOT/'configs/task_manifest.json').read_text()); validate_manifest(manifest)
    jobs=plan(manifest,variant=config['variant'])
    if smoke: jobs=[next(j for j in jobs if j['task']=='retail/dev/4')]
    if tasks:
        if smoke or not set(tasks)<={j['task'] for j in jobs}:
            raise ValueError('Targeted debugging only accepts tasks from the fixed development pool')
        jobs=[j for j in jobs if j['task'] in set(tasks)]
    if resume:
        folder=Path(resume)
        pause=folder/'quota_pause.json'
        if pause.exists():
            reset_at=json.loads(pause.read_text()).get('quota_reset_at')
            if reset_at and datetime.now(timezone.utc)<datetime.fromisoformat(reset_at)+timedelta(seconds=5):
                raise ValueError(f'Account quota has not reset: {reset_at}; wait five extra seconds')
        protocol=json.loads((folder/'protocol.json').read_text())
        saved=protocol['jobs']
        if [(j['run_id'],j['task_ref']) for j in saved] != [(j['run_id'],j['task_ref']) for j in jobs]:
            raise ValueError('Resume jobs differ from the fixed development plan')
        changed={k for k in set(protocol['config'])|set(config) if protocol['config'].get(k)!=config.get(k)}
        allowed={'max_retries','retry_wait_seconds','min_request_interval_seconds','unknown_request_reserve_cny'}
        if not changed <= allowed: raise ValueError('Resume non-transport config differs')
        if changed:
            revision={'authorization':'User: pace online calls; wait 60 seconds before transient-error retry',
                'changes':{k:{'before':protocol['config'].get(k),'after':config.get(k)} for k in sorted(changed)},
                'implementation_commit':git(ROOT,'rev-parse','HEAD').strip()}
            protocol.setdefault('execution_revisions',[]).append(revision)
            protocol['active_config']=config
            revision_hash=hashlib.sha256(json.dumps(revision,sort_keys=True).encode()).hexdigest()
            from runtime.store import RunStore
            store=RunStore(folder/'runs.sqlite')
            for job in saved:
                if store.get(job['run_id'])['status']=='not_started':
                    job['config_hash']=revision_hash
                    job['execution_revision']=len(protocol['execution_revisions'])
        commits=protocol.setdefault('resume_commits',[])
        commits.append(git(ROOT,'rev-parse','HEAD').strip())
        dump(folder/'protocol.json',protocol)
        print('Resuming:',folder,flush=True)
        run_batch(saved,config,budget,folder,resume=True)
        return folder
    prefix='m1_smoke' if smoke else ('m1_targeted_' if tasks else 'm1_')+config['variant']
    folder=ROOT/'docs/results'/datetime.now().strftime(prefix+'_%Y%m%d_%H%M%S')
    folder.mkdir()
    policy=(ROOT/'vendor/tau-bench/tau_bench/envs/retail/wiki.md').read_bytes()
    frozen={'config':config,'jobs':jobs,'base':base,'implementation_commit':git(ROOT,'rev-parse','HEAD').strip(),
            'policy_sha256':hashlib.sha256(policy).hexdigest()}
    frozen['config_hash']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    for job in jobs: job['config_hash']=frozen['config_hash']
    dump(folder/'protocol.json',frozen)
    print('Results:',folder,flush=True)
    run_batch(jobs,config,budget,folder)
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume')
    parser.add_argument('--config',default='configs/m1_baseline.json')
    parser.add_argument('--tasks',nargs='+',help='Explicit fixed-pool task names for targeted debugging')
    args=parser.parse_args();main(args.live,args.smoke,args.resume,args.config,args.tasks)
