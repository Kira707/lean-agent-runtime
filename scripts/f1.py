"""Formal paired execution entry. Preparing a freeze never loads test tasks."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from eval.freeze import make_freeze, validate_freeze
from eval.runner import run_batch, execute_job, dump
from runtime.types import RunBudget

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/'configs/f1_freeze.json'


def prepare(freeze_path=None):
    target=Path(freeze_path) if freeze_path else FREEZE
    if target.exists():raise ValueError('Freeze already exists; inspect it instead of overwriting')
    dump(target,make_freeze(ROOT));print(target)


def execute(*,live=False,resume=None,freeze_path=None):
    if not live:raise ValueError('F1 requires --live and approved freeze')
    target=Path(freeze_path) if freeze_path else FREEZE
    frozen=json.loads(target.read_text());validate_freeze(ROOT,frozen)
    load_dotenv(ROOT/'.env');os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
    configs=frozen['core']['configs'];baseline=configs['baseline']
    for actor in ('agent','user'):
        if not os.environ.get(baseline[actor]['api_key_env']):raise ValueError(f'Missing {actor} credential')
    folder=Path(resume).resolve() if resume else ROOT/'docs/results'/datetime.now().strftime('f1_%Y%m%d_%H%M%S')
    folder.mkdir(parents=True,exist_ok=True)
    if resume:
        previous=json.loads((folder/'protocol.json').read_text())
        if previous['protocol_hash']!=frozen['protocol_hash']:raise ValueError('Resume protocol differs')
        pause=folder/'quota_pause.json'
        if pause.exists():
            from datetime import timedelta,timezone
            reset=json.loads(pause.read_text()).get('quota_reset_at')
            if reset and datetime.now(timezone.utc)<datetime.fromisoformat(reset)+timedelta(seconds=5):
                raise ValueError(f'Wait for quota reset: {reset}')
    else:dump(folder/'protocol.json',frozen)
    def worker(job,common,budget,batch,result_dir,store):
        config=configs[job['variant']]
        return execute_job(job,config,RunBudget(**config['budget']),batch,result_dir,store)
    print('F1 results:',folder,flush=True)
    rows=run_batch(frozen['jobs'],baseline,RunBudget(**baseline['budget']),folder,worker=worker,resume=bool(resume))
    from eval.f1_report import report_f1
    report_f1(folder)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--live',action='store_true');parser.add_argument('--resume')
    parser.add_argument('--freeze',help='New multi-H snapshot path; original H1 snapshot is retained')
    args=parser.parse_args()
    prepare(args.freeze) if args.command=='prepare' else execute(live=args.live,resume=args.resume,freeze_path=args.freeze)
