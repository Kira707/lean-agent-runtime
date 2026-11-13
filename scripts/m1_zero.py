"""M1-0: three serial development rounds through the unchanged reference agent."""

import argparse
import importlib.metadata
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from math import comb
from pathlib import Path
from unittest.mock import patch

from m0.protocol import git, load_config, task_key, validate_manifest, verify_base
from m0.recording import CallRecorder

ROOT = Path(__file__).resolve().parents[1]
WRITE_TOOLS = {'cancel_pending_order', 'exchange_delivered_order_items',
    'return_delivered_order_items', 'modify_pending_order_address',
    'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address'}


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def plan_runs(manifest):
    validate_manifest(manifest)
    if len(manifest['development']) != 40:
        raise ValueError('M1-0 requires the frozen forty-task development pool')
    return [{'run_id': f"r{r:02d}_{t['domain']}_{t['split']}_{t['index']:03d}",
             'round': r, 'task': task_key(t), 'task_ref': t}
            for r in (1, 2, 3) for t in manifest['development']]


def scores(rows, expected=120):
    groups = defaultdict(list)
    for row in rows:
        groups[row['task']].append(row['reward'] == 1)
    complete = len(rows) == expected and all(len(g) == 3 for g in groups.values())
    result = {'complete': complete, 'attempted': len(rows), 'expected': expected,
              'pass1': None, 'pass2': None, 'pass3': None, 'pass1_minus_pass3': None}
    if complete:
        for k in (1, 2, 3):
            result[f'pass{k}'] = sum(comb(sum(g), k) / comb(3, k) for g in groups.values()) / len(groups)
        result['pass1_minus_pass3'] = result['pass1'] - result['pass3']
    result['per_task_successes'] = {t: sum(g) for t, g in groups.items()}
    return result


def saturation(labels, audits, policy):
    repair_tasks = sorted({r['task'] for r in labels if r['category'] in policy['repair_categories']})
    if any(r['category'] is None for r in labels):
        rule = 'pending_failure_labels'
    elif len(repair_tasks) >= policy['minimum_repair_tasks']:
        rule = 'continue_retail_success_rate'
    elif len(audits) != policy['audit_count'] or any(r['policy_violation'] is None for r in audits):
        rule = 'pending_success_audit'
    elif sum(r['policy_violation'] for r in audits) >= policy['minimum_policy_violations']:
        rule = 'compliance_direction'
    else:
        rule = 'consider_airline'
    return {'rule': rule, 'repair_task_count': len(repair_tasks), 'repair_tasks': repair_tasks,
            'final_direction': 'User decides after reviewing the evidence'}


def summarize(folder):
    rows = json.loads((folder / 'runs.json').read_text(encoding='utf-8'))
    policy = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    result = scores(rows)
    labels_path, audit_path = folder / 'failure_labels.json', folder / 'success_audit.json'
    if not labels_path.exists():
        dump(labels_path, [{k: r[k] for k in ('run_id', 'task', 'round', 'trace')} |
                          {'category': None, 'evidence': ''} for r in rows if r['reward'] != 1])
    if not audit_path.exists():
        eligible = sorted([r for r in rows if r['reward'] == 1 and r['has_write']], key=lambda r: r['run_id'])
        selected = random.Random(policy['success_audit_seed']).sample(eligible, min(policy['audit_count'], len(eligible)))
        dump(audit_path, [{k: r[k] for k in ('run_id', 'task', 'round', 'trace')} |
                         {'policy_violation': None, 'evidence': ''} for r in selected])
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    audits = json.loads(audit_path.read_text(encoding='utf-8'))
    if any(r['category'] is not None and (type(r['category']) is not int or not 1 <= r['category'] <= 10) for r in labels):
        raise ValueError('Failure category must be null or an integer from 1 to 10')
    if any(r['policy_violation'] is not None and type(r['policy_violation']) is not bool for r in audits):
        raise ValueError('Audit policy_violation must be null, true or false')
    result['saturation'] = saturation(labels, audits, policy) if result['complete'] else {'rule': 'incomplete_batch'}
    result['estimated_cost_cny'] = (sum(r['estimated_cost_cny'] for r in rows)
        if all(r['estimated_cost_cny'] is not None for r in rows) else None)
    dump(folder / 'summary.json', result)
    text = ['# M1-0参考Agent开发池结果', '', f"运行：{len(rows)}/120；完整：{result['complete']}。",
            '预测pass¹≈90%、pass³≈80%为运行前用户直觉，不是结论。', '',
            f"pass¹：{result['pass1']}；pass²：{result['pass2']}；pass³：{result['pass3']}。",
            f"pass¹−pass³：{result['pass1_minus_pass3']}（仅辅助信息）。",
            f"折算费用：{result['estimated_cost_cny']}元，非Coding Plan账单。", '',
            '失败主因请填写failure_labels.json；成功合规抽样请填写success_audit.json。',
            f"规则状态：{result['saturation']['rule']}；最终方向由用户决定。", '',
            '| 任务 | 三次成功数 |', '|---|---:|']
    text += [f'| {task} | {count}/3 |' for task, count in result['per_task_successes'].items()]
    (folder / 'summary.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    return result


def has_successful_write(messages):
    writes = {c['id'] for m in messages for c in m.get('tool_calls') or []
              if c['function']['name'] in WRITE_TOOLS}
    return any(m['role'] == 'tool' and m.get('tool_call_id') in writes and
               not m['content'].startswith(('Error', 'Unknown action')) for m in messages)


def run(live=False):
    if not live:
        raise ValueError('M1-0 paid run requires --live')
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    policy = json.loads((ROOT / 'configs/m1_zero.json').read_text())
    config = load_config(ROOT / policy['call_config'])
    if policy['rounds'] != 3 or policy['concurrency'] != 1:
        raise ValueError('M1-0 requires three sequential rounds, concurrency=1')
    config['cost_cap_cny'] = policy['cost_cap_cny']
    for actor in ('agent', 'user'):
        if not os.environ.get(config[actor]['api_key_env']):
            raise ValueError(f"Missing {config[actor]['api_key_env']}")
    fingerprint = verify_base(ROOT / 'vendor/tau-bench', ROOT / 'patches/tau_user_simulator.patch')
    jobs = plan_runs(json.loads((ROOT / 'configs/task_manifest.json').read_text()))
    import litellm
    import tau_bench.agents.tool_calling_agent as agent_module
    import tau_bench.envs.user as user_module
    from tau_bench.envs import get_env
    from pydantic import ValidationError
    folder = ROOT / 'docs/results' / datetime.now().strftime('m1_zero_%Y%m%d_%H%M%S')
    folder.mkdir()
    dump(folder / 'protocol.json', policy | {'call_config': config, 'jobs': jobs})
    dump(folder / 'environment.json', fingerprint | {'implementation_commit': git(ROOT, 'rev-parse', 'HEAD').strip(),
         'litellm_version': importlib.metadata.version('litellm')})
    rows, spent, recorder = [], 0, None
    agent_complete, user_complete = agent_module.completion, user_module.completion
    with patch.object(agent_module, 'completion', lambda **kw: recorder.call(agent_complete, 'agent', **kw)), \
         patch.object(user_module, 'completion', lambda **kw: recorder.call(user_complete, 'user', **kw)):
        for job in jobs:
            if spent >= float(config['cost_cap_cny']):
                break
            path = folder / job['run_id']
            path.mkdir()
            recorder = CallRecorder(path / 'calls.jsonl', config)
            recorder.spent_upper = spent
            recorder.task, recorder.phase = job['task'], 'initialization'
            row = {k: job[k] for k in ('run_id', 'task', 'round')}
            row.update(status='running', reward=None, trace=f"{job['run_id']}/messages.json", has_write=False)
            print(f"[{len(rows)+1}/120] round={job['round']} {job['task']}", flush=True)
            stop = False
            try:
                t = job['task_ref']
                env = get_env('retail', 'llm', config['user']['model'], t['split'],
                    config['user']['provider'], t['index'], user_temperature=config['user']['temperature'])
                recorder.phase = 'task'
                agent = agent_module.ToolCallingAgent(env.tools_info, env.wiki, config['agent']['model'],
                    config['agent']['provider'], config['agent']['temperature'])
                result = agent.solve(env, t['index'], max_num_steps=config['max_steps'])
                dump(path / 'messages.json', result.messages)
                dump(path / 'evaluation.json', result.info)
                row.update(status='finished', reward=result.reward, has_write=has_successful_write(result.messages))
            except (litellm.APIError, json.JSONDecodeError, ValidationError, KeyError, TypeError, RuntimeError) as error:
                recorder.record_failure(error)
                row.update(status='error', error_type=type(error).__name__, error=str(error),
                           trace=f"{job['run_id']}/calls.jsonl")
                stop = isinstance(error, (litellm.APIError, RuntimeError))
            spent = recorder.spent_upper
            costs = [r['estimated_cost_cny'] for r in recorder.records]
            row['estimated_cost_cny'] = sum(costs) if all(c is not None for c in costs) else None
            row['call_count'] = len(recorder.records)
            dump(path / 'run.json', row)
            rows.append(row)
            dump(folder / 'runs.json', rows)
            dump(folder / 'progress.json', {'completed_attempts': len(rows), 'expected': 120,
                 'last': job['run_id'], 'spent_uncached_upper_cny': float(spent)})
            print(f"  {row['status']} reward={row['reward']} cost={row['estimated_cost_cny']}", flush=True)
            if stop:
                break
    summarize(folder)
    print(f'Results: {folder}', flush=True)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'summarize'])
    parser.add_argument('folder', nargs='?')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    run(args.live) if args.command == 'run' else summarize(Path(args.folder))
