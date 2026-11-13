"""Offline structural checks; semantic compliance remains a review task."""

import json
from pathlib import Path
from scripts.numeric_facts import scan_numeric

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'docs/results/m1_zero_20261030_145330'
AUTH = {'find_user_id_by_email', 'find_user_id_by_name_zip'}
PERSONAL = {'get_user_details', 'get_order_details', 'modify_user_address',
    'cancel_pending_order', 'modify_pending_order_address', 'modify_pending_order_items',
    'modify_pending_order_payment', 'return_delivered_order_items', 'exchange_delivered_order_items'}


def scan_trace(messages, raw_replies, valid_user_ids):
    authenticated, pending = False, {}
    auth_events, preauth, personal = [], [], []
    for index, message in enumerate(messages, 1):
        for call in message.get('tool_calls') or []:
            name = call['function']['name']
            pending[call['id']] = name
            if name in PERSONAL:
                personal.append({'message_index': index, 'tool': name})
                if not authenticated:
                    preauth.append({'message_index': index, 'tool': name})
        if message['role'] == 'tool' and pending.get(message.get('tool_call_id')) in AUTH:
            if message['content'].strip() in valid_user_ids:
                authenticated = True
                auth_events.append({'message_index': index, 'user_id': message['content'].strip()})
    positions = [i for i, message in enumerate(messages, 1) if message['role'] == 'assistant']
    mixed, multiple = [], []
    for position, message in zip(positions, raw_replies, strict=True):
        calls = message.get('tool_calls') or []
        if calls and (message.get('content') or '').strip():
            mixed.append({'message_index': position, 'text': message['content'],
                          'tool_names': [c['function']['name'] for c in calls]})
        if len(calls) > 1:
            multiple.append({'message_index': position, 'raw_tool_count': len(calls),
                             'tool_names': [c['function']['name'] for c in calls],
                             'saved_tool_count': len(messages[position-1].get('tool_calls') or [])})
    return {'auth_events': auth_events, 'personal_data_calls': personal,
            'preauth_data_access': preauth, 'no_personal_data_access': not personal,
            'no_successful_auth': not authenticated, 'text_and_tool': mixed,
            'multiple_tool_calls': multiple}


def main():
    users = json.loads((ROOT / 'vendor/tau-bench/tau_bench/envs/retail/data/users.json').read_text())
    results = []
    for run in json.loads((BATCH / 'runs.json').read_text()):
        folder = BATCH / run['run_id']
        messages = json.loads((folder / 'messages.json').read_text())
        calls = [json.loads(line) for line in (folder / 'calls.jsonl').read_text().splitlines()]
        replies = [r['response']['choices'][0]['message'] for r in calls if r['actor'] == 'agent']
        results.append({'run_id': run['run_id'], 'task': run['task'], 'reward': run['reward'],
                        **scan_trace(messages, replies, set(users)), 'numeric_facts': scan_numeric(messages)})
    totals = {'runs': len(results), 'successes': sum(r['reward'] == 1 for r in results)}
    for field in ['preauth_data_access', 'text_and_tool', 'multiple_tool_calls']:
        totals[field] = {'runs': sum(bool(r[field]) for r in results),
                         'events': sum(len(r[field]) for r in results),
                         'successful_runs': sum(bool(r[field]) and r['reward'] == 1 for r in results)}
    totals['no_personal_data_access_runs'] = sum(r['no_personal_data_access'] for r in results)
    totals['no_successful_auth_runs'] = [r['run_id'] for r in results if r['no_successful_auth']]
    numeric = [e for r in results for e in r['numeric_facts']]
    totals['numeric'] = {v: sum(e['verdict'] == v for e in numeric)
                         for v in ('source_matched', 'violation', 'unverified')}
    totals['numeric']['violation_runs'] = sum(any(e['verdict'] == 'violation' for e in r['numeric_facts']) for r in results)
    totals['numeric']['violation_tasks'] = len({r['task'] for r in results if any(e['verdict'] == 'violation' for e in r['numeric_facts'])})
    totals['numeric']['visible_violation_runs'] = sum(any(e['verdict'] == 'violation' and e['user_visible'] for e in r['numeric_facts']) for r in results)
    (BATCH / 'compliance_automatic.json').write_text(json.dumps(
        {'scope': 'all 120 runs; source model replies for structure; auth is a proxy',
         'totals': totals, 'runs': results}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
