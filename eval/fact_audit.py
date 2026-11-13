"""Offline H2 coverage, with withheld text kept separate from delivered text."""
from domains.tau.fact_hook import fact_claims


def audit_facts(messages, trace):
    withheld = {e['message_index'] for e in trace if e['event'] == 'h_block'}
    results = []
    for index, message in enumerate(messages, 1):
        if message['role'] != 'assistant' or message.get('tool_calls'):
            continue
        for claim in fact_claims(messages[:index-1], message.get('content')):
            results.append(claim | {'message_index': index, 'user_visible': index not in withheld})
    return results


def fact_totals(audits):
    events = [(r, c) for r in audits for c in r['retail_facts']]
    wrong = [(r, c) for r, c in events if c['verdict'] == 'violation']
    return {'checked_claims': len(events), 'visible_checked_claims': sum(c['user_visible'] for r, c in events),
        'error_events': len(wrong), 'error_runs': len({r['run_id'] for r, c in wrong}),
        'visible_error_events': sum(c['user_visible'] for r, c in wrong),
        'visible_error_runs': len({r['run_id'] for r, c in wrong if c['user_visible']}),
        'kinds': {k: sum(c['kind'] == k for r, c in events) for k in sorted({c['kind'] for r, c in events})},
        'scope': 'Explicit uniquely bound claims only. Unparsed/ambiguous prose is unverified, not counted as correct.'}
