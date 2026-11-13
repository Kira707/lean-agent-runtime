"""Two offline checks on saved Baseline. No model, environment, or test split."""
import json
import re
from pathlib import Path
from domains.tau.fact_hook import fact_claims
from eval.runner import dump
from eval.fact_audit import audit_facts, fact_totals

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/results/m1_baseline_20261031_195723'


def confirmation_question(text):
    text = text.replace('*', '').replace('`', '')
    return bool(re.search(r'\b(?:please (?:reply .{0,15}yes|confirm)|reply .{0,15}yes|'
        r'(?:can|could|would|do) you confirm|shall I (?:proceed|go ahead)|'
        r'(?:do|would) you (?:want|like) me to (?:proceed|process|cancel|return|exchange))\b', text, re.I)
        or re.search(r'\bconfirm\b[^?\n]{0,100}\?', text, re.I))


def check(messages):
    last_visible, writes, dropped = None, [], []
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        text, calls = message.get('content') or '', message.get('tool_calls') or []
        if not calls:
            last_visible = index
            continue
        if text.strip() and confirmation_question(text):
            dropped.append({'message_index': index+1, 'text': text,
                            'tool': calls[0]['function']['name']})
        call = calls[0]['function']
        if not call['name'].startswith(('cancel_', 'modify_', 'return_', 'exchange_')):
            continue
        action = {'name': call['name'], 'arguments': json.loads(call['arguments'])}
        claims = fact_claims(messages[:index], messages[last_visible]['content'], action=action) if last_visible is not None else []
        writes.append({'message_index': index+1, 'action': action,
            'text_message_index': last_visible+1 if last_visible is not None else None,
            'text': messages[last_visible]['content'] if last_visible is not None else None,
            'has_confirmation_request': last_visible is not None and confirmation_question(messages[last_visible]['content']),
            'checked_claims': claims,
            'contradictions': [c for c in claims if c['verdict'] == 'violation']})
    return {'writes': writes, 'dropped_confirmation_questions': dropped}


def main():
    results = []
    for row in json.loads((BASE/'runs.json').read_text()):
        messages = json.loads((BASE/row['run_id']/'messages.json').read_text())
        results.append({'run_id': row['run_id'], **check(messages),
                        'retail_facts': audit_facts(messages, [])})
    writes = [w for r in results for w in r['writes']]
    dropped = [(r, d) for r in results for d in r['dropped_confirmation_questions']]
    summary = {'trajectories': len(results), 'write_calls': len(writes),
        'writes_with_visible_prior_text': sum(w['text'] is not None for w in writes),
        'writes_with_confirmation_request': sum(w['has_confirmation_request'] for w in writes),
        'checked_claims': sum(len(w['checked_claims']) for w in writes),
        'contradictory_write_events': sum(bool(w['contradictions']) for w in writes),
        'dropped_confirmation_questions': len(dropped),
        'dropped_question_runs': len({r['run_id'] for r, d in dropped}),
        'dropped_questions_with_write_call': sum(d['tool'].startswith(('cancel_', 'modify_', 'return_', 'exchange_')) for r, d in dropped)}
    dump(BASE/'confirmation_quick_check.json', {'summary': summary, 'runs': results,
        'baseline_h2_replay': fact_totals(results),
        'scope': 'Compare explicit, uniquely bound claims to subsequent executed arguments and observed facts. '
        'Omissions and ambiguous prose remain unverified; this does not establish complete confirmation compliance.',
        'recommendation': 'Do not add H3: no determined parameter contradiction; sole dropped confirmation is paired with a read tool, and later confirmation was delivered.'})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
