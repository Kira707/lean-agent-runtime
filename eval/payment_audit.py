"""Judge withheld/delivered text and interceptions independently of H feedback."""


def audit_interceptions(numeric_events, trace, payment_methods, block_reviews=None):
    blocks = [e for e in trace if e['event']=='h_block']
    withheld = {e['message_index'] for e in blocks}
    for event in numeric_events:
        if event['message_index'] in withheld:
            event['user_visible'] = False
            event['withheld_by_harness'] = True
    judged = []
    for block in blocks:
        verdicts = []
        for problem in block['problems']:
            if 'payment_method_id' not in problem:
                verdicts.append(None)  # H2 blocks are reviewed from the raw tool/statement relation.
                continue
            method = payment_methods.get(problem['payment_method_id'])
            verdicts.append(problem['claimed_value'][-4:] != str(method['last_four']) if method else None)
        verdict = ('correct' if any(v is True for v in verdicts) else
                   'false' if all(v is False for v in verdicts) else 'unknown')
        if block_reviews and str(block['message_index']) in block_reviews:
            verdict = block_reviews[str(block['message_index'])]['verdict']
        judged.append({'message_index':block['message_index'],'verdict':verdict,
                       'fact_verdicts':verdicts,'text':block['text'],'problems':block['problems'],
                       'mechanisms':sorted({'H1' if 'payment_method_id' in p else 'H2' for p in block['problems']}),
                       'review':(block_reviews or {}).get(str(block['message_index']))})
    correcting, extra_http_attempts = False, 0
    for event in trace:
        if event['event']=='h_block':correcting=True
        elif event['event'] in ('h_rewrite_accepted','h_fail_open','user_turn'):correcting=False
        elif event['event']=='model_call' and event.get('actor')=='agent' and correcting:
            extra_http_attempts += 1
    return {'blocked_messages':len(blocks),'extra_http_attempts':extra_http_attempts,
            'correct_interceptions':sum(e['verdict']=='correct' for e in judged),
            'false_interceptions':sum(e['verdict']=='false' for e in judged),
            'unknown_interceptions':sum(e['verdict']=='unknown' for e in judged),
            'interceptions':judged}
