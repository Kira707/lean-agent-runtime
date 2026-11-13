"""Payment feedback from observed tools only; no catalog, task or grader access."""
import json
from domains.tau.facts import extract


def payment_feedback(messages, text):
    methods = {}
    for index, message in enumerate(messages, 1):
        if message['role'] != 'tool' or message.get('name') not in ('get_user_details','get_order_details'):
            continue
        content = message['content']
        if not content.startswith('{'):  # Native Error observations carry no payment facts.
            continue
        data = json.loads(content)
        for method_id, method in data.get('payment_methods',{}).items():
            if method_id.startswith('credit_card_'):
                methods[method_id] = {'last_four':str(method['last_four']), 'source_message_index':index}
        for payment in data.get('payment_history',[]):
            method_id = payment['payment_method_id']
            if method_id.startswith('credit_card_'):
                methods.setdefault(method_id, {'last_four':None,'source_message_index':index})
    problems = []
    for fact in extract(text or ''):
        if fact['kind'] != 'card_last_four':
            continue
        method_id = 'credit_card_' + fact['value']
        known = methods.get(method_id)
        if known is None or (known['last_four'] is not None and fact['value'][-4:] == known['last_four']):
            continue
        problems.append({'claimed_value':fact['value'], 'payment_method_id':method_id,
            'correct_last_four':known['last_four'], 'correct_known_field':'payment_method_id',
            'correct_known_value':method_id,'source_message_index':known['source_message_index']})
    if not problems:
        return None
    feedback = ('Runtime verification feedback, NOT a new customer message. Your previous text was NOT sent. '
        'It confuses a payment_method_id suffix with bank card ending digits. Correct observed facts: '
        + json.dumps(problems,ensure_ascii=False) + '. Rewrite your response using these facts. '
        'If correct_last_four is null, it has not been observed: query get_user_details or explicitly '
        'label the known payment_method_id instead of claiming card digits. Do not invent last_four '
        'or repeat business writes; this is only a response correction.')
    return {'text':feedback, 'problems':problems}
