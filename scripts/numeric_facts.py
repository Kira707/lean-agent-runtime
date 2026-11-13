"""Bounded 5a extractor: explicit identifiers, dollar amounts and card endings.

Only prior tool data is authoritative here. Unmatched money/IDs are unverified,
not automatically false; descriptive SKU matching belongs to 5b.
"""
import json
from decimal import Decimal

from domains.tau.facts import PATTERNS, extract


def collect(value, facts):
    if isinstance(value, list):
        for item in value:
            collect(item, facts)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in ('order_id', 'item_id', 'last_four'):
                kind = {'last_four': 'card_last_four'}.get(key, key)
                facts[kind].add(str(item).lstrip('#'))
            if key in ('price', 'amount', 'balance') and isinstance(item, (int, float)):
                facts['amount'].add(str(Decimal(str(item)).quantize(Decimal('.01'))))
            collect(item, facts)


def scan_numeric(messages):
    facts = {key: set() for key in PATTERNS}
    events = []
    for index, message in enumerate(messages, 1):
        text = message.get('content') or ''
        if message['role'] == 'tool' and text.lstrip().startswith(('{', '[')):
            collect(json.loads(text), facts)
        elif message['role'] == 'user':
            # User order/item references are evidence; card/amount claims need typed tool data.
            for fact in extract(text):
                if fact['kind'] in ('order_id', 'item_id'):
                    facts[fact['kind']].add(fact['value'])
        elif message['role'] == 'assistant':
            for fact in extract(text):
                kind, value = fact['kind'], fact['value']
                normalized = str(Decimal(value).quantize(Decimal('.01'))) if kind == 'amount' else value
                known = facts[kind]
                # Matching a typed value does not establish all object relations.
                verdict = 'source_matched' if normalized in known else 'unverified'
                # Candidate detection only; offline authority verification determines errors.
                if kind == 'card_last_four' and (len(value) != 4 or (known and value not in known)):
                    verdict = 'violation'
                events.append(fact | {'message_index': index, 'verdict': verdict,
                    'known_values': sorted(known), 'user_visible': not bool(message.get('tool_calls'))})
    return events


def verify_card_claims(events, payment_methods):
    """Offline grading authority; never expose seed payment data to the Agent.

    A seven-digit 'ending' alone is not proof of a false statement. Confirm that
    it names a payment-method ID and contradicts that card's actual last_four.
    """
    for event in events:
        if event['kind'] != 'card_last_four' or event['verdict'] != 'violation':
            continue
        method_id = 'credit_card_' + event['value']
        method = payment_methods.get(method_id)
        if method is None:
            event['verdict'] = 'unverified'
            continue
        event['authority'] = {'payment_method_id': method_id, 'last_four': method['last_four']}
        if event['value'][-4:] == str(method['last_four']):
            event['verdict'] = 'unverified'
