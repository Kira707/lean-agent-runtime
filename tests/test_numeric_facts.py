from scripts.numeric_facts import extract, scan_numeric, verify_card_claims


def test_typed_card_id_is_not_last_four():
    messages = [{'role': 'tool', 'content': '{"payment_methods":{"credit_card_6173085":{"last_four":"4491"}}}'},
                {'role': 'assistant', 'content': 'Visa card ending 6173085; card ending in 4491.'}]
    events = scan_numeric(messages)
    assert [e['verdict'] for e in events] == ['violation', 'source_matched']


def test_money_and_identifiers_unknown_are_not_errors():
    messages = [{'role': 'tool', 'content': '{"order_id":"#W123","item_id":"1234567890","price":12.5}'},
                {'role': 'assistant', 'content': 'Order #W123 item ID 1234567890 costs $12.50; difference $2.25.'}]
    events = scan_numeric(messages)
    assert [e['verdict'] for e in events] == ['source_matched', 'source_matched', 'source_matched', 'unverified']


def test_only_prior_evidence_and_visible_layer():
    messages = [{'role': 'assistant', 'content': 'Visa ending 1234', 'tool_calls': [{}]},
                {'role': 'tool', 'content': '{"last_four":"1234"}'}]
    event = scan_numeric(messages)[0]
    assert event['verdict'] == 'unverified' and not event['user_visible']
    assert extract('4 days, 5–7 business days, credit_card_1234567') == []
    assert extract('gift card ending in 5701566') == []


def test_card_error_requires_authoritative_last_four_conflict():
    events=[{'kind':'card_last_four','value':v,'verdict':'violation'}
            for v in ('6173085','1001234','9999999')]
    verify_card_claims(events,{'credit_card_6173085':{'last_four':'4491'},
                              'credit_card_1001234':{'last_four':'1234'}})
    assert [e['verdict'] for e in events]==['violation','unverified','unverified']
    assert events[0]['authority']['last_four']=='4491'
