import json
from copy import deepcopy
from pathlib import Path
from domains.tau.fact_hook import fact_feedback, fact_claims
from scripts.quick_confirmation_audit import check, confirmation_question

BASE = Path(__file__).resolve().parents[1]/'docs/results/m1_baseline_20261031_195723'


def saved(run, index):
    messages = json.loads((BASE/run/'messages.json').read_text())
    return messages[:index-1], messages[index-1]['content']


def test_observed_item_price_and_attributes_are_bound_to_the_sku():
    history, text = saved('baseline_r01_train_138', 17)
    assert fact_feedback(history, text) is None
    price = fact_feedback(history, text.replace('$150.58', '$137.32'))
    assert any(p['kind'] == 'item_price' and p['expected_value'] == '150.58' for p in price['problems'])
    attributes = fact_feedback(history, text.replace('black / laser / wireless', 'black / laser / wired'))
    assert any(p['kind'] == 'item_attribute:connectivity' and p['expected_value'] == 'wireless' for p in attributes['problems'])


def test_refund_difference_and_direction_are_not_just_seen_numbers():
    history, text = saved('baseline_r01_train_138', 17)
    bad = fact_feedback(history, text.replace('$13.26', '$137.32'))
    assert any(p['kind'] == 'amount:difference' and p['expected_value'] == '13.26' for p in bad['problems'])
    bad = fact_feedback(history, text.replace('paid by you via PayPal', 'refunded to you via PayPal'))
    assert any(p['kind'] == 'amount:direction' and p['expected_value'] == 'charge' for p in bad['problems'])


def test_multi_item_refund_total_is_calculated_for_selected_items():
    history, text = saved('baseline_r01_dev_013', 15)
    assert fact_feedback(history, text) is None
    bad = fact_feedback(history, text.replace('$766.54', '$194.47'))
    assert any(p['kind'] == 'amount:refund' and p['expected_value'] == '766.54' for p in bad['problems'])


def test_order_and_item_relations_cannot_borrow_another_orders_facts():
    history, text = saved('baseline_r01_train_138', 17)
    order = next(json.loads(m['content']) for m in history if m.get('name') == 'get_order_details')
    other = deepcopy(order)
    other.update(order_id='#W0000001', items=[])
    history += [{'role': 'tool', 'name': 'get_order_details', 'content': json.dumps(other)}]
    bad = fact_feedback(history, text.replace('#W4500945', '#W0000001'))
    assert any(p['kind'] == 'order_item' for p in bad['problems'])
    bad = fact_feedback(history, text.replace('item 8214883393', 'item 3330317167'))
    assert any(p['kind'] in ('item_price', 'item_attribute:sensor type') for p in bad['problems'])


def test_real_comparisons_and_completed_modification_keep_their_meaning():
    for run, index in [('baseline_r01_dev_009', 21), ('baseline_r03_dev_001', 23),
                       ('baseline_r01_dev_019', 21), ('baseline_r02_train_092', 33),
                       ('baseline_r03_dev_004', 13), ('baseline_r01_dev_012', 49)]:
        history, text = saved(run, index)
        assert fact_feedback(history, text) is None, (run, index)


def test_unbound_prose_and_customer_questions_do_not_become_false_claims():
    history, _ = saved('baseline_r01_train_138', 17)
    assert fact_feedback(history, 'Would you like a lower-priced option?') is None
    assert fact_feedback(history, 'Are you referring to order #W0000001?') is None
    assert fact_feedback(history, 'The alternatives might differ; please choose an item.') is None


def test_quick_check_detects_actual_parameter_change_but_not_thanks_for_confirmation():
    messages = json.loads((BASE/'baseline_r01_dev_003/messages.json').read_text())
    altered = deepcopy(messages)
    for message in altered:
        for call in message.get('tool_calls') or []:
            if call['function']['name'] == 'return_delivered_order_items':
                args = json.loads(call['function']['arguments'])
                args['order_id'] = '#W0000001'
                call['function']['arguments'] = json.dumps(args)
    assert not any(w['contradictions'] for w in check(messages)['writes'])
    assert any(p['kind'] == 'order_id' for w in check(altered)['writes'] for p in w['contradictions'])
    assert confirmation_question('Please confirm this is the only item you want to change?')
    assert not confirmation_question("Thank you for confirming. I'll now process the change.")
    assert not confirmation_question('Let me verify the order to confirm the address.')
    messages=json.loads((BASE/'baseline_r02_train_235/messages.json').read_text())
    assert not any(w['contradictions'] for w in check(messages)['writes'])


def test_baseline_visible_replay_has_no_determined_h2_contradictions():
    # Regression against the real local comparisons/formatting used for calibration.
    count = 0
    for row in json.loads((BASE/'runs.json').read_text()):
        messages = json.loads((BASE/row['run_id']/'messages.json').read_text())
        for index, message in enumerate(messages):
            if message['role'] == 'assistant' and not message.get('tool_calls'):
                assert not any(c['verdict'] == 'violation' for c in fact_claims(messages[:index], message['content'])), (row['run_id'], index+1)
                count += 1
    assert count > 300


def test_h2_uses_the_existing_two_rewrite_limit_and_preserves_model_text():
    from runtime.agent import run
    from runtime.types import AgentState, ModelReply, Usage
    history, good = saved('baseline_r01_train_138', 17)
    bad = good.replace('$13.26', '$137.32')
    class Model:
        def __init__(self, texts): self.texts=iter(texts)
        def complete(self, *args, **kwargs):
            return ModelReply({'role':'assistant','content':next(self.texts)}, Usage(), {})
    class Env:
        mode='dialogue';done=False
        def __init__(self): self.sent=[]
        def list_tools(self): return []
        def respond(self, text): self.sent.append(text);self.done=True;return '###STOP###'
    for responses, expected, blocks, fail_open in [([bad, good], good, 1, 0), ([bad, bad, bad], bad, 2, 1)]:
        env, events = Env(), []
        result=run(Model(responses),env,env,system='policy',state=AgentState('h2',messages=deepcopy(history)),
            before_respond=fact_feedback,emit=lambda event,**data:events.append({'event':event,**data}))
        assert env.sent == [expected]
        assert result.state.hook_state['payment']['blocked_messages'] == blocks
        assert result.state.hook_state['payment']['fail_open_messages'] == fail_open
        assert sum(e['event']=='h_block' for e in events) == blocks


def test_combined_checks_share_feedback_and_do_not_duplicate_rewrite_allowances():
    from domains.tau.response_hooks import response_feedback
    history, text = saved('baseline_r01_train_138', 17)
    profile={'role':'tool','name':'get_user_details','content':json.dumps({
        'payment_methods':{'credit_card_6173085':{'last_four':'4491'}}})}
    bad=text.replace('$13.26','$137.32')+' Card ending 6173085.'
    assert response_feedback(history+[profile],bad,payment=False,retail=False) is None
    both=response_feedback(history+[profile],bad,payment=True,retail=True)
    assert any(p.get('payment_method_id')=='credit_card_6173085' for p in both['problems'])
    assert any(p.get('kind')=='amount:difference' for p in both['problems'])


def test_real_two_item_refund_and_charge_are_not_compared_to_the_net():
    path=BASE.parent/'m1_fact_guard_20261101_194039/fact_guard_r01_dev_001/messages.json'
    messages=json.loads(path.read_text())
    history,text=messages[:24],messages[24]['content']
    assert fact_feedback(history,text) is None
    bad=fact_feedback(history,text.replace('refund of $13.36','refund of $55.79'))
    assert any(p['kind']=='amount:refund' and p['expected_value']=='13.36' for p in bad['problems'])
    bad=fact_feedback(history,text.replace('charge of $69.15','charge of $55.79'))
    assert any(p['kind']=='amount:charge' and p['expected_value']=='69.15' for p in bad['problems'])


def test_amount_roles_and_order_referents_from_h1_saved_trajectories():
    base=BASE.parent/'m1_payment_guard_20261031_232015'
    for run_id,index in [('payment_guard_r02_dev_001',19),('payment_guard_r02_dev_019',17),
                         ('payment_guard_r02_train_044',19),('payment_guard_r03_train_369',27)]:
        messages=json.loads((base/run_id/'messages.json').read_text())
        assert fact_feedback(messages[:index-1],messages[index-1]['content']) is None,(run_id,index)


def test_unnamed_exchange_amount_and_explicit_product_id_are_not_mistyped():
    base=BASE.parent/'m1_fact_guard_20261101_195618'
    for run_id,index in [('fact_guard_r03_dev_001',25),('fact_guard_r03_dev_012',19),('fact_guard_r03_train_400',13)]:
        messages=json.loads((base/run_id/'messages.json').read_text())
        assert fact_feedback(messages[:index-1],messages[index-1]['content']) is None
    path=BASE.parent/'m1_targeted_fact_guard_20261101_214406/fact_guard_r01_dev_012/messages.json'
    messages=json.loads(path.read_text())
    assert fact_feedback(messages[:20],messages[20]['content']) is None
