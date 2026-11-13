"""H2: scoped, deterministic retail claims; ambiguous prose is unverified."""
import json
import re
from domains.tau.retail_evidence import (observed, money, contains, selected_order,
                                         bound_units, action_amounts)

AMOUNTS = re.compile(r'\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)')


def inferred_action(evidence, text, order, units):
    if not order:
        return None
    previous = evidence['last_action']
    if previous and previous['arguments'].get('order_id') == order['order_id'] and order['status'] not in ('pending', 'delivered'):
        return previous
    new = {}
    for unit in units:
        item = unit['item']
        if item and unit['phase'] == 'new':
            new.setdefault(item['product_id'], set()).add(item['item_id'])
    originals = {}
    for item in order['items']:
        originals.setdefault(item['product_id'], []).append(item['item_id'])
    if new and all(len(ids) == 1 and len(originals.get(p, [])) == 1 for p, ids in new.items()):
        return {'name': 'modify_pending_order_items', 'arguments': {
            'order_id': order['order_id'], 'item_ids': [originals[p][0] for p in new],
            'new_item_ids': [next(iter(ids)) for ids in new.values()]}}
    if re.search(r'\bcancel(?:lation|led|ling)?\b', text, re.I) and not new:
        # A generic "cannot cancel after modification" is not a cancellation proposal.
        if re.search(r'\b(?:cancel (?:this |the |your )?order|cancellation|cancelled and|cancelling)\b', text, re.I):
            return {'name': 'cancel_pending_order', 'arguments': {'order_id': order['order_id']}}
    match = re.search(r'\bitems? to return\s*[:\-]?([^\n]*(?:\n(?!\s*\n)[^\n]*)*)', text, re.I)
    selected = []
    if match:
        block = re.split(r'\b(?:refund|total|note)\b', match.group(1), maxsplit=1, flags=re.I)[0]
        selected = [i['item_id'] for i in order['items'] if contains(block, i['name']) or i['item_id'] in block]
        # Repeated SKUs/identical product names without explicit quantity are ambiguous.
        if len(selected) != len(set(selected)):
            return None
    if not selected and len(order['items']) == 1 and re.search(r'\breturn\b', text, re.I):
        selected = [order['items'][0]['item_id']]
    if selected:
        return {'name': 'return_delivered_order_items',
                'arguments': {'order_id': order['order_id'], 'item_ids': selected}}
    previous = evidence['last_action']
    if previous and previous['arguments'].get('order_id') == order['order_id'] and re.search(
            r'\b(?:processed|submitted|requested|successfully|completed)\b', text, re.I):
        return previous
    return None


def amount_role(text, start, end):
    """Nearest explicit monetary label. Item prices are handled by bound units."""
    before = text[max(0, start-45):start].split('\n')[-1].lower()
    after = text[end:end+40].split('\n')[0].lower()
    if re.search(r'\b(?:balance|budget)\b', before[-22:]):
        return None
    if re.search(r'\boriginal (?:charge|total|payment)(?: amount)?\s*(?::|of|was|is)?\s*$',before):
        return 'original_total'
    if re.search(r'\border total\b',before):
        return 'order_total'
    if re.search(r'\b(?:difference|cheaper|more|less|reduction|increase|additional)\b', before[-30:]+after[:22]):
        return 'difference'
    if re.search(r'\b(?:total|combined|net|full refund)\b', before[-30:]):
        return 'refund' if 'refund' in before[-30:] else 'total'
    if re.search(r'\brefund(?:ed)?(?:\s+(?:amount|of|for|is|would be))*[\s:*\-]*$', before) or re.match(
            r'[\s*]*(?:refund|refunded)\b', after):
        return 'refund'
    if re.search(r'\b(?:charge|charged|pay|owe)\s*(?::|of)?\s*$',before) or re.match(
            r'\s*(?:charged to|charge to|to (?:pay|be charged))\b',after):
        return 'charge'
    return None


def fact_claims(messages, text, *, action=None):
    """Return evaluated claims, not a boolean for every arbitrary sentence.

    action is supplied ONLY by offline quick check (future executed arguments).
    Runtime calls never pass it and use already-observed evidence exclusively.
    """
    evidence = observed(messages)
    clean = (text or '').replace('*', '').replace('`', '')
    order = selected_order(evidence, clean)
    units = bound_units(evidence, clean, order)
    claims = []

    def add(kind, claimed, expected, source, context):
        claims.append({'kind': kind, 'claimed_value': str(claimed), 'expected_value': str(expected),
                       'verdict': 'correct' if claimed == expected else 'violation',
                       'source_message_index': source, 'context': context})

    for unit in units:
        item, part = unit['item'], unit['text']
        ids = unit['explicit_ids']
        if len(ids) == 1 and ids[0] in evidence['products'] and re.search(
                r'\b(?:item(?:\s*id)?|sku)\s*[:# ]*'+re.escape(ids[0])+r'\b', part, re.I):
            add('item_id_type', 'product_id', 'item_id', evidence['products'][ids[0]]['source_message_index'], part)
        if not item:
            continue
        source = item['source_message_index']
        names = {i['name'] for i in evidence['items'].values() if contains(part, i['name'])}
        if len(names) == 1 and ids:
            add('item_name', next(iter(names)), item['name'], source, part)
        # Attribute contradictions need an explicit SKU, not a SKU inferred from these same attributes.
        comparative = re.search(r'\boriginal\b.+\b(?:would be|compared to|rather than)\b', part, re.I)
        if len(ids) == 1 and not comparative:
            for key, value in unit['claims'].items():
                add('item_attribute:'+key, value, item['options'].get(key), source, part)
        if order and unit['phase'] == 'old' and ids and re.search(r'^\s*[-\s]*(?:current|original|item to|item being|from)\b', part, re.I):
            if not any(i['item_id'] == item['item_id'] for i in order['items']):
                add('order_item', item['item_id'], 'an item in '+order['order_id'], order['source_message_index'], part)
        amounts = list(AMOUNTS.finditer(part))
        if unit['price_binding'] and len(amounts) == 1 and not re.search(r'\b(?:refund\w*|difference|total|cheaper|increase|reduc\w*|balance|charge)\b|[+−–]\s*\$', part, re.I):
            if ids or names or re.search(r'\b(?:current|original|old|new|from|to|with)\s*:', part, re.I):
                add('item_price', money(amounts[0][1].replace(',', '')), money(item['price']), source, part)

    explicit = action or inferred_action(evidence, clean, order, units)
    expected = action_amounts(evidence, explicit) if explicit else None
    for match in AMOUNTS.finditer(clean):
        line = clean[:match.start()].split('\n')[-1] + match[0] + clean[match.end():].split('\n')[0]
        # A line with multiple item prices is evaluated locally, not as one aggregate.
        if len(AMOUNTS.findall(line)) > 1:
            continue
        role = amount_role(clean, match.start(), match.end())
        if role is None:
            continue
        value = money(match[1].replace(',', ''))
        if role=='order_total' and order:
            add('amount:order_total',value,sum((money(i['price']) for i in order['items']),money(0)),
                order['source_message_index'],line)
            continue
        if not expected:
            continue
        local_expected = expected
        named_parts = [part for name,part in expected.get('_parts',{}).items() if contains(line,name)]
        aggregate = re.search(r'\b(?:net|combined|total)\b', line, re.I)
        if role in ('difference','refund','charge') and len(expected.get('_parts',{}))>1 and not named_parts and not aggregate:
            continue  # Unnamed per-item amount is unbound; never assume it means the net.
        if len(named_parts) == 1 and not aggregate:
            local_expected = named_parts[0]
        target = local_expected.get(role)
        if role == 'total' and 'difference' in local_expected:
            if not re.search(r'\b(?:net|combined|difference|charge|refund)\b',line,re.I):
                continue  # Unqualified "Total" may mean order value or net change.
            target = abs(local_expected['difference'])
        if target is None:
            continue
        if role == 'difference':
            target = abs(target)
            # A named per-item difference in a multi-item exchange is not the net amount.
            if explicit and len(explicit['arguments'].get('item_ids', [])) > 1 and not aggregate and local_expected is expected:
                continue
        add('amount:'+role, value, target, order['source_message_index'] if order else None, line)
        if 'difference' in local_expected and value == target and role in ('difference','refund'):
            direction = ('refund' if re.search(r'\b(?:refund(?:ed)?|cheaper|less|reduction)\b', line, re.I) else
                         'charge' if re.search(r'\b(?:charge|charged|pay|additional|more|increase|owe)\b', line, re.I) else None)
            if direction and local_expected['difference']:
                add('amount:direction', direction, 'charge' if local_expected['difference'] > 0 else 'refund',
                    order['source_message_index'] if order else None, line)
    if action:
        order_ids = {'#'+i.lstrip('#') for i in re.findall(r'#?\bW[0-9]+\b', clean)}
        if len(order_ids) == 1 and action['arguments'].get('order_id'):
            add('order_id', next(iter(order_ids)), action['arguments']['order_id'], None, clean)
        wanted = set(action['arguments'].get('item_ids', []) + action['arguments'].get('new_item_ids', []))
        for unit in units:
            # Explicit action-detail SKU labels only. Other options and order inventories may be mentioned.
            if wanted and len(unit['explicit_ids']) == 1 and re.search(r'\b(?:new item|item to (?:return|exchange)|item being exchanged)\b', unit['text'], re.I):
                item_id = unit['explicit_ids'][0]
                if item_id not in wanted:
                    add('action_item_id', item_id, sorted(wanted), None, unit['text'])
    return claims


def fact_feedback(messages, text):
    problems = [c for c in fact_claims(messages, text) if c['verdict'] == 'violation']
    if not problems:
        return None
    return {'problems': problems, 'text':
        'Runtime verification feedback, NOT a new customer message. Your previous text was NOT sent. '
        'The following explicit retail claims contradict observed tool data or arithmetic: '
        + json.dumps(problems, ensure_ascii=False) + '. Rewrite using the correct values and their '
        'corresponding order/items. Query tools if more facts are needed. Do not repeat business writes. '
        'This is a response correction; do not claim that the customer has confirmed new details.'}
