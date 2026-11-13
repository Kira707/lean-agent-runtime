"""Observed retail entities only. No task, seed database, or scoring imports."""
from decimal import Decimal
import json
import re


def money(value):
    return Decimal(str(value)).quantize(Decimal('.01'))


def normalized(text):
    return re.sub(r'[^a-z0-9.%]+', ' ', str(text).lower()).strip()


def contains(text, value):
    return re.search(r'(?<![a-z0-9])'+re.escape(normalized(value))+r'(?![a-z0-9])', normalized(text)) is not None


def observed(messages):
    orders, items, products, methods = {}, {}, {}, {}
    last_action = None
    calls = {}
    for index, message in enumerate(messages, 1):
        for call in message.get('tool_calls') or []:
            calls[call['id']] = call['function']
        if message['role'] != 'tool' or not message.get('content', '').lstrip().startswith('{'):
            continue
        data = json.loads(message['content'])
        source = {'source_message_index': index}
        call = calls.get(message.get('tool_call_id'))
        if call and call['name'].startswith(('cancel_', 'modify_', 'return_', 'exchange_')):
            last_action = {'name': call['name'], 'arguments': json.loads(call['arguments'])}
            last_action['amounts'] = action_amounts({'orders': orders, 'items': items}, last_action)
        if 'order_id' in data and 'items' in data:
            orders[data['order_id']] = data | source
            for item in data['items']:
                items[item['item_id']] = item | source
        if 'variants' in data:
            products[data['product_id']] = data | source
            for item_id, item in data['variants'].items():
                items[item_id] = item | {'name': data['name'], 'product_id': data['product_id']} | source
        methods.update(data.get('payment_methods', {}))
    return {'orders': orders, 'items': items, 'products': products,
            'methods': methods, 'last_action': last_action}


def selected_order(evidence, text):
    ids = set(re.findall(r'#?\bW[0-9]+\b', text))
    ids = {'#'+v.lstrip('#') for v in ids}
    if len(ids) == 1:
        return evidence['orders'].get(next(iter(ids)))
    if not ids:
        item_ids=set(re.findall(r'\b[0-9]{10}\b',text))
        related=[o for o in evidence['orders'].values() if any(i['item_id'] in item_ids for i in o['items'])]
        if len(related)==1:
            return related[0]
        if len(evidence['orders'])==1:
            return next(iter(evidence['orders'].values()))
    return None


def option_claims(text, pool):
    vocabulary = {}
    for item in pool:
        for key, value in item['options'].items():
            vocabulary.setdefault(key, set()).add(value)
    claims = {}
    for key, values in vocabulary.items():
        matched = {v for v in values if contains(text, v)}
        # 'gold' inside 'rose gold' and 'blue' inside 'light blue' are one claim.
        matched = {v for v in matched if not any(v != w and contains(w, v) for w in matched)}
        if len(matched) == 1:
            claims[key] = next(iter(matched))
    return claims


def bound_units(evidence, text, order):
    """Bind explicit IDs or uniquely described variants, never by claimed price.

    Lines/arrows/table cells are local units. Ambiguous prose stays unbound.
    """
    product_id, phase = None, None
    units = []
    for line in text.splitlines():
        clean = line.replace('*', '').replace('`', '')
        names = {item['product_id'] for item in evidence['items'].values() if contains(clean, item['name'])}
        if len(names) == 1:
            product_id = next(iter(names))
        elif len(names) > 1:
            product_id = None
        if re.search(r'^\s*[-\s]*(?:(?:new|replacement)(?: item| variant| price)?\s*:|(?:to|with)\s*:)', clean, re.I):
            phase = 'new'
        elif re.search(r'^\s*[-\s]*(?:(?:current|original|old)(?: item| price)?\s*:|(?:from|replace)\s*:|item (?:to|being))', clean, re.I):
            phase = 'old'
        parts = re.split(r'→|\s+->\s+|\||(?<=[.!?])\s+(?=[A-Z])', clean)
        for position, part in enumerate(parts):
            local_phase = ('old' if position == 0 else 'new') if '→' in clean or ' -> ' in clean else phase
            ids = re.findall(r'\b[0-9]{10}\b', part)
            explicit = [evidence['items'][i] for i in ids if i in evidence['items']]
            if len(explicit) == 1:
                product_id = explicit[0]['product_id']
            pool = [i for i in evidence['items'].values() if i['product_id'] == product_id]
            claims = option_claims(part, pool)
            candidates = explicit
            if not ids and pool:
                candidates = pool
                if order and local_phase != 'new':
                    originals = {i['item_id'] for i in order['items']}
                    candidates = [i for i in candidates if i['item_id'] in originals]
                candidates = [i for i in candidates if all(i['options'].get(k) == v for k, v in claims.items())]
                if not claims and not names and local_phase != 'old':
                    candidates = []
            item = candidates[0] if len(candidates) == 1 else None
            units.append({'text': part, 'item': item, 'claims': claims, 'phase': local_phase,
                          'explicit_ids': ids, 'product_id': product_id,
                          'price_binding': clean.count('→') + clean.count(' -> ') <= 1})
    return units


def action_amounts(evidence, action):
    """Amounts implied by one explicit action, with duplicate item counts kept."""
    if 'amounts' in action:
        return action['amounts']
    args = action['arguments']
    order = evidence['orders'].get(args.get('order_id'))
    if not order:
        return None
    old = {i['item_id']: i for i in order['items']}
    items = args.get('item_ids', [])
    replacements = args.get('new_item_ids', [])
    if any(i not in old for i in items) or any(i not in evidence['items'] for i in replacements):
        return None
    if replacements:
        if len(items) != len(replacements):
            return None
        parts = {}
        for a, b in zip(items, replacements):
            name = old[a]['name']
            parts[name] = parts.get(name, Decimal(0)) + money(evidence['items'][b]['price'])-money(old[a]['price'])
        delta = sum(parts.values(), Decimal(0))
        return {'difference': delta, 'refund': max(-delta, Decimal(0)),
                'charge': max(delta, Decimal(0)),
                'original_total': sum((money(i['price']) for i in order['items']),Decimal(0)),
                '_parts': {name:{'difference':d,'refund':max(-d,Decimal(0)),'charge':max(d,Decimal(0))}
                           for name,d in parts.items()}}
    if action['name'].startswith('return_'):
        return {'refund': sum((money(old[i]['price']) for i in items), Decimal(0))}
    total = sum((money(i['price']) for i in order['items']), Decimal(0))
    return {'total': total, 'refund': total}
