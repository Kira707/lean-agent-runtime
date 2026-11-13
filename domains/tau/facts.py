"""Pure retail fact extraction shared by runtime and offline evaluation."""
import re

PATTERNS = {
    'card_last_four': r'\b(?:card|visa|mastercard|amex)(?:\s+credit\s+card)?[^.!?\n]{0,65}?\bending(?:\s+in)?\s*[:*` ]*([0-9]{4,})\b',
    'order_id': r'#?\b(W[0-9]+)\b',
    'item_id': r'\bitem\s*(?:id)?\s*[:#*` ]*([0-9]{10})\b',
    'amount': r'\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)',
}


def extract(text):
    return sorted([{'kind': kind, 'value': match.group(1).replace(',', ''),
                    'start': match.start(1), 'end': match.end(1)}
                   for kind, pattern in PATTERNS.items()
                   for match in re.finditer(pattern, text, re.I)
                   if not (kind == 'card_last_four' and
                           re.search(r'gift\s*$', text[max(0, match.start()-12):match.start()].replace('*', ''), re.I))],
                  key=lambda f: f['start'])


