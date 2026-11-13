"""Selected response checks share one two-rewrite allowance."""
from domains.tau.payment_hook import payment_feedback
from domains.tau.fact_hook import fact_feedback


def response_feedback(messages, text, *, payment=False, retail=False):
    feedback = []
    if payment:
        result = payment_feedback(messages, text)
        if result:
            feedback.append(result)
    if retail:
        result = fact_feedback(messages, text)
        if result:
            feedback.append(result)
    if not feedback:
        return None
    return {'text': '\n\n'.join(f['text'] for f in feedback),
            'problems': [p for f in feedback for p in f['problems']]}
