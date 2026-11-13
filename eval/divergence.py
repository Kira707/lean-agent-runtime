"""Candidate inspection location only; not a classifier or alternative grader."""


def first_unmatched(actual, standard):
    # Ignore auxiliary reads; alternative legal action order is left for analysis.
    for index, action in enumerate(standard):
        if action not in actual:
            return {'standard_index':index,'unmatched_action':action,
                    'actual_tail':actual[-4:],'classification':None}
    return None
