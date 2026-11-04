"""Exact request-only empty-recovery shape from Hermes 818c13be.

These rows deliberately have no SessionDB identity. A derived source marker
accounts for their position; it does not turn arbitrary unstored input into a row.
Unknown native shapes continue to refuse until independently qualified.
"""
import hashlib
import math


NATIVE_EMPTY_NUDGE = (
    'You just executed tool calls but returned an empty response. Please process the tool '
    'results above and continue with the task.'
)
NATIVE_EMPTY_MARKER = (
    'hermes-818c13be-native-empty-recovery',
    hashlib.sha256(NATIVE_EMPTY_NUDGE.encode()).hexdigest(),
)
_ASSISTANT_FIELDS = frozenset(('role', 'content', '_empty_recovery_synthetic',
                              'timestamp', 'reasoning', 'reasoning_content', 'finish_reason'))
_USER_FIELDS = frozenset(('role', 'content', '_empty_recovery_synthetic', 'timestamp'))


def native_empty_pair(messages, index):
    if index < 1 or index + 1 >= len(messages):
        return False
    previous, assistant, user = messages[index - 1:index + 2]
    if not all(isinstance(row, dict) for row in (previous, assistant, user)):
        return False
    row_id = previous.get('_row_id')
    if previous.get('role') != 'tool' or type(row_id) is not int or row_id <= 0:
        return False
    if not (assistant.keys() <= _ASSISTANT_FIELDS and user.keys() <= _USER_FIELDS):
        return False
    if not (assistant.get('role') == 'assistant' and assistant.get('content') == '(empty)'
            and user.get('role') == 'user' and user.get('content') == NATIVE_EMPTY_NUDGE):
        return False
    if assistant.get('reasoning') not in (None, '') or assistant.get('reasoning_content') not in (None, ''):
        return False
    if assistant.get('finish_reason', 'stop') != 'stop':
        return False
    for row in (assistant, user):
        if row.get('_empty_recovery_synthetic') is not True:
            return False
        if 'timestamp' in row and (type(row['timestamp']) not in (int, float)
                                   or not math.isfinite(row['timestamp'])):
            return False
    return True
