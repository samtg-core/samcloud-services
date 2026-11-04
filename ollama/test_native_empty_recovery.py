"""A native request-only nudge is source identity, never a fabricated DB row."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .hermes_queue_identity import HeldSource, UnpersistedSource
from .inference_arbiter import Conflict, encoded
from .inference_identity import IdentityStore


NUDGE = ('You just executed tool calls but returned an empty response. Please process the tool '
         'results above and continue with the task.')


def history(*, later=False):
    messages = [{'role': 'user', 'content': 'task', '_row_id': 1},
                {'role': 'assistant', 'content': '', '_row_id': 2},
                {'role': 'tool', 'content': 'result', '_row_id': 3},
                {'role': 'assistant', 'content': '(empty)', 'reasoning': None,
                 'finish_reason': 'stop', 'timestamp': 1.0, '_empty_recovery_synthetic': True},
                {'role': 'user', 'content': NUDGE, 'timestamp': 2.0,
                 '_empty_recovery_synthetic': True}]
    if later:
        messages += [{'role': 'assistant', 'content': '', '_row_id': 4},
                     {'role': 'tool', 'content': 'later result', '_row_id': 5}]
    return messages


def capture(messages, *, api='api-2', turn='turn'):
    return HeldSource.capture('profile', 'session', 'main', messages,
                              turn_id=turn, api_request_id=api)


class NativeEmptyRecovery(unittest.TestCase):
    def test_pair_in_middle_keeps_later_rows_and_exact_intent_across_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'identity.sqlite'
            messages = history(later=True)
            source = capture(messages)
            self.assertEqual(source.rows, (1, 2, 3, 4, 5))
            request = {'model': 'qwen', 'messages': ['full wire request']}
            first = source.bind_stint(path, request)
            self.assertEqual(capture(deepcopy(messages)).bind_stint(path, request), first)
            code = '''import sys,json
from ollama.hermes_queue_identity import HeldSource
x=json.load(sys.stdin)
s=HeldSource.capture('profile','session','main',x['messages'],turn_id='turn',api_request_id='api-2')
k=s.bind_stint(x['path'],x['request']);print(json.dumps([k.grant_key,k.call_key]))
'''
            child = subprocess.run([sys.executable, '-c', code],
                input=json.dumps({'messages': messages, 'path': str(path), 'request': request}),
                capture_output=True, text=True, check=True,
                cwd=Path(__file__).resolve().parents[1], env={'PATH': os.defpath, 'LANG': 'C.UTF-8'})
            self.assertEqual(json.loads(child.stdout), [first.grant_key, first.call_key])
            changed = capture(messages, api='api-3').bind_stint(path, request)
            self.assertEqual(first.grant_key, changed.grant_key)
            self.assertNotEqual(first.call_key, changed.call_key)
            with self.assertRaises(Conflict):
                capture(messages).bind_stint(path, request | {'temperature': 1})
            without_pair = capture(messages[:3] + messages[5:]).bind_stint(path, request)
            self.assertNotEqual(first.call_key, without_pair.call_key)
            ordinary = capture(messages[:3] + messages[5:])
            store = IdentityStore(path)
            try:
                old = store.bind(ordinary.scope,
                    'conversation-stint:' + hashlib.sha256(encoded(['main', 'qwen']).encode()).hexdigest(),
                    'rows:' + hashlib.sha256(encoded(ordinary.rows).encode()).hexdigest(), request)
                self.assertEqual(without_pair.call_key, old.call_key)
            finally:
                store.close()
            # The marker derives from position, not just presence or canonical rows.
            moved = deepcopy(messages[:3] + messages[5:] + messages[3:5])
            self.assertNotEqual(first.call_key, capture(moved).bind_stint(path, request).call_key)
            self.assertNotIn(NUDGE.encode(), path.read_bytes())

    def test_only_one_exact_native_pair_is_accepted(self):
        cases = []
        for index, key, value in [(3, 'content', 'changed'), (4, 'content', NUDGE + '!'),
                                  (3, '_empty_recovery_synthetic', 1),
                                  (4, '_empty_recovery_synthetic', False),
                                  (3, 'reasoning', 'unqualified reasoning'),
                                  (3, 'finish_reason', 'length'),
                                  (3, 'tool_calls', []), (4, 'api_content', 'extra'),
                                  (3, '_row_id', None), (4, 'timestamp', True)]:
            case = history(); case[index][key] = value; cases.append(case)
        cases += [history()[3:], history()[:-1], history(later=True) + history()[3:],
                  history()[:3] + [{'role': 'user', 'content': 'unstored'}],
                  history() + [{'role': 'user', 'content': 'unstored'}]]
        case = history(); case[2]['role'] = 'assistant'; cases.append(case)
        case = history(); case[2]['_row_id'] = True; cases.append(case)
        for messages in cases:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                capture(messages)
        for turn, api in [('', 'api'), ('turn', '')]:
            with self.assertRaises(UnpersistedSource):
                capture(history(), turn=turn, api=api)
        with self.assertRaises(UnpersistedSource):
            HeldSource.capture('profile', 'session', 'compression', history())


if __name__ == '__main__':
    unittest.main()
