"""Read exact private raw-token receipts; no model, journal mutation or authority.

The native generator writes these before output parsers. Complete here means
the file's bounded token sequence ended, never that all ranks drained or a lease
may be released. Numerical comparison separately requires the provider's end.
"""

import base64
import hashlib
import os
import re
import stat

from .http import decode_json
from .prepare import canonical
from .receipt_files import ReceiptUnavailable, _directory, _identity, _read, boot_id

_IDENTITY = {'group_id', 'generation', 'rank', 'device_rank', 'rank_manifest_sha256',
             'command_id', 'task_id', 'model'}


def parse_tokens(raw, expected):
    if (not isinstance(expected, dict) or set(expected) != _IDENTITY
            or not isinstance(raw, bytes) or len(raw) > 128 * 1024 or not raw.endswith(b'\n')):
        raise ReceiptUnavailable('bounded complete raw token receipt required')
    try:
        rows = [decode_json(line) for line in raw.splitlines()]
        keys = {'kind', 'group_id', 'generation', 'rank', 'device_rank', 'rank_manifest_sha256',
                'command_id', 'task_id', 'model', 'task_params_json', 'prompt', 'prompt_token_ids',
                'token_limit', 'collection'}
        if len(rows) != 18 or set(rows[0]) != keys:
            raise ValueError('shape')
        head = rows[0]
        if (head['kind'] != 'exo_raw_tokens_v1' or head['collection'] != 'before_output_parsers'
                or canonical({key: head[key] for key in expected}) != canonical(expected)
                or type(head['generation']) is not int or head['generation'] < 1
                or type(head['device_rank']) is not int or head['device_rank'] not in (0, 1, 2)
                or type(head['token_limit']) is not int or head['token_limit'] != 16
                or not isinstance(head['prompt'], str) or not isinstance(head['task_params_json'], str)
                or not isinstance(head['prompt_token_ids'], list) or not head['prompt_token_ids']
                or any(type(token) is not int or token < 0 for token in head['prompt_token_ids'])):
            raise ValueError('header')
        task = decode_json(head['task_params_json'])
        if (task.get('model') != expected['model'] or task.get('max_output_tokens') != 16
                or type(task.get('max_output_tokens')) is not int
                or type(task.get('temperature')) not in (float, int) or task['temperature'] != 0
                or task.get('bench') is not False or task.get('images') != []):
            raise ValueError('task')
        for index, row in enumerate(rows[1:17]):
            if (set(row) != {'kind', 'index', 'token_id', 'finish_reason', 'usage_json'}
                    or row['kind'] != 'token' or type(row['index']) is not int or row['index'] != index
                    or type(row['token_id']) is not int or row['token_id'] < 0
                    or row['finish_reason'] != ('length' if index == 15 else None)
                    or row['usage_json'] is not None and not isinstance(row['usage_json'], str)):
                raise ValueError('token')
        if canonical(rows[-1]) != canonical({'kind': 'complete', 'tokens': 16, 'finish_reason': 'length'}):
            raise ValueError('end')
        usage = decode_json(rows[16]['usage_json'])
        if (type(usage.get('completion_tokens')) is not int or usage['completion_tokens'] != 16
                or type(usage.get('prompt_tokens')) is not int
                or usage['prompt_tokens'] != len(head['prompt_token_ids'])):
            raise ValueError('usage')
        return {'header': head, 'task': task, 'token_ids': [row['token_id'] for row in rows[1:17]],
                'usage': usage}
    except (KeyError, TypeError, ValueError):
        raise ReceiptUnavailable('raw token identity or sequence unconfirmed') from None


def collect_tokens(home, uid, boot, expected, *, boot_reader=boot_id):
    if (type(uid) is not int or uid != os.getuid() or boot_reader() != boot
            or not isinstance(expected, dict) or set(expected) != _IDENTITY
            or not isinstance(expected['command_id'], str)
            or not re.fullmatch('[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
                                expected['command_id'])
            or not isinstance(expected['task_id'], str) or not 0 < len(expected['task_id']) <= 256):
        raise ReceiptUnavailable('exact own-host raw token binding required')
    directory = _directory(home + '/qualification-tokens')
    try:
        info = os.fstat(directory)
        if info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o700:
            raise ReceiptUnavailable('owned private token directory required')
        name = expected['command_id'] + '-' + hashlib.sha256(expected['task_id'].encode()).hexdigest() + '.jsonl'
        before = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if stat.S_IMODE(before.st_mode) != 0o600:
            raise ReceiptUnavailable('private raw token file mode changed')
        raw = _read(directory, name, uid)
        after = os.stat(name, dir_fd=directory, follow_symlinks=False)
        # Reading may update access time. Pin content and pathname identity,
        # including ownership, mode and link count, without rejecting that read.
        if (_identity(before), before.st_nlink) != (_identity(after), after.st_nlink):
            raise ReceiptUnavailable('raw token file changed during collection')
        parse_tokens(raw, expected)
        if boot_reader() != boot:
            raise ReceiptUnavailable('host boot changed during token read')
        return {'raw_base64': base64.b64encode(raw).decode('ascii'), 'raw_bytes': len(raw),
                'raw_sha256': hashlib.sha256(raw).hexdigest()}
    finally:
        os.close(directory)
