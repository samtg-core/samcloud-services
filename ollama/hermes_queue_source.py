"""Private, explicit compaction intents for not-yet-persisted native inputs.

An operation ID is allocated, never derived from prompt equality. The complete
held snapshot is durably saved before inference. Recovery requires that exact
ID and owning profile/session; there is deliberately no newest-file lookup.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .inference_arbiter import encoded, required
from .inference_identity import IdentityStore


def owner_scope(profile, session):
    required(profile)
    required(session)
    return 'hermes:' + hashlib.sha256(encoded([profile, session]).encode()).hexdigest()


@dataclass(frozen=True)
class CompactionSource:
    directory: Path
    scope: str
    operation: str
    digest: str
    purpose: str = 'compression'

    @classmethod
    def capture(cls, directory, profile, session, messages, *, purpose='compression'):
        if purpose not in ('compression','detached_fork','tool_auxiliary'):
            raise ValueError('explicit supported source purpose required')
        if not isinstance(messages, list) or not messages or any(
                not isinstance(message, dict) for message in messages):
            raise ValueError('compaction requires a held message list')
        scope = owner_scope(profile, session)
        source = encoded(messages)
        digest = hashlib.sha256(source.encode()).hexdigest()
        operation = uuid.uuid4().hex
        directory = Path(directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        body = encoded({'schema': 1, 'operation': operation, 'scope': scope,
            'purpose': purpose, 'digest': digest, 'messages': json.loads(source)})
        path = directory / (operation + '.json')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        # Confirm directory entry durability, not just a flushed file buffer.
        for folder in (directory, directory.parent, directory.parent.parent):
            parent = os.open(folder, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        return cls(directory, scope, operation, digest, purpose)

    @classmethod
    def load(cls, directory, profile, session, operation):
        if not isinstance(operation, str) or not re.fullmatch('[a-f0-9]{32}', operation):
            raise ValueError('explicit compaction operation ID required')
        directory = Path(directory)
        row = json.loads((directory / (operation + '.json')).read_text())
        scope = owner_scope(profile, session)
        if row.get('schema') != 1 or row.get('operation') != operation or row.get('scope') != scope:
            raise ValueError('compaction source ownership differs')
        purpose=row.get('purpose')
        if purpose not in ('compression','detached_fork','tool_auxiliary'):
            raise ValueError('unsupported held source purpose')
        source = cls(directory, scope, operation, row.get('digest', ''),purpose)
        source.messages()  # validates the snapshot before returning authority
        return source

    def messages(self):
        row = json.loads((self.directory / (self.operation + '.json')).read_text())
        if row.get('schema') != 1:
            raise ValueError('unsupported compaction source schema')
        if (row.get('scope'), row.get('operation'), row.get('purpose'), row.get('digest')) != (
                self.scope, self.operation, self.purpose, self.digest):
            raise ValueError('compaction source metadata changed')
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages or any(
                not isinstance(message, dict) for message in messages):
            raise ValueError('compaction source is not a held message list')
        if hashlib.sha256(encoded(messages).encode()).hexdigest() != self.digest:
            raise ValueError('compaction source snapshot changed')
        return messages

    def validate(self, messages):
        self.messages()
        if hashlib.sha256(encoded(messages).encode()).hexdigest() != self.digest:
            raise ValueError('explicit compaction source differs from held input')

    def bind(self, path, request):
        self.messages()
        store = IdentityStore(path)
        try:
            return store.bind_execution(self.scope, self.purpose,
                'operation:' + self.operation, request)
        finally:
            store.close()
