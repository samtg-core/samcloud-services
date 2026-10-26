"""Explicit native transcript boundaries for the opt-in queue provider.

No profile/env/DB-latest-row discovery. The owning host passes its profile scope,
session and the transcript actually held for this call. Main calls require
persisted rows; compaction may explicitly capture a private durable source intent
before native input persistence. Cold-turn reconstruction remains separate.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from .inference_arbiter import encoded, required
from .inference_identity import IdentityStore, LocalSourceRefused
from .hermes_queue_source import CompactionSource, owner_scope


def intake_callback(path, scope, turn_id, keys, api_request_id=''):
    """Capture exact native ownership; advance its fence only after service intake."""
    def accepted(job):
        store = IdentityStore(path)
        try:
            store.record_turn_call(scope, turn_id, keys, api_request_id=api_request_id, job=job)
        finally:
            store.close()
    return accepted


class UnpersistedSource(LocalSourceRefused):
    pass


@dataclass(frozen=True)
class HeldSource:
    scope: str
    purpose: str
    rows: tuple[int, ...]
    turn_id: str = ''
    api_request_id: str = ''

    def __post_init__(self):
        required(self.scope)
        required(self.purpose)
        if self.turn_id:
            required(self.turn_id)
        if self.api_request_id:
            required(self.api_request_id)
        if not isinstance(self.rows, tuple) or not self.rows or any(
                type(row) is not int or row <= 0 for row in self.rows):
            raise ValueError('held source requires persisted positive row IDs')

    @classmethod
    def capture(cls, profile_scope, session_id, purpose, messages, *, turn_id='', api_request_id=''):
        for identifier in (profile_scope, session_id, purpose):
            required(identifier)
        if not isinstance(messages, list):
            raise ValueError('held transcript must be a list')
        rows = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError('held transcript has an invalid message')
            if message.get('role') == 'system':
                continue
            row = message.get('_row_id')
            if row is None:
                raise UnpersistedSource('held transcript contains an unpersisted message')
            if type(row) is not int or row <= 0:
                raise ValueError('held transcript contains an invalid persisted row ID')
            rows.append(row)
        if not rows:
            raise ValueError('held transcript has no persisted source rows')
        scope = 'hermes:' + hashlib.sha256(encoded([profile_scope, session_id]).encode()).hexdigest()
        return cls(scope, purpose, tuple(rows), turn_id, api_request_id)

    def bind(self, path, request):
        boundary = 'rows:' + hashlib.sha256(encoded(self.rows).encode()).hexdigest()
        store = IdentityStore(path)
        try:
            return store.bind_execution(self.scope, self.purpose, boundary, request)
        finally:
            store.close()

    def bind_stint(self, path, request):
        """One foreground conversation/model grant, distinct durable call keys.

        Separate namespace from per-execution grants: switching modes must not
        reinterpret a released execution grant as a reusable work session.
        Compaction uses bind_compaction with a validated native foreground owner;
        background sources cannot borrow this scheduling preference.
        """
        if self.purpose != 'main':
            raise ValueError('conversation stints require an owning main source')
        model = required(request.get('model'))
        work = 'conversation-stint:' + hashlib.sha256(encoded([self.purpose, model]).encode()).hexdigest()
        boundary = 'rows:' + hashlib.sha256(encoded(self.rows).encode()).hexdigest()
        store = IdentityStore(path)
        try:
            keys = store.bind(self.scope, work, boundary, request)
            if self.turn_id:
                keys = replace(keys, on_accepted=intake_callback(path, self.scope, self.turn_id, keys, self.api_request_id))
            return keys
        finally:
            store.close()


class HostBindings:
    """Context-local hook and engine seams; never a process-wide 'latest call'."""
    def __init__(self, identity_path, profile_scope):
        required(profile_scope)
        self.path, self.scope = identity_path, profile_scope
        self.main = ContextVar('queue_main_source', default=None)
        self.auxiliary = ContextVar('queue_aux_source', default=None)
        self.compaction_resume = ContextVar('queue_compaction_resume', default=None)
        self.compaction_owner = ContextVar('queue_compaction_owner', default=None)
        self.tool = ContextVar('queue_tool_source',default=None)
        self.tool_api = ContextVar('queue_tool_api',default=None)

    @contextmanager
    def tool_scope(self,session_id,turn_id,api_request_id,tool_call_id):
        ids=(owner_scope(self.scope,session_id),turn_id,api_request_id,tool_call_id)
        store=IdentityStore(self.path)
        try:
            known=all(isinstance(value,str) and value for value in ids) and store.db.execute(
                'SELECT 1 FROM native_tool_call WHERE scope=? AND turn_id=? AND api_request_id=? AND tool_call_id=?',
                ids).fetchone() is not None
        finally:store.close()
        token=self.tool.set((session_id,ids) if known else None)
        try:yield
        finally:self.tool.reset(token)

    @contextmanager
    def tool_api_scope(self):
        """One native API intention; a failed observer keeps its pending key.

        A subsequent call can advance only after a positively returned full
        response. Unknown contact reattaches to the exact saved operation;
        changed input is refused rather than silently submitted twice.
        """
        if self.tool.get() is None:raise ValueError('no exact owning tool call is bound')
        call={}
        token=self.tool_api.set(call)
        try:
            yield
            if call.get('operation'):
                store=IdentityStore(self.path)
                try:store.db.execute("UPDATE tool_aux_intent SET state='complete' WHERE operation=?",
                                     (call['operation'],))
                finally:store.close()
        finally:self.tool_api.reset(token)

    def tool_keys(self,request):
        bound,call=self.tool.get(),self.tool_api.get()
        if bound is None or call is None:raise ValueError('no owning auxiliary API intention')
        session,ids=bound
        directory=Path(self.path).parent/'tool-sources'
        store=IdentityStore(self.path)
        try:
            store.db.execute('BEGIN IMMEDIATE')
            try:
                row=store.db.execute("SELECT operation FROM tool_aux_intent WHERE scope=? AND turn_id=? "
                    "AND api_request_id=? AND tool_call_id=? AND state='pending'",ids).fetchone()
                if row:
                    source=CompactionSource.load(directory,self.scope,session,row[0])
                    source.validate(request.get('messages'))
                else:
                    source=CompactionSource.capture(directory,self.scope,session,request.get('messages'),
                                                    purpose='tool_auxiliary')
                    step=store.db.execute('SELECT COALESCE(MAX(step),0)+1 FROM tool_aux_intent WHERE '
                        'scope=? AND turn_id=? AND api_request_id=? AND tool_call_id=?',ids).fetchone()[0]
                    store.db.execute("INSERT INTO tool_aux_intent VALUES(?,?,?,?,?,?,'pending')",
                                     (*ids,step,source.operation))
                store.db.execute('COMMIT')
            except BaseException:
                store.db.execute('ROLLBACK');raise
            keys=source.bind(self.path,request)
            keys=store.bind_handoff(keys,store.conversation_stints(source.scope))
            call['operation']=source.operation
            return keys
        finally:store.close()

    def pre_api_request(self, *, session_id, conversation_history, turn_id='', api_request_id='', **unused):
        # Native observer hooks are fail-open. A rejected new snapshot must not
        # leave the previous request's keys usable by the custom provider.
        self.main.set(None)
        self.main.set(HeldSource.capture(self.scope, session_id, 'main', conversation_history,
                                        turn_id=turn_id, api_request_id=api_request_id))

    def post_api_request(self, **unused):
        self.main.set(None)

    def begin_foreground(self,session_id,turn_id):
        store=IdentityStore(self.path)
        try:
            store.foreground_turn(owner_scope(self.scope,session_id),turn_id,record=True)
        finally:
            store.close()

    def detached_source(self,session_id,turn_id,messages):
        """Native detached forks skip pre_llm_call and own no SessionDB rows.

        Require a known owning conversation and no positive foreground marker.
        An ordinary foreground persistence failure must still refuse, never
        silently switch to detached execution. Payload is private and durable.
        """
        store=IdentityStore(self.path)
        try:
            scope=owner_scope(self.scope,session_id)
            if (store.foreground_turn(scope,turn_id) or not store.conversation_stints(scope)):
                raise LocalSourceRefused('unpersisted foreground or unowned fork; no submission')
        finally:
            store.close()
        return CompactionSource.capture(Path(self.path).parent/'detached-sources',
            self.scope,session_id,messages,purpose='detached_fork')

    def finish_turn(self, session_id, turn_id):
        store = IdentityStore(self.path)
        try:
            return store.finish_turn(owner_scope(self.scope, session_id), turn_id)
        finally:
            store.close()

    def pending_events(self, after=0):
        store = IdentityStore(self.path)
        try:
            return store.pending_events(after=after)
        finally:
            store.close()

    def dependency_wait(self, session_id, turn_id, api_request_id, tool_call_id):
        store = IdentityStore(self.path)
        try:
            return store.dependency_wait(owner_scope(self.scope,session_id),
                                         turn_id,api_request_id,tool_call_id)
        finally:
            store.close()

    def complete_api(self, session_id, turn_id, api_request_id, tool_call_ids):
        store = IdentityStore(self.path)
        try:
            store.complete_api(owner_scope(self.scope,session_id),turn_id,api_request_id,tool_call_ids)
        finally:
            store.close()

    def acknowledge_event(self, event, verdict):
        store = IdentityStore(self.path)
        try:
            store.acknowledge_event(event, verdict)
        finally:
            store.close()

    @contextmanager
    def compression_scope(self, session_id, messages, *, purpose='compression', persist_unstored=False, owner_provider=None):
        source = self.compaction_resume.get()
        if source is not None:
            source = CompactionSource.load(Path(self.path).parent / 'compaction-sources',
                self.scope, session_id, source.operation)
            source.validate(messages)
        else:
            try:
                source = HeldSource.capture(self.scope, session_id, purpose, messages)
            except UnpersistedSource:
                if not persist_unstored or purpose != 'compression':
                    raise
                source = CompactionSource.capture(Path(self.path).parent / 'compaction-sources',
                    self.scope, session_id, messages)
        token = self.auxiliary.set(source)
        owner_token = self.compaction_owner.set(owner_provider)
        try:
            yield source
        finally:
            self.compaction_owner.reset(owner_token)
            self.auxiliary.reset(token)

    @contextmanager
    def resume_compression(self, session_id, operation):
        source = CompactionSource.load(Path(self.path).parent / 'compaction-sources',
            self.scope, session_id, operation)
        token = self.compaction_resume.set(source)
        try:
            yield source.messages()
        finally:
            self.compaction_resume.reset(token)

    def _keys(self, source, request):
        if source is None:
            raise LocalSourceRefused('no owning persisted call boundary is bound')
        return source.bind(self.path, request)

    def main_keys(self, request):
        source=self.main.get()
        keys=self._keys(source, request)
        return replace(keys,execution_only=True) if isinstance(source,CompactionSource) else keys

    def main_stint_keys(self, request):
        source = self.main.get()
        if isinstance(source,CompactionSource) and source.purpose=='detached_fork':
            return replace(source.bind(self.path,request),execution_only=True)
        if not isinstance(source, HeldSource):
            raise LocalSourceRefused('no owning foreground source is bound for a stint')
        return source.bind_stint(self.path, request)

    def auxiliary_keys(self, request):
        source = self.auxiliary.get()
        if source is None:
            raise LocalSourceRefused('no owning persisted call boundary is bound')
        store = IdentityStore(self.path)
        try:
            if source.purpose == 'compression':
                owner_provider = self.compaction_owner.get()
                owner = owner_provider() if owner_provider is not None else None
                if owner_provider is not None and (not isinstance(owner, tuple) or len(owner) != 2
                        or any(not isinstance(v, str) or not v.strip() for v in owner)):
                    raise LocalSourceRefused('foreground compaction owner ended or changed before submission')
                if isinstance(source, HeldSource):
                    boundary = 'rows:' + hashlib.sha256(encoded(source.rows).encode()).hexdigest()
                else:
                    source.messages()
                    boundary = 'operation:' + source.operation
                keys = store.bind_compaction(source.scope, boundary, request,
                    stint_model=owner[1] if owner else '')
                if keys.work_session:
                    return replace(keys, on_accepted=intake_callback(self.path, source.scope, owner[0], keys))
            else:
                keys = self._keys(source, request)
            return store.bind_handoff(keys, store.conversation_stints(source.scope))
        finally:
            store.close()
