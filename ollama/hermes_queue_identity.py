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
from .hermes_queue_recovery import NATIVE_EMPTY_MARKER, native_empty_pair


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
    recovery_after: int | None = None

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
        if self.recovery_after is not None and (
                type(self.recovery_after) is not int or not 0 < self.recovery_after <= len(self.rows)
                or self.purpose != 'main' or not self.turn_id or not self.api_request_id):
            raise ValueError('native recovery requires an exact foreground API boundary')

    @classmethod
    def capture(cls, profile_scope, session_id, purpose, messages, *, turn_id='', api_request_id=''):
        for identifier in (profile_scope, session_id, purpose):
            required(identifier)
        if not isinstance(messages, list):
            raise ValueError('held transcript must be a list')
        rows, recovery_after, index = [], None, 0
        while index < len(messages):
            message = messages[index]
            index += 1
            if not isinstance(message, dict):
                raise ValueError('held transcript has an invalid message')
            if message.get('role') == 'system':
                continue
            row = message.get('_row_id')
            if row is None:
                if (purpose == 'main' and turn_id and api_request_id and recovery_after is None
                        and native_empty_pair(messages, index - 1)):
                    recovery_after = len(rows)
                    index += 1
                    continue
                raise UnpersistedSource('held transcript contains an unpersisted message')
            if message.get('_empty_recovery_synthetic'):
                raise UnpersistedSource('native recovery cannot claim a persisted row ID')
            if type(row) is not int or row <= 0:
                raise ValueError('held transcript contains an invalid persisted row ID')
            rows.append(row)
        if not rows:
            raise ValueError('held transcript has no persisted source rows')
        scope = 'hermes:' + hashlib.sha256(encoded([profile_scope, session_id]).encode()).hexdigest()
        return cls(scope, purpose, tuple(rows), turn_id, api_request_id, recovery_after)

    def call_boundary(self):
        if self.recovery_after is None:
            return 'rows:' + hashlib.sha256(encoded(self.rows).encode()).hexdigest()
        parts = list(self.rows)
        parts.insert(self.recovery_after, (*NATIVE_EMPTY_MARKER, self.rows[self.recovery_after - 1]))
        # A native retry reuses its API identity; a later explicit API intention
        # must not replay a cached empty response merely because no DB row grew.
        return 'native-empty:' + hashlib.sha256(encoded(
            [self.turn_id, self.api_request_id, parts]).encode()).hexdigest()

    def bind(self, path, request, *, job_turn=''):
        boundary = self.call_boundary()
        store = IdentityStore(path)
        try:
            return store.bind_execution(self.scope, self.purpose, boundary, request, job_turn=job_turn)
        finally:
            store.close()

    def bind_stint(self, path, request, *, job_turn=''):
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
        boundary = self.call_boundary()
        store = IdentityStore(path)
        try:
            keys = store.bind(self.scope, work, boundary, request, job_turn=job_turn)
            if self.turn_id:
                keys = replace(keys, on_accepted=intake_callback(path, self.scope, self.turn_id, keys, self.api_request_id))
            return keys
        finally:
            store.close()


class HostBindings:
    """Context-local hook and engine seams; never a process-wide 'latest call'."""
    def __init__(self, identity_path, profile_scope, *, job_continuity=False):
        required(profile_scope)
        if type(job_continuity) is not bool:
            raise ValueError('explicit boolean job policy required')
        self.path, self.scope = identity_path, profile_scope
        self.job_continuity = job_continuity
        self.main = ContextVar('queue_main_source', default=None)
        self.auxiliary = ContextVar('queue_aux_source', default=None)
        self.compaction_resume = ContextVar('queue_compaction_resume', default=None)
        self.compaction_owner = ContextVar('queue_compaction_owner', default=None)
        self.tool = ContextVar('queue_tool_source',default=None)
        self.tool_api = ContextVar('queue_tool_api',default=None)
        self._session_databases = {}
        self._verified_foregrounds = {}
        if job_continuity:
            store = IdentityStore(self.path)
            try:
                store.db.execute('''CREATE TABLE IF NOT EXISTS native_session_alias (
                    profile TEXT NOT NULL, session_id TEXT NOT NULL, owner_session TEXT NOT NULL,
                    parent_session TEXT NOT NULL, PRIMARY KEY(profile,session_id))''')
            finally:
                store.close()

    def owner_session(self, session_id):
        """Only explicit, committed native compaction lineage shares ownership."""
        if not self.job_continuity:
            return session_id
        store = IdentityStore(self.path)
        try:
            row = store.db.execute('SELECT owner_session FROM native_session_alias '
                'WHERE profile=? AND session_id=?', (self.scope, session_id)).fetchone()
            return row[0] if row is not None else session_id
        finally:
            store.close()

    def bind_session_database(self, session_id, database):
        """An owning context engine supplies its exact SessionDB, never a search."""
        if self.job_continuity and session_id and database is not None:
            self._session_databases[session_id] = database

    def foreground_bound(self, session_id, turn_id):
        owner = self.owner_session(session_id)
        if self._verified_foregrounds.get(owner) != turn_id:
            return False
        store = IdentityStore(self.path)
        try:
            scope = owner_scope(self.scope, owner)
            return store.foreground_turn(scope, turn_id) and store.db.execute(
                'SELECT 1 FROM native_job_binding WHERE scope=? AND turn_id=?',
                (scope, turn_id)).fetchone() is not None
        finally:
            store.close()

    def verify_foreground(self, session_id, turn_id, *, enabled):
        owner = self.owner_session(session_id)
        if enabled:
            self._verified_foregrounds[owner] = turn_id
        elif self._verified_foregrounds.get(owner) == turn_id:
            self._verified_foregrounds.pop(owner, None)

    def recover_compaction_lineage(self, session_id, turn_id):
        """Recover the publication/alias crash window from exact native lineage.

        This runs only at a positively verified foreground begin. A normal new,
        reset or fork session cannot inherit an old job by username or recency.
        Partial alias commits can be replayed; old call identities never move.
        """
        if not self.job_continuity or self.owner_session(session_id) != session_id:
            return
        database = self._session_databases.get(session_id)
        if database is None:
            raise LocalSourceRefused('native job requires its owning queued context-engine SessionDB')
        from agent.relay_runtime import active_turn
        turn = active_turn(session_id)
        if turn is None or turn.turn_id != turn_id or turn.lease.profile_key != self.scope:
            raise LocalSourceRefused('session lineage recovery has no live foreground owner')
        child = database.get_session(session_id)
        links, visited = [], {session_id}
        while isinstance(child, dict) and child.get('parent_session_id'):
            parent_id = child['parent_session_id']
            if parent_id in visited:
                raise LocalSourceRefused('native compression lineage contains a cycle')
            visited.add(parent_id)
            parent = database.get_session(parent_id)
            if (not isinstance(parent, dict) or parent.get('end_reason') != 'compression'
                    or parent.get('source') != child.get('source')
                    or parent.get('profile_name') != child.get('profile_name')):
                break
            links.append((parent_id, child['id']))
            child = parent
        store = IdentityStore(self.path)
        try:
            candidates = {}
            for parent_id, _ in links:
                owner = self.owner_session(parent_id)
                scope = owner_scope(self.scope, owner)
                active = store.db.execute('SELECT job_key FROM native_requester_job WHERE scope=?',
                                          (scope,)).fetchone()
                if active is not None:
                    candidates[owner] = active[0]
            if not candidates:
                return
            if len(candidates) != 1:
                raise LocalSourceRefused('native lineage contains multiple unfinished parents')
            owner, expected = next(iter(candidates.items()))
            # Older completed ancestors are history, not this unfinished job.
            boundary = next(index for index, (parent_id, _) in enumerate(links)
                            if self.owner_session(parent_id) == owner)
            links = links[:boundary + 1]
            # An already-used raw child scope must never be reinterpreted as
            # the ancestor's calls. Such a pre-existing conflict needs recovery.
            for _, child_id in links:
                if self.owner_session(child_id) == child_id:
                    child_scope = owner_scope(self.scope, child_id)
                    independent = store.db.execute(
                        'SELECT job_key FROM native_requester_job WHERE scope=?', (child_scope,)).fetchone()
                    if ((independent is not None and independent[0] != expected)
                            or store.db.execute('SELECT 1 FROM work_identity WHERE scope=?',
                                                (child_scope,)).fetchone()):
                        raise LocalSourceRefused('compression child already has frozen independent work')
            if store.begin_native_job(owner_scope(self.scope, owner), turn_id,
                    is_current=lambda: active_turn(session_id) is turn) != expected:
                raise LocalSourceRefused('native parent changed during lineage recovery')
        finally:
            store.close()
        # Persist from the root outward. A process exit between these commits
        # leaves an exact prefix which the next owner can recover identically.
        for parent_id, child_id in reversed(links):
            self.link_compaction_session(parent_id, child_id, turn_id)

    def link_compaction_session(self, parent_session, child_session, turn_id):
        """The context engine calls this only after durable native publication."""
        required(parent_session); required(child_session); required(turn_id)
        if not self.job_continuity or parent_session == child_session:
            return
        store = IdentityStore(self.path)
        try:
            store.db.execute('BEGIN IMMEDIATE')
            parent = store.db.execute('SELECT owner_session FROM native_session_alias '
                'WHERE profile=? AND session_id=?', (self.scope, parent_session)).fetchone()
            owner = parent[0] if parent is not None else parent_session
            scope = owner_scope(self.scope, owner)
            active = store.db.execute('SELECT turn_id FROM native_requester_job WHERE scope=?',
                                      (scope,)).fetchone()
            if active is None or active[0] != turn_id or not store.foreground_turn(scope, turn_id):
                raise LocalSourceRefused('compaction has no matching live native parent')
            old = store.db.execute('SELECT owner_session,parent_session FROM native_session_alias '
                'WHERE profile=? AND session_id=?', (self.scope, child_session)).fetchone()
            if old is not None and old != (owner, parent_session):
                raise LocalSourceRefused('compaction child already has another owner')
            store.db.execute('INSERT OR IGNORE INTO native_session_alias VALUES(?,?,?,?)',
                             (self.scope, child_session, owner, parent_session))
            store.db.execute('COMMIT')
        except BaseException:
            store.db.execute('ROLLBACK')
            raise
        finally:
            store.close()

    @contextmanager
    def tool_scope(self,session_id,turn_id,api_request_id,tool_call_id):
        session_id = self.owner_session(session_id)
        ids=(owner_scope(self.scope,session_id),turn_id,api_request_id,tool_call_id)
        store=IdentityStore(self.path)
        try:
            known=all(isinstance(value,str) and value for value in ids) and store.db.execute(
                'SELECT 1 FROM native_tool_call WHERE scope=? AND turn_id=? AND api_request_id=? AND tool_call_id=?',
                ids).fetchone() is not None
        finally:store.close()
        owner = None
        if known and self.job_continuity:
            from agent.relay_runtime import active_turn
            turn = active_turn()
            if (turn is not None and turn.turn_id == turn_id and turn.lease.profile_key == self.scope
                    and self.owner_session(turn.lease.session_id) == session_id):
                owner = turn
        token=self.tool.set((session_id,ids,owner) if known else None)
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
        session,ids,owner=bound
        directory=Path(self.path).parent/'tool-sources'
        store=IdentityStore(self.path)
        try:
            job_turn = ''
            handoff_parents = None
            if self.job_continuity:
                parents = store.db.execute('SELECT DISTINCT c.job_key FROM native_tool_call t '
                    'JOIN call_identity c ON c.key=t.call_key WHERE t.scope=? AND t.turn_id=? '
                    'AND t.api_request_id=? AND t.tool_call_id=?', ids).fetchall()
                if len(parents) != 1:
                    raise LocalSourceRefused('tool has no unambiguous frozen parent linkage')
                handoff_parents = tuple(row[0] for row in store.db.execute(
                    'SELECT DISTINCT t.grant_key FROM native_tool_call t '
                    'JOIN work_identity w ON w.key=t.grant_key WHERE t.scope=? AND t.turn_id=? '
                    "AND t.api_request_id=? AND t.tool_call_id=? AND w.boundary LIKE 'conversation-stint:%'",
                    ids))
                if parents[0][0] is not None:
                    bound = store.db.execute('SELECT job_key FROM native_job_binding WHERE scope=? AND turn_id=?',
                                             ids[:2]).fetchone()
                    if bound is None or bound[0] != parents[0][0]:
                        raise LocalSourceRefused('tool parent does not match its exact native job')
                    job_turn = ids[1]
            store.db.execute('BEGIN IMMEDIATE')
            try:
                live = True
                if self.job_continuity:
                    from agent.relay_runtime import active_turn
                    live = owner is not None and active_turn() is owner
                    if store.db.execute('SELECT 1 FROM native_turn_end WHERE scope=? AND turn_id=?', ids[:2]).fetchone():
                        live = False
                    if job_turn:
                        active = store.db.execute('SELECT job_key,turn_id FROM native_requester_job WHERE scope=?',
                                                  (ids[0],)).fetchone()
                        live = live and active == (parents[0][0], ids[1])
                if not live and call.get('operation'):
                    # An inherited API intention names its exact completed step;
                    # never recover by choosing the latest similar tool result.
                    row=store.db.execute('SELECT operation FROM tool_aux_intent WHERE scope=? AND turn_id=? '
                        'AND api_request_id=? AND tool_call_id=? AND operation=?',
                        (*ids,call['operation'])).fetchone()
                    if row is None:
                        raise LocalSourceRefused('saved tool intention no longer has its exact operation')
                else:
                    row=store.db.execute("SELECT operation FROM tool_aux_intent WHERE scope=? AND turn_id=? "
                        "AND api_request_id=? AND tool_call_id=? AND state='pending'",ids).fetchone()
                if row:
                    source=CompactionSource.load(directory,self.scope,session,row[0])
                    source.validate(request.get('messages'))
                else:
                    if not live:
                        raise LocalSourceRefused('new tool inference requires its live originating native turn')
                    source=CompactionSource.capture(directory,self.scope,session,request.get('messages'),
                                                    purpose='tool_auxiliary')
                    step=store.db.execute('SELECT COALESCE(MAX(step),0)+1 FROM tool_aux_intent WHERE '
                        'scope=? AND turn_id=? AND api_request_id=? AND tool_call_id=?',ids).fetchone()[0]
                    store.db.execute("INSERT INTO tool_aux_intent VALUES(?,?,?,?,?,?,'pending')",
                                     (*ids,step,source.operation))
                call_boundary = 'operation:' + source.operation
                source.messages()
                observation_only = not live
                if observation_only:
                    keys = store.saved_execution(source.scope,source.purpose,call_boundary,request)
                else:
                    if self.job_continuity and active_turn() is not owner:
                        raise LocalSourceRefused('native tool owner ended before identity commit')
                    keys = store.bind_execution_in_transaction(source.scope,source.purpose,call_boundary,
                                                               request,job_turn=job_turn)
                store.db.execute('COMMIT')
            except BaseException:
                store.db.execute('ROLLBACK');raise
            call['operation']=source.operation
            if observation_only:
                return keys
            keys=store.bind_handoff(keys,store.conversation_stints(source.scope)
                                    if handoff_parents is None else handoff_parents)
            if keys.job_key is not None:
                keys=replace(keys,on_accepted=intake_callback(self.path,source.scope,ids[1],keys))
            return keys
        finally:store.close()

    def pre_api_request(self, *, session_id, conversation_history, turn_id='', api_request_id='', **unused):
        # Native observer hooks are fail-open. A rejected new snapshot must not
        # leave the previous request's keys usable by the custom provider.
        self.main.set(None)
        self.main.set(HeldSource.capture(self.scope, self.owner_session(session_id), 'main', conversation_history,
                                        turn_id=turn_id, api_request_id=api_request_id))

    def post_api_request(self, **unused):
        self.main.set(None)

    def begin_foreground(self,session_id,turn_id, *, is_current=None):
        session_id = self.owner_session(session_id)
        store=IdentityStore(self.path)
        try:
            scope = owner_scope(self.scope,session_id)
            if self.job_continuity:
                store.begin_native_job(scope, turn_id, is_current=is_current)
            else:
                store.foreground_turn(scope,turn_id,record=True)
        finally:
            store.close()

    def detached_source(self,session_id,turn_id,messages):
        """Native detached forks skip pre_llm_call and own no SessionDB rows.

        Require a known owning conversation and no positive foreground marker.
        An ordinary foreground persistence failure must still refuse, never
        silently switch to detached execution. Payload is private and durable.
        """
        session_id = self.owner_session(session_id)
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
        session_id = self.owner_session(session_id)
        store = IdentityStore(self.path)
        try:
            return store.finish_turn(owner_scope(self.scope, session_id), turn_id)
        finally:
            store.close()

    def observe_native_outcome(self, session_id, turn_id, *, completed, failed, interrupted):
        session_id = self.owner_session(session_id)
        store = IdentityStore(self.path)
        try:
            scope = owner_scope(self.scope, session_id)
            if not store.foreground_turn(scope, turn_id):
                return 0  # independent background/fork work does not end a parent
            return store.observe_native_outcome(scope, turn_id, completed=completed,
                                               failed=failed, interrupted=interrupted)
        finally:
            store.close()

    def pending_event_highwater(self):
        store = IdentityStore(self.path)
        try:
            return store.pending_event_highwater()
        finally:
            store.close()

    def pending_events(self, after=0):
        store = IdentityStore(self.path)
        try:
            return store.pending_events(after=after)
        finally:
            store.close()

    def dependency_wait(self, session_id, turn_id, api_request_id, tool_call_id):
        session_id = self.owner_session(session_id)
        store = IdentityStore(self.path)
        try:
            return store.dependency_wait(owner_scope(self.scope,session_id),
                                         turn_id,api_request_id,tool_call_id)
        finally:
            store.close()

    def complete_api(self, session_id, turn_id, api_request_id, tool_call_ids):
        session_id = self.owner_session(session_id)
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
        session_id = self.owner_session(session_id)
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
        session_id = self.owner_session(session_id)
        source = CompactionSource.load(Path(self.path).parent / 'compaction-sources',
            self.scope, session_id, operation)
        token = self.compaction_resume.set(source)
        try:
            yield source.messages()
        finally:
            self.compaction_resume.reset(token)

    def _keys(self, source, request, *, job_turn=''):
        if source is None:
            raise LocalSourceRefused('no owning persisted call boundary is bound')
        return source.bind(self.path, request, job_turn=job_turn) if job_turn else source.bind(self.path, request)

    def _main_job_turn(self, source):
        if not self.job_continuity or not isinstance(source, HeldSource) or not source.turn_id:
            return ''
        store = IdentityStore(self.path)
        try:
            if not store.foreground_turn(source.scope, source.turn_id):
                return ''
            if store.db.execute('SELECT 1 FROM native_job_binding WHERE scope=? AND turn_id=?',
                                (source.scope, source.turn_id)).fetchone() is None:
                raise LocalSourceRefused('foreground job has no owning native binding')
            return source.turn_id
        finally:
            store.close()

    def main_keys(self, request):
        source=self.main.get()
        keys=self._keys(source, request, job_turn=self._main_job_turn(source))
        if self.job_continuity and isinstance(source, HeldSource) and source.turn_id:
            keys=replace(keys,on_accepted=intake_callback(self.path,source.scope,source.turn_id,keys,
                                                         source.api_request_id))
        return replace(keys,execution_only=True) if isinstance(source,CompactionSource) else keys

    def main_stint_keys(self, request):
        source = self.main.get()
        if isinstance(source,CompactionSource) and source.purpose=='detached_fork':
            return replace(source.bind(self.path,request),execution_only=True)
        if not isinstance(source, HeldSource):
            raise LocalSourceRefused('no owning foreground source is bound for a stint')
        job_turn = self._main_job_turn(source)
        if self.job_continuity and not job_turn:
            return replace(source.bind(self.path,request),execution_only=True)
        return source.bind_stint(self.path, request, job_turn=job_turn)

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
                job_turn = ''
                if self.job_continuity and owner:
                    if store.begin_native_job(source.scope, owner[0],
                            is_current=lambda: owner_provider() == owner) is None:
                        raise LocalSourceRefused('foreground compaction owner already finished')
                    job_turn = owner[0]
                keys = store.bind_compaction(source.scope, boundary, request,
                    stint_model=owner[1] if owner else '',job_turn=job_turn)
                if keys.work_session or (keys.job_key is not None and owner):
                    if keys.job_key is not None:
                        keys=store.bind_handoff(keys,())
                    return replace(keys, on_accepted=intake_callback(self.path, source.scope, owner[0], keys))
            else:
                keys = self._keys(source, request)
            parents = () if self.job_continuity else store.conversation_stints(source.scope)
            return store.bind_handoff(keys, parents)
        finally:
            store.close()
