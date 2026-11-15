"""Group-create identity linked to requests in the existing inference journal.

These are ownership/provenance rows, not another scheduler. Only the existing
arbiter orders and dispatches inference. No HTTP, lease operation or authority
is implied by a saved intent or group id.
"""

from dataclasses import asdict
import json
import re
import time

from ..inference_arbiter import Arbiter, Conflict, NotFound, encoded
from .plan import GroupPlan, HOST_RESOURCES, check_rank_manifest
from .member import ExpectedMember


class GroupIntents:
    def __init__(self, journal: Arbiter):
        if not isinstance(journal, Arbiter):
            raise TypeError('existing inference Arbiter required')
        self.journal = journal
        with journal.transaction():
            journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_group_intents (
                owner TEXT NOT NULL,
                request_key TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                request_json TEXT NOT NULL,
                group_id TEXT UNIQUE,
                create_started INTEGER NOT NULL DEFAULT 0 CHECK(create_started IN (0,1)),
                launch_json TEXT,
                renewal_json TEXT,
                generation INTEGER NOT NULL CHECK(generation=1),
                created REAL NOT NULL,
                PRIMARY KEY(owner,request_key)
            )''')
            columns = {row['name'] for row in journal.db.execute('PRAGMA table_info(exo_group_intents)')}
            if (not {'create_started', 'launch_json'} <= columns
                    and journal.db.execute('SELECT 1 FROM exo_group_intents LIMIT 1').fetchone()):
                # The unreleased draft format did not record whether create was
                # submitted. Never guess that an old intent was output-free or
                # unsubmitted and silently reacquire after upgrading it.
                raise Conflict('legacy EXO intents require explicit reconciliation')
            for name, declaration in (
                    ('create_started', 'INTEGER NOT NULL DEFAULT 0 CHECK(create_started IN (0,1))'),
                    ('launch_json', 'TEXT'), ('renewal_json', 'TEXT')):
                if name not in columns:
                    journal.db.execute(f'ALTER TABLE exo_group_intents ADD COLUMN {name} {declaration}')
            journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_group_demands (
                execution_id TEXT PRIMARY KEY REFERENCES executions(id),
                owner TEXT NOT NULL,
                request_key TEXT NOT NULL,
                FOREIGN KEY(owner,request_key)
                    REFERENCES exo_group_intents(owner,request_key)
            )''')
            journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_group_staging (
                owner TEXT NOT NULL, request_key TEXT NOT NULL,
                policy_sha256 TEXT NOT NULL, runtime_sha256 TEXT NOT NULL,
                proof_json TEXT NOT NULL, observed REAL NOT NULL,
                PRIMARY KEY(owner,request_key),
                FOREIGN KEY(owner,request_key) REFERENCES exo_group_intents(owner,request_key)
            )''')
            journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_group_closures (
                owner TEXT NOT NULL, request_key TEXT NOT NULL,
                generation INTEGER NOT NULL CHECK(generation>0), created REAL NOT NULL,
                PRIMARY KEY(owner,request_key,generation),
                FOREIGN KEY(owner,request_key) REFERENCES exo_group_intents(owner,request_key)
            )''')

    def prepare(self, execution_id: str, plan: GroupPlan, manifest: bytes) -> dict:
        """Commit the exact create body before a coordinator can submit it.

        A provider-owned plan may be shared by several saved calls. Clients
        cannot select the placement through their inference request. A new
        demand must be queued on Slice; saved terminal/uncertain links remain
        readable for reconciliation and never become new compute permission.
        """
        if not isinstance(plan, GroupPlan):
            raise TypeError('provider-owned GroupPlan required')
        plan_json = encoded(asdict(plan))
        checked = GroupPlan.parse(json.loads(plan_json))
        check_rank_manifest(checked, manifest)
        request_json = encoded(checked.request())
        journal = self.journal
        with journal.transaction():
            execution = journal.db.execute('''SELECT x.state,g.model,g.resource,
                g.release_requested FROM executions x JOIN grants g ON g.id=x.grant_id
                WHERE x.id=?''', (execution_id,)).fetchone()
            if execution is None:
                raise NotFound()
            linked = journal.db.execute('SELECT owner,request_key FROM exo_group_demands '
                'WHERE execution_id=?', (execution_id,)).fetchone()
            if linked is not None and tuple(linked) != (plan.owner_id, plan.request_key):
                raise Conflict('execution already names another EXO group intent')
            if linked is None and (
                    execution['state'] != 'queued' or execution['release_requested']
                    or execution['model'] != plan.model_id
                    or execution['resource'] != HOST_RESOURCES['slice']):
                raise Conflict('only a queued exact Slice model call can add group demand')
            if linked is None and self.closing(plan.owner_id, plan.request_key):
                raise Conflict('group session is closing; no new demand or reacquisition')
            existing = journal.db.execute('SELECT * FROM exo_group_intents '
                'WHERE owner=? AND request_key=?', (plan.owner_id, plan.request_key)).fetchone()
            if existing is not None:
                if (existing['plan_json'], existing['request_json']) != (plan_json, request_json):
                    raise Conflict('group request key already names another pinned placement')
            else:
                journal.db.execute('''INSERT INTO exo_group_intents
                    (owner,request_key,plan_json,request_json,generation,created)
                    VALUES(?,?,?,?,1,?)''',
                    (plan.owner_id, plan.request_key, plan_json, request_json, time.time()))
            if linked is None:
                journal.db.execute('INSERT INTO exo_group_demands VALUES(?,?,?)',
                    (execution_id, plan.owner_id, plan.request_key))
        return self.snapshot(plan.owner_id, plan.request_key)

    def snapshot(self, owner: str, request_key: str) -> dict:
        row = self.journal.db.execute('SELECT * FROM exo_group_intents '
            'WHERE owner=? AND request_key=?', (owner, request_key)).fetchone()
        if row is None:
            raise NotFound()
        result = dict(row)
        # A database observation is provenance, never saved launch authority.
        result['plan'] = json.loads(result.pop('plan_json'))
        result['request'] = json.loads(result.pop('request_json'))
        result['launches'] = json.loads(result.pop('launch_json') or 'null')
        result['renewal_policy'] = json.loads(result.pop('renewal_json') or 'null')
        return result

    def bind_renewal_policy(self, owner, request_key, renew_before_s):
        """Record explicit enabled/disabled policy; never silently change it."""
        policy = {'renew_before_s': renew_before_s}
        with self.journal.transaction():
            saved = self.snapshot(owner, request_key)
            if saved['renewal_policy'] is not None and saved['renewal_policy'] != policy:
                raise Conflict('renewal policy differs from the recorded provider control')
            self.journal.db.execute('UPDATE exo_group_intents SET renewal_json=? '
                'WHERE owner=? AND request_key=?', (encoded(policy), owner, request_key))
        return self.snapshot(owner, request_key)

    def create_request(self, owner: str, request_key: str) -> dict | None:
        """Same server-idempotent body after lost reply, only for queued demand.

        With a known group id the coordinator reads that exact group instead.
        An uncertain/running/terminal call alone does not initiate acquisition.
        An already-committed submission decision outlives caller cancellation:
        its same-key metadata operation must be reconciled, not forgotten.
        The actor still verifies its identity and serializes release/cancel
        against acquisition; this method performs no plane mutation.
        """
        saved = self.snapshot(owner, request_key)
        if saved['group_id'] is not None:
            return None
        if saved['create_started']:
            return saved['request']
        if self.closing(owner, request_key):
            return None
        pending = self.journal.db.execute('''SELECT 1 FROM exo_group_demands d
            JOIN executions x ON x.id=d.execution_id JOIN grants g ON g.id=x.grant_id
            WHERE d.owner=? AND d.request_key=? AND x.state='queued'
            AND g.state IN ('queued','granted') AND g.release_requested=0 LIMIT 1''',
            (owner, request_key)).fetchone()
        return saved['request'] if pending is not None else None

    def closing(self, owner, request_key):
        return self.journal.db.execute('SELECT 1 FROM exo_group_closures '
            'WHERE owner=? AND request_key=? LIMIT 1', (owner, request_key)).fetchone() is not None

    def begin_drain(self, owner, request_key, generation):
        """Freeze session closure before any authority withdrawal or plane write.

        Caller release still owns queued cancellation. Running/uncertain work
        must drain or be recovered; closure cannot turn it into clean work.
        An in-flight create remains owned and is recovered by its original key.
        """
        if type(generation) is not int or generation < 1:
            raise Conflict('explicit positive coordinator generation required')
        with self.journal.transaction():
            self.snapshot(owner, request_key)
            pending = self.journal.db.execute('''SELECT 1 FROM exo_group_demands d
                JOIN executions x ON x.id=d.execution_id WHERE d.owner=? AND d.request_key=?
                AND x.state IN ('queued','running','uncertain') LIMIT 1''',
                (owner, request_key)).fetchone()
            if pending is not None:
                raise Conflict('group calls must end or be explicitly cancelled before closure')
            self.journal.db.execute('INSERT OR IGNORE INTO exo_group_closures VALUES(?,?,?,?)',
                (owner, request_key, generation, time.time()))

    def begin_create(self, owner: str, request_key: str) -> dict | None:
        """Commit service ownership before sending the exact create request."""
        with self.journal.transaction():
            request = self.create_request(owner, request_key)
            if request is not None:
                if self.staging(owner, request_key) is None:
                    raise Conflict('verified staging must precede group submission')
                self.journal.db.execute('UPDATE exo_group_intents SET create_started=1 '
                    'WHERE owner=? AND request_key=?', (owner, request_key))
            return request

    def staging(self, owner, request_key):
        row = self.journal.db.execute('SELECT * FROM exo_group_staging '
            'WHERE owner=? AND request_key=?', (owner, request_key)).fetchone()
        return dict(row) if row is not None else None

    def record_staging(self, owner, request_key, proof):
        from .staging import VerifiedStaging
        if not isinstance(proof, VerifiedStaging):
            raise Conflict('verified host staging proof required')
        with self.journal.transaction():
            saved = self.snapshot(owner, request_key)
            proof.check(GroupPlan.parse(saved['plan']))
            previous = self.staging(owner, request_key)
            if previous is not None and (previous['policy_sha256'], previous['runtime_sha256']) != (
                    proof.policy_sha256, proof.runtime_sha256):
                raise Conflict('staging policy differs from the frozen group intent')
            if previous is not None:
                old = VerifiedStaging(**json.loads(previous['proof_json']))
                if old.boots != proof.boots:
                    raise Conflict('staging host boot differs from the frozen group intent')
            self.journal.db.execute('INSERT OR REPLACE INTO exo_group_staging VALUES(?,?,?,?,?,?)',
                (owner, request_key, proof.policy_sha256, proof.runtime_sha256,
                 encoded(asdict(proof)), time.time()))
        return self.staging(owner, request_key)

    def bind_launches(self, owner: str, request_key: str, launches: tuple) -> dict:
        """Freeze member identities before rank startup; expiry may be renewed."""
        with self.journal.transaction():
            saved = self.snapshot(owner, request_key)
            plan = GroupPlan.parse(saved['plan'])
            expected = tuple(ExpectedMember.from_launch(plan, binding) for binding in launches)
            if (saved['group_id'] is None or len(expected) != 3
                    or tuple(item.binding.pipeline_rank for item in expected) != (0, 1, 2)
                    or len({item.binding.lease_id for item in expected}) != 3
                    or any(item.binding.group_id != saved['group_id']
                        or type(item.binding.pipeline_rank) is not int
                        or type(item.binding.generation) is not int
                        or item.binding.generation != saved['generation'] for item in expected)):
                raise Conflict('launches differ from the recorded group generation')
            identity = [{key: value for key, value in asdict(item.binding).items()
                         if key != 'expires_at'} for item in expected]
            # Normalize tuple layer ranges through the JSON wire representation.
            identity = json.loads(encoded(identity))
            if saved['launches'] is not None and saved['launches'] != identity:
                raise Conflict('group member identities changed after binding')
            self.journal.db.execute('UPDATE exo_group_intents SET launch_json=? '
                'WHERE owner=? AND request_key=?', (encoded(identity), owner, request_key))
        return self.snapshot(owner, request_key)

    def record_group(self, owner: str, request_key: str, group: dict) -> dict:
        """Bind the one reply identity; never adopt a successor generation."""
        with self.journal.transaction():
            saved = self.snapshot(owner, request_key)
            plan = GroupPlan.parse(saved['plan'])
            group_id = group.get('id')
            if (group.get('owner_id') != owner or group.get('request_key') != request_key
                    or group.get('intent_sha256') != plan.intent_sha256
                    or type(group.get('generation')) is not int or group['generation'] < 1
                    or not isinstance(group_id, str)
                    or not re.fullmatch(r'group_[a-f0-9]{24}', group_id)):
                raise Conflict('group reply differs from the durable provider intent')
            if saved['group_id'] not in (None, group_id):
                raise Conflict('provider intent already names another group')
            self.journal.db.execute('UPDATE exo_group_intents SET group_id=? '
                'WHERE owner=? AND request_key=?', (group_id, owner, request_key))
            # Generation remains the original 1 even if a lost create reply is
            # recovered after a fence. Fresh authority will refuse the old actor.
        return self.snapshot(owner, request_key)
