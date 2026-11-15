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
                generation INTEGER NOT NULL CHECK(generation=1),
                created REAL NOT NULL,
                PRIMARY KEY(owner,request_key)
            )''')
            journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_group_demands (
                execution_id TEXT PRIMARY KEY REFERENCES executions(id),
                owner TEXT NOT NULL,
                request_key TEXT NOT NULL,
                FOREIGN KEY(owner,request_key)
                    REFERENCES exo_group_intents(owner,request_key)
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
        return result

    def create_request(self, owner: str, request_key: str) -> dict | None:
        """Same server-idempotent body after lost reply, only for queued demand.

        With a known group id the coordinator reads that exact group instead.
        An uncertain/running/terminal call alone does not initiate acquisition.
        The actor still verifies its identity and serializes release/cancel
        against acquisition; this method performs no plane mutation.
        """
        saved = self.snapshot(owner, request_key)
        if saved['group_id'] is not None:
            return None
        pending = self.journal.db.execute('''SELECT 1 FROM exo_group_demands d
            JOIN executions x ON x.id=d.execution_id JOIN grants g ON g.id=x.grant_id
            WHERE d.owner=? AND d.request_key=? AND x.state='queued'
            AND g.state IN ('queued','granted') AND g.release_requested=0 LIMIT 1''',
            (owner, request_key)).fetchone()
        return saved['request'] if pending is not None else None

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
