"""Supervisor-owned exact settlement after observed host cleanup.

The existing arbiter journal owns closure, launch pins and immutable evidence
intents. A disconnected HTTP observer cannot cancel these metadata operations.
No inference retry, physical stop, new scheduler or default deadline lives here.
"""

import asyncio
import copy
from dataclasses import asdict
import hashlib
import json

from ..inference_arbiter import Conflict
from .cleanup import CleanupExpectation, _aware, _ended, validate_cleanup
from .member import ExpectedMember
from .plan import BindingRefused, LaunchBinding
from .prepare import canonical
from .rank_process import Listener, ProcessIdentity, RunBinding


class GroupCompletion:
    def __init__(self, actor):
        self.actor, self.intents, self.plane = actor, actor.intents, actor.plane
        self.journal, self.plan = actor.intents.journal, actor.plan
        self._task, self._operation = None, None
        with self.journal.transaction():
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_group_cleanup_pins (
                owner TEXT NOT NULL, request_key TEXT NOT NULL, lease_id TEXT NOT NULL,
                expectation_json TEXT NOT NULL, PRIMARY KEY(owner,request_key,lease_id),
                FOREIGN KEY(owner,request_key) REFERENCES exo_group_intents(owner,request_key)
            )''')
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_group_settlements (
                owner TEXT NOT NULL, request_key TEXT NOT NULL, lease_id TEXT NOT NULL,
                generation INTEGER NOT NULL, intent_json TEXT NOT NULL, confirmed_json TEXT,
                PRIMARY KEY(owner,request_key,lease_id,generation),
                FOREIGN KEY(owner,request_key,lease_id)
                    REFERENCES exo_group_cleanup_pins(owner,request_key,lease_id)
            )''')

    def _saved(self):
        return self.intents.snapshot(self.plan.owner_id, self.plan.request_key)

    def _key(self, lease_id):
        return self.plan.owner_id, self.plan.request_key, lease_id

    def pin_launch(self, pinned):
        """Called from the trusted host-start observation, BEFORE requesting drain.

        Runtime/child/baseline/listeners come from the actual launch owner, never
        from the later cleanup artifact. Reopening preserves the same exact pin.
        """
        if not isinstance(pinned, CleanupExpectation):
            raise BindingRefused('actual launch cleanup expectation required')
        run = pinned.binding
        key = self._key(run.lease_id)
        raw = canonical(asdict(pinned)).decode()
        with self.journal.transaction():
            previous = self.journal.db.execute('SELECT expectation_json FROM exo_group_cleanup_pins '
                'WHERE owner=? AND request_key=? AND lease_id=?', key).fetchone()
            if previous is not None:
                if previous['expectation_json'] != raw:
                    raise Conflict('actual host launch cleanup pin is immutable')
                return
            if self.intents.closing(self.plan.owner_id, self.plan.request_key):
                raise Conflict('a closing session cannot acquire a post-hoc launch pin')
            saved = self._saved()
            staged = self.intents.staging(self.plan.owner_id, self.plan.request_key)
            launch = {k: v for k, v in asdict(run.expected.binding).items() if k != 'expires_at'}
            if (run.group_id != saved['group_id'] or run.generation != saved['generation']
                    or ExpectedMember.from_launch(self.plan, run.expected.binding) != run.expected
                    or saved['launches'] is None or json.loads(canonical(launch)) not in saved['launches']
                    or staged is None or staged['runtime_sha256'] != run.runtime_sha256):
                raise BindingRefused('actual launch differs from the journaled group/runtime')
            self.journal.db.execute('INSERT INTO exo_group_cleanup_pins VALUES(?,?,?,?)', (*key, raw))

    def _pin(self, lease_id):
        row = self.journal.db.execute('SELECT expectation_json FROM exo_group_cleanup_pins '
            'WHERE owner=? AND request_key=? AND lease_id=?', self._key(lease_id)).fetchone()
        if row is None:
            raise BindingRefused('actual pre-drain host launch pin is missing')
        value = json.loads(row['expectation_json'])
        run = value['binding']
        launch = run['expected']['binding']
        launch['layers'] = tuple(launch['layers'])
        expected = ExpectedMember.from_launch(self.plan, LaunchBinding(**launch))
        binding = RunBinding.from_expected(expected, run['runtime_sha256'])
        pinned = CleanupExpectation(binding, ProcessIdentity(**value['child']),
            tuple(Listener(**item) for item in value['listeners']),
            value['baseline_in_use_bytes'], value['tolerance_bytes'])
        if canonical(asdict(pinned)).decode() != row['expectation_json']:
            raise BindingRefused('saved launch cleanup pin differs from its plan')
        return pinned

    async def _owned(self, identity, operation):
        if self._task is not None and not self._task.done():
            if self._operation != identity:
                raise Conflict('another group cleanup operation is still owned')
        else:
            self._operation = identity
            self._task = self.actor.task_factory(operation())
            self._task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(self._task)

    async def wait(self):
        if self._task is not None:
            return await asyncio.shield(self._task)

    async def begin_release(self, *, generation):
        # Synchronous commit before task registration/authority withdrawal.
        self.intents.begin_drain(self.plan.owner_id, self.plan.request_key, generation)
        self.actor.wake()
        return await self._owned(('release', generation), lambda: self._begin(generation))

    async def _group(self, generation):
        saved = self._saved()
        if saved['group_id'] is None:
            raise BindingRefused('group-create outcome requires exact reconciliation')
        group = await self.plane.group(saved['group_id'])
        self.intents.record_group(self.plan.owner_id, self.plan.request_key, group)
        if group['generation'] != generation:
            raise BindingRefused('coordinator generation changed; no write')
        if (group.get('phase') not in ('pending', 'acquiring', 'admitted', 'serving',
                                      'releasing', 'uncertain', 'released', 'cancelled')
                or type(group.get('release_requested')) is not bool):
            raise BindingRefused('group state is unconfirmed; no write')
        members = group.get('members')
        if (not isinstance(members, list) or len(members) != len(self.plan.members)
                or {m.get('resource_id') for m in members if isinstance(m, dict)}
                    != {m.resource_id for m in self.plan.members}):
            raise BindingRefused('group member set differs from the pinned placement')
        return group

    async def _begin(self, generation):
        await self.actor.wait()  # A create already in flight must keep its identity.
        await self.plane.verify_owner()
        saved = self._saved()
        if saved['group_id'] is None:
            if not saved['create_started']:
                return {'state': 'closed_without_submission', 'group_id': None}
            group = await self.plane.create(saved['request'])  # Same idempotent metadata key only.
            self.intents.record_group(self.plan.owner_id, self.plan.request_key, group)
        group = await self._group(generation)
        if not group.get('release_requested') and group['phase'] not in ('released', 'cancelled'):
            await self.plane.begin_release(group['id'], generation)
        fresh = await self._group(generation)
        terminal = await self._terminal(fresh)
        if terminal:
            fresh = await self._group(generation)
            terminal = await self._terminal(fresh)
        self.actor.wake()
        await self.actor.wait()
        return {'state': 'group_settlement_confirmed' if terminal else 'host_cleanup_required',
                'group_id': fresh['id'], 'generation': generation, 'phase': fresh['phase']}

    @staticmethod
    def _member(group, run):
        members = [m for m in group['members'] if m.get('lease_id') == run.lease_id
                   and m.get('resource_id') == run.resource_id]
        if len(members) != 1 or not 1 <= run.generation <= group['generation']:
            raise BindingRefused('exact launch member is absent, replaced or from a future generation')
        return members[0]

    async def _lease(self, run):
        lease = await self.plane.lease(run.lease_id)
        self._lease_dimensions(lease, run.expected.member, run.lease_id)
        return lease

    def _lease_dimensions(self, lease, member, lease_id):
        wanted = {'id': lease_id, 'resource_id': member.resource_id, 'agent_id': self.plan.owner_id,
            'exclusive': True, 'service_id': None, 'priority': 'normal',
            'memory_mb': member.memory_mb, 'ttl_s': member.ttl_seconds, 'max_total_s': member.max_total_s}
        if any(type(lease.get(k)) is not type(v) or lease.get(k) != v for k, v in wanted.items()):
            raise BindingRefused('exact lease ownership or dimensions changed')

    async def settle_member(self, lease_id, *, generation, file_id, sha256):
        # The pin must pre-exist: cleanup bytes cannot supply their own authority.
        self._pin(lease_id)
        self.intents.begin_drain(self.plan.owner_id, self.plan.request_key, generation)
        self.actor.wake()
        return await self._owned(('settle', lease_id, generation, file_id, sha256),
            lambda: self._settle(lease_id, generation, file_id, sha256))

    async def _artifact(self, file_id, sha256, pinned):
        import re
        if (type(file_id) is not int or file_id <= 0 or not isinstance(sha256, str)
                or not re.fullmatch(r'[a-f0-9]{64}', sha256)):
            raise BindingRefused('exact published file id and SHA256 required')
        metadata = await self.plane.artifact(file_id)
        if (type(metadata.get('id')) is not int or metadata['id'] != file_id
                or metadata.get('sha256') != sha256 or type(metadata.get('bytes')) is not int
                or not 0 < metadata['bytes'] <= 128 * 1024):
            raise BindingRefused('published host evidence metadata differs')
        raw = await self.plane.artifact_bytes(file_id)
        if len(raw) != metadata['bytes'] or hashlib.sha256(raw).hexdigest() != sha256:
            raise BindingRefused('published host evidence bytes differ')
        validate_cleanup(raw, pinned)
        return {'evidence_ref': f'https://cloud.samtg.xyz/api/v1/files/{file_id}/raw',
                'evidence_sha256': sha256}

    async def _confirm(self, group, run, proof, generation):
        if group['generation'] != generation:
            raise BindingRefused('generation changed during settlement readback')
        member = self._member(group, run)
        evidence, receipt = member.get('cleanup_evidence'), member.get('release_receipt')
        if (not isinstance(evidence, dict) or not isinstance(receipt, dict)
                or any(evidence.get(k) != v for k, v in proof.items())
                or evidence.get('lease_id') != run.lease_id or receipt.get('lease_id') != run.lease_id):
            raise BindingRefused('exact host evidence and member receipt are not acknowledged')
        lease = await self._lease(run)
        if not _ended(lease) or any(receipt.get(k) != lease.get(k)
                                  for k in ('status', 'released_at', 'settled_at')):
            raise BindingRefused('exact terminal settlement is unconfirmed')
        _aware(lease['released_at'] if lease['status'] == 'released' else lease['settled_at'])
        return {'state': 'exact_member_settlement_confirmed', 'group_id': run.group_id,
            'generation': generation, 'launch_generation': run.generation, 'lease_id': run.lease_id,
            'resource_id': run.resource_id, 'cleanup_evidence': evidence, 'release_receipt': receipt}

    async def _settle(self, lease_id, generation, file_id, sha256):
        await self.actor.wait()
        pinned = self._pin(lease_id)
        run = pinned.binding
        group = await self._group(generation)
        self._member(group, run)
        proof = await self._artifact(file_id, sha256, pinned)
        intent = canonical({'group_id': run.group_id, 'generation': generation, **proof}).decode()
        key = (*self._key(lease_id), generation)
        with self.journal.transaction():
            previous = self.journal.db.execute('SELECT intent_json FROM exo_group_settlements '
                'WHERE owner=? AND request_key=? AND lease_id=? AND generation=?', key).fetchone()
            # Core binds evidence immutably even across a coordinator fence.
            other = self.journal.db.execute('SELECT intent_json FROM exo_group_settlements '
                'WHERE owner=? AND request_key=? AND lease_id=?', key[:3]).fetchall()
            if (previous is not None and previous['intent_json'] != intent
                    or any(any(json.loads(row['intent_json']).get(k) != v for k, v in proof.items())
                           for row in other)):
                raise Conflict('exact member already names different cleanup evidence')
            self.journal.db.execute('INSERT OR IGNORE INTO exo_group_settlements VALUES(?,?,?,?,?,NULL)',
                                    (*key, intent))
        # Re-read after artifact I/O; the server checks the same explicit CAS.
        group = await self._group(generation)
        member = self._member(group, run)
        await self._lease(run)
        if member.get('release_receipt') is None:
            group = await self.plane.release_member(run.group_id, lease_id, generation, proof)
            self.intents.record_group(self.plan.owner_id, self.plan.request_key, group)
        await self._confirm(group, run, proof, generation)
        result = await self._confirm(await self._group(generation), run, proof, generation)
        with self.journal.transaction():
            self.journal.db.execute('UPDATE exo_group_settlements SET confirmed_json=? '
                'WHERE owner=? AND request_key=? AND lease_id=? AND generation=?',
                (canonical(result).decode(), *key))
        self.actor.wake()
        await self.actor.wait()
        return copy.deepcopy(result)

    async def _terminal(self, group):
        if group['phase'] not in ('released', 'cancelled') or not group.get('release_requested'):
            return False
        # Terminal group state is read from the plane, not inferred from local
        # command completion. Every allocated member is independently checked.
        for member in group['members']:
            lease_id = member.get('lease_id')
            if lease_id is None:
                continue
            lease = await self.plane.lease(lease_id)
            plan = next(m for m in self.plan.members if m.resource_id == member['resource_id'])
            self._lease_dimensions(lease, plan, lease_id)
            receipt = member.get('release_receipt')
            if not isinstance(receipt, dict) or receipt.get('lease_id') != lease_id:
                raise BindingRefused('terminal group member receipt unavailable')
            if lease.get('status') == 'cancelled' and lease.get('granted_at') is None:
                if receipt.get('status') != 'cancelled':
                    raise BindingRefused('queued cancellation is unconfirmed')
            else:
                pinned = self._pin(lease_id)
                row = self.journal.db.execute('SELECT intent_json FROM exo_group_settlements '
                    'WHERE owner=? AND request_key=? AND lease_id=? ORDER BY generation DESC LIMIT 1',
                    self._key(lease_id)).fetchone()
                if row is None:
                    raise BindingRefused('terminal group cleanup intent is missing')
                intent = json.loads(row['intent_json'])
                proof = {k: intent[k] for k in ('evidence_ref', 'evidence_sha256')}
                await self._confirm(group, pinned.binding, proof, group['generation'])
        return True
