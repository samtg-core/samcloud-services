"""Once-only private instance creation on an already admitted group.

Host rank launch/staging and cleanup are separate effects. This owner never
replays a create POST after a lost reply, and never treats acceptance as ready.
"""

from dataclasses import asdict
import json
import time
from uuid import UUID

from ..inference_arbiter import Conflict, NotFound, encoded
from .calls import ExecutionBindings
from .instance import InstanceContract
from .plan import BindingRefused, GroupPlan
from .prepare import canonical


class InstanceStarts:
    def __init__(self, bindings):
        if not isinstance(bindings, ExecutionBindings):
            raise TypeError('exact instance/execution bindings required')
        self.bindings, self.journal = bindings, bindings.journal
        with self.journal.transaction():
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_instance_starts (
                owner TEXT NOT NULL, request_key TEXT NOT NULL,
                contract_sha256 TEXT NOT NULL, state_event_idx INTEGER NOT NULL,
                command_id TEXT UNIQUE, created REAL NOT NULL,
                PRIMARY KEY(owner,request_key),
                FOREIGN KEY(owner,request_key) REFERENCES exo_private_instances(owner,request_key)
            )''')

    def snapshot(self, owner, request_key):
        row = self.journal.db.execute('SELECT * FROM exo_instance_starts '
            'WHERE owner=? AND request_key=?', (owner, request_key)).fetchone()
        if row is None:
            raise NotFound()
        return dict(row)

    def begin(self, plan, contract, *, state_event_idx):
        if (not isinstance(plan, GroupPlan) or not isinstance(contract, InstanceContract)
                or type(state_event_idx) is not int or state_event_idx < -1):
            raise Conflict('exact startup plan, contract and baseline required')
        with self.journal.transaction():
            saved = self.bindings.intents.snapshot(plan.owner_id, plan.request_key)
            row = self.journal.db.execute('SELECT contract_sha256 FROM exo_private_instances '
                'WHERE owner=? AND request_key=?', (plan.owner_id, plan.request_key)).fetchone()
            staged = self.bindings.intents.staging(plan.owner_id, plan.request_key)
            if (row is None or row['contract_sha256'] != contract.sha256
                    or saved['plan'] != json.loads(encoded(asdict(plan)))
                    or saved['group_id'] != contract.group_id
                    or saved['generation'] != contract.generation or saved['launches'] is None
                    or staged is None or staged['runtime_sha256'] != contract.runtime_sha256):
                raise Conflict('startup differs from the frozen private instance')
            if self.journal.db.execute('SELECT 1 FROM exo_instance_starts '
                    'WHERE owner=? AND request_key=?', (plan.owner_id, plan.request_key)).fetchone():
                raise Conflict('private instance submission already owned; do not replay')
            self.journal.db.execute('INSERT INTO exo_instance_starts VALUES(?,?,?,?,NULL,?)',
                (plan.owner_id, plan.request_key, contract.sha256, state_event_idx, time.time()))
        return self.snapshot(plan.owner_id, plan.request_key)

    def accept(self, plan, contract, response):
        expected = next(iter(contract._instance['shardAssignments']['runnerToShard'].values()))[
            'PipelineShardMetadata']['modelCard']
        command = response.get('command_id') if isinstance(response, dict) else None
        try:
            valid = isinstance(command, str) and str(UUID(command)) == command
        except ValueError:
            valid = False
        if (not valid or canonical(response.get('model_card')) != canonical(expected)):
            raise Conflict('private instance acceptance differs from the submitted contract')
        with self.journal.transaction():
            saved = self.snapshot(plan.owner_id, plan.request_key)
            if (saved['contract_sha256'] != contract.sha256
                    or saved['command_id'] not in (None, command)):
                raise Conflict('private instance acceptance changed identity')
            self.journal.db.execute('UPDATE exo_instance_starts SET command_id=? '
                'WHERE owner=? AND request_key=?', (command, plan.owner_id, plan.request_key))
        return self.snapshot(plan.owner_id, plan.request_key)


class InstanceStartup:
    def __init__(self, actor, private, starts, contract, launches, verify_runtime):
        if (actor.intents is not starts.bindings.intents or not callable(verify_runtime)
                or len(launches) != 3):
            raise ValueError('exact startup actor, bindings and runtime verifier required')
        self.actor, self.private, self.starts = actor, private, starts
        self.contract, self.launches, self.verify_runtime = contract, launches, verify_runtime

    def _authorized(self, epoch):
        return self.actor._epoch == epoch and self.actor.may_start_ranks()

    async def submit(self):
        self.actor.wake()
        await self.actor.wait()
        epoch = self.actor._epoch
        if not self._authorized(epoch):
            raise BindingRefused('fresh admitted all-member authority required')
        runtime = await self.verify_runtime()
        state = await self.private.state()
        self.contract.check_empty(self.actor.plan, self.launches, state, runtime_sha256=runtime)
        if not self._authorized(epoch):
            raise BindingRefused('startup authority changed before submission')
        return await self.private.create_instance(self.starts, self.actor.plan, self.contract,
            state_event_idx=state['lastEventAppliedIdx'])
