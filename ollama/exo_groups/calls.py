"""Private inference provenance in the existing queue journal, never a scheduler.

EXO assigns command IDs inside its chat handler. Commit the exact execution,
instance and wire request before POST; bind the command on first observation.
An unknown POST outcome has no automatic replay path, even with zero output.
"""

from dataclasses import asdict
import json
import re
import time

from ..inference_arbiter import Arbiter, Conflict, NotFound, encoded
from .instance import InstanceContract
from .intents import GroupIntents
from .plan import GroupPlan, HOST_RESOURCES
from .prepare import digest


class ExecutionBindings:
    def __init__(self, intents: GroupIntents):
        if not isinstance(intents, GroupIntents) or not isinstance(intents.journal, Arbiter):
            raise TypeError("existing group intents and inference journal required")
        self.intents, self.journal = intents, intents.journal
        with self.journal.transaction():
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_private_instances (
                owner TEXT NOT NULL, request_key TEXT NOT NULL,
                contract_json TEXT NOT NULL, contract_sha256 TEXT NOT NULL,
                instance_id TEXT UNIQUE NOT NULL,
                PRIMARY KEY(owner,request_key),
                FOREIGN KEY(owner,request_key) REFERENCES exo_group_intents(owner,request_key)
            )''')
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_execution_bindings (
                execution_id TEXT PRIMARY KEY REFERENCES executions(id),
                owner TEXT NOT NULL,
                request_key TEXT NOT NULL,
                contract_json TEXT NOT NULL,
                contract_sha256 TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                request_sha256 TEXT NOT NULL,
                command_id TEXT UNIQUE,
                created REAL NOT NULL,
                state_event_idx INTEGER,
                end_json TEXT,
                FOREIGN KEY(owner,request_key) REFERENCES exo_group_intents(owner,request_key)
            )''')
            columns = {row['name'] for row in self.journal.db.execute(
                'PRAGMA table_info(exo_execution_bindings)')}
            for name, declaration in (('state_event_idx', 'INTEGER'), ('end_json', 'TEXT')):
                if name not in columns:
                    self.journal.db.execute(f'ALTER TABLE exo_execution_bindings ADD COLUMN {name} {declaration}')

    def bind_instance(self, owner: str, request_key: str, contract: InstanceContract) -> None:
        """Freeze provider configuration before private instance creation.

        This idempotent metadata write is not permission to POST /instance;
        the startup owner still needs an explicit once-only launch decision.
        """
        if not isinstance(contract, InstanceContract):
            raise TypeError("provider-prepared instance contract required")
        with self.journal.transaction():
            saved = self.intents.snapshot(owner, request_key)
            plan = GroupPlan.parse(saved['plan'])
            if (saved['group_id'] != contract.group_id or saved['generation'] != contract.generation
                    or saved['launches'] is None or type(contract.generation) is not int
                    or contract.plan_sha256 != digest(asdict(plan))):
                raise Conflict("private instance differs from the frozen group intent")
            value = encoded(asdict(contract))
            previous = self.journal.db.execute('SELECT contract_json FROM exo_private_instances '
                'WHERE owner=? AND request_key=?', (owner, request_key)).fetchone()
            if previous is not None:
                if previous['contract_json'] != value:
                    raise Conflict("private instance configuration already frozen")
                return
            if self.journal.db.execute('SELECT 1 FROM exo_private_instances WHERE instance_id=?',
                                        (contract.instance_id,)).fetchone():
                raise Conflict("private instance already belongs to another group intent")
            self.journal.db.execute('INSERT INTO exo_private_instances VALUES(?,?,?,?,?)',
                (owner, request_key, value, contract.sha256, contract.instance_id))

    def begin(self, execution_id: str, contract: InstanceContract, body: dict,
              *, state_event_idx: int) -> dict:
        """Consume one POST decision while the existing supervisor owns dispatch.

        Repeated calls refuse, including after crash or lost first response.
        This is provenance only; the adapter must separately verify current
        authority, private state and host receipts before calling this method.
        """
        if not isinstance(contract, InstanceContract):
            raise TypeError("provider-prepared instance contract required")
        if type(state_event_idx) is not int or state_event_idx < -1:
            raise Conflict("exact pre-dispatch EXO state index required")
        journal = self.journal
        with journal.transaction():
            row = journal.db.execute('''SELECT x.state,x.request,g.model,g.resource,
                g.active_execution,d.owner,d.request_key FROM executions x
                JOIN grants g ON g.id=x.grant_id
                JOIN exo_group_demands d ON d.execution_id=x.id WHERE x.id=?''',
                (execution_id,)).fetchone()
            if row is None:
                raise NotFound()
            if (row['state'] != 'running' or row['active_execution'] != execution_id
                    or row['resource'] != HOST_RESOURCES['slice']):
                raise Conflict("private POST requires the existing dispatched execution")
            saved = self.intents.snapshot(row['owner'], row['request_key'])
            plan = GroupPlan.parse(saved['plan'])
            instance = journal.db.execute('SELECT contract_json FROM exo_private_instances '
                'WHERE owner=? AND request_key=?', (row['owner'], row['request_key'])).fetchone()
            if (saved['group_id'] != contract.group_id or saved['generation'] != contract.generation
                    or saved['launches'] is None or type(contract.generation) is not int
                    or instance is None or instance['contract_json'] != encoded(asdict(contract))
                    or contract.plan_sha256 != digest(asdict(plan)) or row['model'] != plan.model_id):
                raise Conflict("private instance differs from the frozen group intent")
            request = json.loads(row['request'])
            if (request.get('operation') != 'chat' or not isinstance(request.get('body'), dict)
                    or request['body'].get('model') != plan.model_id
                    or encoded(body) != encoded(request['body'])):
                raise Conflict("private wire body differs from the saved inference request")
            if journal.db.execute('SELECT 1 FROM exo_execution_bindings WHERE execution_id=?',
                                  (execution_id,)).fetchone():
                raise Conflict("private inference submission already owned; do not replay")
            journal.db.execute('''INSERT INTO exo_execution_bindings
                (execution_id,owner,request_key,contract_json,contract_sha256,instance_id,
                 request_sha256,created,state_event_idx) VALUES(?,?,?,?,?,?,?,?,?)''',
                (execution_id, row['owner'], row['request_key'], encoded(asdict(contract)),
                 contract.sha256, contract.instance_id, digest(body), time.time(), state_event_idx))
        return self.snapshot(execution_id)

    def snapshot(self, execution_id: str) -> dict:
        row = self.journal.db.execute('SELECT * FROM exo_execution_bindings WHERE execution_id=?',
                                      (execution_id,)).fetchone()
        if row is None:
            raise NotFound()
        result = dict(row)
        result['contract'] = json.loads(result.pop('contract_json'))
        result['end'] = json.loads(result.pop('end_json') or 'null')
        return result

    def completed_snapshot(self, execution_id: str, *, group_owner: str,
                           request_key: str, contract: InstanceContract) -> dict:
        """Observe this group's completed call without impersonating its caller.

        The group's canonical plane owner and the grant's authenticated queue
        username have different meanings. Join the original demand and binding;
        never translate either identity or bypass the public queue's privacy.
        """
        if not isinstance(contract, InstanceContract):
            raise Conflict('original private instance contract required')
        row = self.journal.db.execute('''SELECT b.*,x.state AS call_state,
            x.request AS call_request,g.model AS call_model,g.resource AS call_resource,
            i.plan_json AS group_plan FROM exo_execution_bindings b
            JOIN executions x ON x.id=b.execution_id
            JOIN grants g ON g.id=x.grant_id
            JOIN exo_group_demands d ON d.execution_id=b.execution_id
                AND d.owner=b.owner AND d.request_key=b.request_key
            JOIN exo_group_intents i ON i.owner=b.owner AND i.request_key=b.request_key
            WHERE b.execution_id=? AND b.owner=? AND b.request_key=?''',
            (execution_id, group_owner, request_key)).fetchone()
        if row is None:
            raise NotFound()
        result = dict(row)
        plan = GroupPlan.parse(json.loads(result.pop('group_plan')))
        state, body = result.pop('call_state'), json.loads(result.pop('call_request'))
        model, resource = result.pop('call_model'), result.pop('call_resource')
        if (state != 'complete' or model != plan.model_id or resource != HOST_RESOURCES['slice']
                or contract.plan_sha256 != digest(asdict(plan))
                or result['contract_json'] != encoded(asdict(contract))
                or result['contract_sha256'] != contract.sha256
                or result['instance_id'] != contract.instance_id
                or not isinstance(body, dict) or body.get('operation') != 'chat'
                or not isinstance(body.get('body'), dict)
                or body['body'].get('model') != plan.model_id
                or digest(body['body']) != result['request_sha256']):
            raise Conflict('completed call differs from its original group binding')
        result['contract'] = json.loads(result.pop('contract_json'))
        result['end'] = json.loads(result.pop('end_json') or 'null')
        return result

    def record_end(self, execution_id: str, proof: dict) -> dict:
        """Save exact all-rank runtime evidence; never finish or release here."""
        with self.journal.transaction():
            saved = self.snapshot(execution_id)
            contract = InstanceContract(**saved['contract'])
            instance = contract._instance
            if (not isinstance(proof, dict) or set(proof) != {
                    'command_id', 'instance_id', 'task_id', 'runner_ids', 'end_event_idx',
                    'group_id', 'generation', 'rank_manifest_sha256'}
                    or saved['command_id'] is None or proof['command_id'] != saved['command_id']
                    or proof['instance_id'] != saved['instance_id']
                    or proof['group_id'] != contract.group_id
                    or type(proof['generation']) is not int or proof['generation'] != contract.generation
                    or proof['rank_manifest_sha256'] != GroupPlan.parse(
                        self.intents.snapshot(saved['owner'], saved['request_key'])['plan']).rank_manifest_sha256
                    or type(proof['end_event_idx']) is not int
                    or type(saved['state_event_idx']) is not int
                    or proof['end_event_idx'] <= saved['state_event_idx']
                    or not isinstance(proof['task_id'], str) or not proof['task_id']
                    or proof['runner_ids'] != sorted(instance['shardAssignments']['runnerToShard'])):
                raise Conflict("private end evidence differs from the submitted execution")
            if saved['end'] is not None and encoded(saved['end']) != encoded(proof):
                raise Conflict("private end evidence already frozen")
            self.journal.db.execute('UPDATE exo_execution_bindings SET end_json=? WHERE execution_id=?',
                                    (encoded(proof), execution_id))
        return self.snapshot(execution_id)

    def observe_command(self, execution_id: str, command_id: str) -> dict:
        """Bind EXO's actual UUID once, before publishing any of its output.

        A late observation may improve uncertain-call provenance. It cannot
        complete the execution, authorize another POST or free physical HOLD.
        """
        if (not isinstance(command_id, str)
                or not re.fullmatch(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', command_id)):
            raise Conflict("private response lacks its exact EXO command UUID")
        with self.journal.transaction():
            saved = self.snapshot(execution_id)
            if saved['command_id'] not in (None, command_id):
                raise Conflict("private response changed command identity")
            other = self.journal.db.execute('SELECT execution_id FROM exo_execution_bindings '
                'WHERE command_id=?', (command_id,)).fetchone()
            if other is not None and other['execution_id'] != execution_id:
                raise Conflict("private command already belongs to another execution")
            self.journal.db.execute('UPDATE exo_execution_bindings SET command_id=? WHERE execution_id=?',
                                    (command_id, execution_id))
        return self.snapshot(execution_id)
