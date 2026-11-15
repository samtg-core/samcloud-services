"""Once-only submission decisions in the real inference SQLite journal."""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

from . import test_exo_group_plan as fixtures
from .exo_groups.calls import ExecutionBindings
from .exo_groups.instance import InstanceContract
from .exo_groups.intents import GroupIntents
from .exo_groups.plan import HOST_RESOURCES
from .exo_groups.prepare import canonical, digest
from .inference_arbiter import Arbiter, Conflict, NotFound


class ExecutionBindingControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'queue.sqlite'
        self.journal = Arbiter(self.path)
        self.intents = GroupIntents(self.journal)
        self.calls = ExecutionBindings(self.intents)
        f = fixtures.GroupBindingControls()
        f.setUp()
        self.plan, self.manifest = f.plan, f.manifest
        self.group = dict(f.group, id='group_' + 'a' * 24)
        self.launches = tuple(replace(b, group_id=self.group['id']) for b in f.bind())
        # Journal controls use an inert contract. Actual EXO wire generation and
        # readiness are independently exercised by the type and instance suites.
        instance_id = digest([self.group['id'], 1, self.plan.rank_manifest_sha256, 'instance'])
        self.contract = InstanceContract(digest(asdict(self.plan)), self.group['id'], 1,
            'e' * 64, canonical({'instance': {'MlxRingInstance': {'instance_id': instance_id}}}).decode(),
            canonical({'MlxRingInstance': {'instanceId': instance_id}}).decode())
        self.body = {'model': self.plan.model_id, 'messages': [{'role': 'user', 'content': 'hello'}],
                     'stream': False}
        self.command = '318b2c23-d63e-4c4f-8dc5-eaef0753d68a'

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def enqueue(self, key='call'):
        grant = self.journal.submit('caller', key, HOST_RESOURCES['slice'], self.plan.model_id,
                                    lifetime='execution')
        call = self.journal.enqueue(grant['id'], 'caller', key,
            {'operation': 'chat', 'body': self.body})
        self.intents.prepare(call['id'], self.plan, self.manifest)
        self.intents.record_group(self.plan.owner_id, self.plan.request_key, self.group)
        self.intents.bind_launches(self.plan.owner_id, self.plan.request_key, self.launches)
        return grant, call

    def begin(self, call):
        return self.calls.begin(call['id'], self.contract, self.body | {'stream': True})

    def bind(self):
        self.calls.bind_instance(self.plan.owner_id, self.plan.request_key, self.contract)

    def test_request_instance_and_no_command_commit_before_post_decision_returns(self):
        _, call = self.enqueue()
        self.bind()
        with self.assertRaises(Conflict):
            self.begin(call)  # queued is not dispatched
        self.journal.dispatch_next(HOST_RESOURCES['slice'])
        saved = self.begin(call)
        self.assertIsNone(saved['command_id'])
        self.assertEqual(saved['instance_id'], self.contract.instance_id)
        self.assertEqual(saved['contract_sha256'], self.contract.sha256)
        self.assertEqual(saved['request_sha256'], digest(self.body | {'stream': True}))
        observer = Arbiter(self.path)
        try:
            self.assertEqual(ExecutionBindings(GroupIntents(observer)).snapshot(call['id']), saved)
        finally:
            observer.close()
        self.assertEqual(self.journal.execution(call['id'], 'caller')['state'], 'running')

    def test_restart_or_lost_first_frame_cannot_make_a_second_submission(self):
        _, call = self.enqueue()
        self.bind()
        self.journal.dispatch_next(HOST_RESOURCES['slice'])
        saved = self.begin(call)
        with self.assertRaises(Conflict):
            self.begin(call)
        self.journal.close()
        self.journal = Arbiter(self.path)
        self.intents = GroupIntents(self.journal)
        self.calls = ExecutionBindings(self.intents)
        self.assertEqual(self.calls.snapshot(call['id']), saved)
        with self.assertRaises(Conflict):
            self.begin(call)
        self.journal.quarantine_running()
        with self.assertRaises(Conflict):
            self.begin(call)
        self.assertIsNone(self.journal.dispatch_next(HOST_RESOURCES['slice']))

    def test_observed_command_binds_once_and_uncertain_late_read_cannot_finish(self):
        _, call = self.enqueue()
        self.bind()
        self.journal.dispatch_next(HOST_RESOURCES['slice'])
        self.begin(call)
        self.journal.uncertain(call['id'])
        saved = self.calls.observe_command(call['id'], self.command)
        self.assertEqual(saved['command_id'], self.command)
        self.assertEqual(self.calls.observe_command(call['id'], self.command), saved)
        with self.assertRaises(Conflict):
            self.calls.observe_command(call['id'], '418b2c23-d63e-4c4f-8dc5-eaef0753d68a')
        self.assertEqual(self.journal.execution(call['id'], 'caller')['state'], 'uncertain')
        self.assertEqual(self.journal.events(call['id'], 'caller')['events'], [])

    def test_changed_request_model_contract_or_missing_instance_binding_refuses(self):
        _, call = self.enqueue()
        self.journal.dispatch_next(HOST_RESOURCES['slice'])
        with self.assertRaises(Conflict):
            self.begin(call)
        self.bind()
        for body in (self.body, self.body | {'stream': True, 'model': 'other'},
                     self.body | {'stream': True, 'messages': []}):
            with self.assertRaises(Conflict):
                self.calls.begin(call['id'], self.contract, body)
        for contract in (replace(self.contract, generation=2),
                         replace(self.contract, runtime_sha256='f' * 64)):
            with self.assertRaises(Conflict):
                self.calls.begin(call['id'], contract, self.body | {'stream': True})
        with self.assertRaises(NotFound):
            self.calls.snapshot(call['id'])

    def test_instance_configuration_cannot_change_between_requests(self):
        self.enqueue()
        self.bind()
        self.bind()
        for contract in (replace(self.contract, generation=True),
                         replace(self.contract, runtime_sha256='d' * 64)):
            with self.assertRaises(Conflict):
                self.calls.bind_instance(self.plan.owner_id, self.plan.request_key, contract)

    def test_two_calls_cannot_claim_same_remote_command(self):
        _, first = self.enqueue('first')
        self.bind()
        self.journal.dispatch_next(HOST_RESOURCES['slice'])
        self.begin(first)
        self.calls.observe_command(first['id'], self.command)
        self.journal.finish(first['id'], {'kind': 'json', 'body': {}})
        _, second = self.enqueue('second')
        self.journal.dispatch_next(HOST_RESOURCES['slice'])
        self.begin(second)
        with self.assertRaises(Conflict):
            self.calls.observe_command(second['id'], self.command)
        for value in ('chatcmpl-' + self.command, '', None, 'secret\ntext'):
            with self.assertRaises(Conflict):
                self.calls.observe_command(second['id'], value)


if __name__ == '__main__':
    unittest.main()
