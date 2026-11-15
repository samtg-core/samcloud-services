"""Real inference journal durability; no registry, provider, process or GPU."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from . import test_exo_group_plan as fixtures
from .exo_groups.intents import GroupIntents
from .exo_groups.plan import BindingRefused, HOST_RESOURCES
from .inference_arbiter import Arbiter, Conflict, NotFound


class ProviderGroupIntents(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'existing-queue.sqlite'
        self.journal = Arbiter(self.path)
        self.intents = GroupIntents(self.journal)
        fixture = fixtures.GroupBindingControls()
        fixture.setUp()
        self.plan, self.manifest = fixture.plan, fixture.manifest
        self.group = dict(fixture.group, id='group_' + 'a' * 24)

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def enqueue(self, key='call', *, model=None, resource=None):
        grant = self.journal.submit('caller', key,
            resource or HOST_RESOURCES['slice'], model or self.plan.model_id,
            lifetime='execution')
        call = self.journal.enqueue(grant['id'], 'caller', key,
            {'operation': 'chat', 'body': {'model': self.plan.model_id}})
        return grant, call

    def prepare(self, call, plan=None):
        return self.intents.prepare(call['id'], plan or self.plan, self.manifest)

    def test_saved_request_survives_reopen_before_any_plane_submission(self):
        _, call = self.enqueue()
        saved = self.prepare(call)
        self.assertEqual(saved['request'], self.plan.request())
        self.assertIsNone(saved['group_id'])
        self.journal.close()
        self.journal = Arbiter(self.path)
        self.intents = GroupIntents(self.journal)
        self.assertEqual(self.intents.create_request(self.plan.owner_id,
            self.plan.request_key), self.plan.request())
        self.assertEqual(self.prepare(call), saved)
        # No second scheduler or separate ownership database was opened.
        self.assertEqual(self.intents.journal.path, self.path)
        self.assertEqual(self.journal.execution(call['id'], 'caller')['state'], 'queued')

    def test_two_queued_calls_share_one_exact_group_intent(self):
        _, first = self.enqueue('first')
        _, second = self.enqueue('second')
        self.assertEqual(self.prepare(first), self.prepare(second))
        self.assertEqual(self.journal.db.execute('SELECT COUNT(*) FROM exo_group_intents').fetchone()[0], 1)
        self.assertEqual(self.journal.db.execute('SELECT COUNT(*) FROM exo_group_demands').fetchone()[0], 2)

    def test_changed_key_or_placement_cannot_rebind_existing_demand(self):
        _, call = self.enqueue()
        self.prepare(call)
        for changed in (replace(self.plan, request_key='replacement'),
                replace(self.plan, owner_id='user:another'),
                replace(self.plan, members=tuple(replace(m, memory_mb=1001) for m in self.plan.members))):
            with self.assertRaises(Conflict):
                self.prepare(call, changed)
        self.assertEqual(self.intents.snapshot(self.plan.owner_id,
            self.plan.request_key)['request'], self.plan.request())

    def test_cancelled_or_uncertain_work_does_not_initiate_new_acquisition(self):
        grant, call = self.enqueue()
        self.prepare(call)
        self.journal.request_release(grant['id'], 'caller')
        self.assertIsNone(self.intents.create_request(self.plan.owner_id, self.plan.request_key))
        _, other = self.enqueue('other')
        self.prepare(other)
        active = self.journal.dispatch_next(HOST_RESOURCES['slice'])
        self.assertEqual(active['execution']['id'], other['id'])
        self.journal.uncertain(other['id'])
        self.assertIsNone(self.intents.create_request(self.plan.owner_id, self.plan.request_key))
        self.assertEqual(self.prepare(other)['request'], self.plan.request())
        self.assertEqual(self.journal.execution(other['id'], 'caller')['state'], 'uncertain')

    def test_known_group_reconciles_without_another_create_or_generation_adoption(self):
        _, call = self.enqueue()
        self.prepare(call)
        saved = self.intents.record_group(self.plan.owner_id, self.plan.request_key,
            dict(self.group, generation=2, phase='releasing'))
        self.assertEqual(saved['generation'], 1)
        self.assertEqual(saved['group_id'], self.group['id'])
        self.assertIsNone(self.intents.create_request(self.plan.owner_id, self.plan.request_key))
        with self.assertRaises(Conflict):
            self.intents.record_group(self.plan.owner_id, self.plan.request_key,
                dict(self.group, id='group_' + 'b' * 24))

    def test_wrong_owner_intent_or_boolean_generation_cannot_bind_reply(self):
        _, call = self.enqueue()
        self.prepare(call)
        for bad in (dict(self.group, owner_id='foreign'),
                dict(self.group, intent_sha256='f' * 64),
                dict(self.group, generation=True), dict(self.group, id='unbound')):
            with self.assertRaises(Conflict):
                self.intents.record_group(self.plan.owner_id, self.plan.request_key, bad)
        self.assertIsNone(self.intents.snapshot(self.plan.owner_id, self.plan.request_key)['group_id'])

    def test_owned_create_decision_survives_caller_cancellation_and_restart(self):
        grant, call = self.enqueue()
        self.prepare(call)
        body = self.intents.begin_create(self.plan.owner_id, self.plan.request_key)
        self.journal.request_release(grant['id'], 'caller')
        self.journal.close()
        self.journal = Arbiter(self.path)
        self.intents = GroupIntents(self.journal)
        self.assertEqual(self.intents.create_request(self.plan.owner_id, self.plan.request_key), body)
        self.assertTrue(self.intents.snapshot(self.plan.owner_id, self.plan.request_key)['create_started'])
        self.assertEqual(self.journal.execution(call['id'], 'caller')['state'], 'cancelled')

    def test_bound_member_identity_survives_renewal_but_not_lease_replacement(self):
        fixture = fixtures.GroupBindingControls()
        fixture.setUp()
        _, call = self.enqueue()
        self.prepare(call)
        launches = tuple(replace(b, group_id=self.group['id']) for b in fixture.bind())
        with self.assertRaises(Conflict):
            self.intents.bind_launches(self.plan.owner_id, self.plan.request_key, launches)
        self.intents.record_group(self.plan.owner_id, self.plan.request_key, self.group)
        first = self.intents.bind_launches(self.plan.owner_id, self.plan.request_key, launches)
        renewed = tuple(replace(b, expires_at='2026-11-14T10:15:00+00:00') for b in launches)
        self.assertEqual(self.intents.bind_launches(self.plan.owner_id,
            self.plan.request_key, renewed)['launches'], first['launches'])
        for bad in (launches[:2], tuple(reversed(launches)),
                launches[:2] + (replace(launches[2], lease_id='replacement'),),
                tuple(replace(b, generation=True) for b in launches)):
            with self.assertRaises(Conflict):
                self.intents.bind_launches(self.plan.owner_id, self.plan.request_key, bad)

    def test_populated_legacy_draft_is_not_guessed_to_be_unsubmitted(self):
        legacy = Arbiter(Path(self.temp.name) / 'legacy.sqlite')
        try:
            legacy.db.execute('''CREATE TABLE exo_group_intents (
                owner TEXT, request_key TEXT, plan_json TEXT, request_json TEXT,
                group_id TEXT, generation INTEGER, created REAL,
                PRIMARY KEY(owner,request_key))''')
            legacy.db.execute("INSERT INTO exo_group_intents VALUES(?,?,?,?,NULL,1,0)",
                ('user:58', 'unknown-create-outcome', '{}', '{}'))
            with self.assertRaisesRegex(Conflict, 'explicit reconciliation'):
                GroupIntents(legacy)
            self.assertEqual(legacy.db.execute('SELECT COUNT(*) FROM exo_group_intents').fetchone()[0], 1)
            self.assertNotIn('create_started', {row['name'] for row in legacy.db.execute(
                'PRAGMA table_info(exo_group_intents)')})
        finally:
            legacy.close()

    def test_new_intent_requires_exact_queued_slice_model_and_manifest(self):
        for key, model, resource in [('wrong-model', 'other', HOST_RESOURCES['slice']),
                ('wrong-host', self.plan.model_id, HOST_RESOURCES['wafer'])]:
            grant, call = self.enqueue(key, model=model, resource=resource)
            with self.assertRaises(Conflict):
                self.prepare(call)
            self.journal.request_release(grant['id'], 'caller')
        _, call = self.enqueue('running')
        self.journal.dispatch_next(HOST_RESOURCES['slice'])
        with self.assertRaises(Conflict):
            self.prepare(call)
        _, queued = self.enqueue('queued')
        with self.assertRaises(BindingRefused):
            self.intents.prepare(queued['id'], self.plan, b'{}')
        with self.assertRaises(NotFound):
            self.intents.prepare('missing', self.plan, self.manifest)


if __name__ == '__main__':
    unittest.main()
