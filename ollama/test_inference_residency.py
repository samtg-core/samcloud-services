"""Durable residency intent, exact receipt linkage and restart uncertainty."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .inference_residency import ResidencyStore, RolloverIntent, Conflict


class ResidencyHandover(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'residency.sqlite'
        self.now = 2_000_000_000
        self.store = self.open()

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def open(self, **overrides):
        options = {'registry_owner':'user:65', 'service_id':'device/model-service',
                   'clock':lambda:self.now, **overrides}
        return ResidencyStore(self.path, **options)

    def at(self, seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

    def receipt(self, **overrides):
        return {'id':'lease-old', 'agent_id':'user:65', 'service_id':'device/model-service',
                'resource_id':'device/gpu', 'memory_mb':17530, 'status':'active', 'exclusive':0,
                'expires_at':self.at(2_000_003_600), 'max_total_s':3600, **overrides}

    def register(self, **overrides):
        return self.store.register('allocation-1', resource='device/gpu', model='qwen',
                                   receipt=self.receipt(**overrides))

    def prepare(self, ttl=3600):
        return self.store.prepare('allocation-1', ttl_seconds=ttl)

    def response(self, **overrides):
        return {'lease_id':'lease-new', 'replaces_lease_id':'lease-old',
                'resource_id':'device/gpu', 'memory_mb':17530, 'status':'active',
                'expires_at':self.at(self.now+3600), 'max_total_s':3600, **overrides}

    def reopen(self):
        self.store.close()
        self.store = self.open()

    def test_intent_is_committed_before_send_and_reopen_reuses_exact_key(self):
        self.register()
        operation = self.prepare()
        saved = self.store.db.execute('SELECT * FROM residency_rollovers').fetchone()
        self.assertEqual(saved['id'], operation.id)
        self.assertEqual(saved['predecessor'], 'lease-old')
        self.assertEqual(saved['state'], 'pending')
        self.reopen()
        self.assertEqual(self.prepare(), operation)
        self.assertFalse(self.store.snapshot('allocation-1')['fresh_receipt'])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_ended_residency_can_reload_with_a_new_exact_receipt(self):
        self.register()
        self.store.retire('allocation-1',self.receipt(status='released'))
        self.assertEqual(self.store.snapshot('allocation-1')['state'],'retired')
        self.reopen()
        result=self.register(id='lease-reload',memory_mb=19000)
        self.assertEqual(result['lease_id'],'lease-reload')
        self.assertTrue(result['fresh_receipt'])
        ended=self.store.db.execute('SELECT lease_id FROM residency_ends').fetchall()
        self.assertEqual([row[0] for row in ended],['lease-old'])
        with self.assertRaises(Conflict):self.register()

    def test_expiry_with_unsettled_charge_cannot_retire_residency(self):
        self.register()
        receipt = self.receipt(status='expired', expiry_reason='ttl_expired', settled_at=None)
        with self.assertRaises(Conflict):
            self.store.retire('allocation-1', receipt)
        self.assertNotEqual(self.store.snapshot('allocation-1')['state'], 'retired')
        self.store.retire('allocation-1', receipt | {'settled_at':self.at(self.now)})
        self.assertEqual(self.store.snapshot('allocation-1')['state'], 'retired')

    def test_retirement_requires_exact_terminal_receipt_and_no_pending_rollover(self):
        self.register()
        for overrides in ({},{'status':'queued'},{'status':'released','id':'other'},
                          {'status':'released','agent_id':'other'},
                          {'status':'released','memory_mb':1}):
            with self.subTest(overrides=overrides),self.assertRaises(Conflict):
                self.store.retire('allocation-1',self.receipt(**overrides))
        self.prepare()
        with self.assertRaises(Conflict):
            self.store.retire('allocation-1',self.receipt(status='released'))
        self.assertEqual(self.store.snapshot('allocation-1')['state'],'handover_pending')

    def test_reopen_does_not_turn_historical_receipt_into_permission_for_new_intent(self):
        self.register()
        self.reopen()
        with self.assertRaises(Conflict):
            self.prepare()
        self.register()  # explicit new trusted registry observation
        self.assertTrue(self.store.snapshot('allocation-1')['fresh_receipt'])
        self.prepare()

    def test_lost_response_and_elapsed_predecessor_do_not_allocate_new_intent(self):
        self.register()
        operation = self.prepare()
        self.now += 100_000  # original expiry says nothing about remote rollover result
        self.reopen()
        self.assertEqual(self.prepare(), operation)
        with self.assertRaises(Conflict):
            self.prepare(1800)
        self.assertEqual(self.store.snapshot('allocation-1')['lease_id'], 'lease-old')
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM residency_rollovers').fetchone()[0], 1)

    def test_linked_active_response_updates_exact_allocation(self):
        self.register()
        operation = self.prepare()
        self.now += 900
        result = self.store.accept(operation,self.response())
        self.assertEqual((result['state'],result['lease_id'],result['pending_id']),
                         ('covered','lease-new',None))
        self.assertTrue(result['fresh_receipt'] and result['changed'] and result['receipt_unexpired'])
        next_operation = self.prepare()
        self.assertEqual(next_operation.predecessor,'lease-new')
        self.assertNotEqual(next_operation.id,operation.id)

    def test_ended_replacement_is_uncovered_not_backend_termination(self):
        for state in ('expired','released'):
            with self.subTest(state=state):
                # Unique allocation/resource fixtures for independent subcases.
                if state == 'released':
                    self.store.db.execute('DELETE FROM residency_rollovers')
                    self.store.db.execute('DELETE FROM allocations')
                self.register()
                operation = self.prepare()
                result = self.store.accept(operation,self.response(status=state))
                self.assertEqual(result['state'],'uncovered')
                self.assertEqual(result['lease_id'],'lease-new')
                self.assertFalse(result['fresh_receipt'])
                with self.assertRaises(Conflict):
                    self.prepare()

    def test_expired_active_response_does_not_establish_coverage(self):
        self.register()
        operation = self.prepare()
        self.now += 10_000
        result = self.store.accept(operation,self.response(expires_at=self.at(2_000_003_600)))
        self.assertEqual(result['state'],'uncovered')
        self.assertFalse(result['receipt_unexpired'])

    def test_mismatched_or_malformed_response_keeps_pending_intent(self):
        self.register()
        operation = self.prepare()
        for bad in ({'replaces_lease_id':'other'}, {'resource_id':'other'},
                    {'memory_mb':20000}, {'memory_mb':True}, {'lease_id':'lease-old'},
                    {'max_total_s':7200}, {'max_total_s':True}, {'status':'queued'},
                    {'expires_at':self.at(self.now+100)}, {'expires_at':None}):
            with self.subTest(bad=bad):
                with self.assertRaises((Conflict,ValueError)):
                    self.store.accept(operation,self.response(**bad))
                self.assertEqual(self.prepare(),operation)
                self.assertEqual(self.store.snapshot('allocation-1')['state'],'handover_pending')

    def test_duplicate_old_response_cannot_overwrite_new_pending_handover(self):
        self.register()
        first = self.prepare()
        self.store.accept(first,self.response())
        second = self.prepare()
        duplicate = self.store.accept(first,self.response())
        self.assertFalse(duplicate['changed'])
        self.assertEqual(duplicate['pending_id'],second.id)
        self.assertFalse(duplicate['fresh_receipt'])

    def test_wrong_operation_object_does_not_apply(self):
        self.register()
        operation = self.prepare()
        for bad in (RolloverIntent(operation.id,'sibling','lease-old',3600),
                    RolloverIntent(operation.id,operation.allocation,'other',3600),
                    RolloverIntent(operation.id,operation.allocation,'lease-old',1800)):
            with self.assertRaises(Conflict):
                self.store.accept(bad,self.response())
        self.assertEqual(self.prepare(),operation)

    def test_bad_ownership_observation_withdraws_freshness_without_deleting_state(self):
        self.register()
        for bad in ({'agent_id':'user:sibling'}, {'service_id':'another-service'},
                    {'resource_id':'other-device'}, {'exclusive':1}, {'status':'expired'}):
            with self.subTest(bad=bad):
                with self.assertRaises(Conflict):
                    self.register(**bad)
                self.assertFalse(self.store.snapshot('allocation-1')['fresh_receipt'])
                self.assertEqual(self.store.snapshot('allocation-1')['lease_id'],'lease-old')
        self.register()
        self.store.forget_confirmation('allocation-1')
        with self.assertRaises(Conflict):
            self.prepare()

    def test_stale_predecessor_confirmation_cannot_erase_unknown_handover(self):
        self.register()
        operation = self.prepare()
        self.register()
        self.assertFalse(self.store.snapshot('allocation-1')['fresh_receipt'])
        self.assertEqual(self.prepare(),operation)

    def test_store_identity_is_fixed_across_reopen(self):
        self.register()
        for bad in ({'registry_owner':'user:other'}, {'service_id':'other-service'}):
            with self.assertRaises(PermissionError):
                self.open(**bad)
        self.assertTrue(self.store.snapshot('allocation-1')['fresh_receipt'])

    def test_model_or_lease_alias_cannot_create_another_allocation(self):
        self.register()
        for key,model,receipt in [('second','qwen',self.receipt(id='other-lease')),
                                  ('second','other',self.receipt())]:
            with self.assertRaises(Conflict):
                self.store.register(key,resource='device/gpu',model=model,receipt=receipt)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM allocations').fetchone()[0],1)

    def test_sibling_connection_change_does_not_inherit_old_confirmation(self):
        self.register()
        sibling = self.open()
        try:
            sibling.register('allocation-1',resource='device/gpu',model='qwen',receipt=self.receipt())
            operation = sibling.prepare('allocation-1',ttl_seconds=3600)
            sibling.accept(operation,self.response())
            self.assertFalse(self.store.snapshot('allocation-1')['fresh_receipt'])
            with self.assertRaises(Conflict):
                self.prepare()
        finally:
            sibling.close()

    def test_sibling_pending_handover_withdraws_old_confirmation(self):
        self.register()
        sibling = self.open()
        try:
            sibling.register('allocation-1',resource='device/gpu',model='qwen',receipt=self.receipt())
            sibling.prepare('allocation-1',ttl_seconds=3600)
            self.assertFalse(self.store.snapshot('allocation-1')['fresh_receipt'])
        finally:
            sibling.close()

    def test_confirmation_is_bound_to_observed_receipt_not_post_commit_pointer(self):
        self.register()
        sibling = self.open()
        sibling.register('allocation-1',resource='device/gpu',model='qwen',receipt=self.receipt())
        snapshot = self.store.snapshot
        raced = []
        def after_commit(key):
            if not raced:
                raced.append(True)
                operation = sibling.prepare(key,ttl_seconds=3600)
                sibling.accept(operation,self.response())
            return snapshot(key)
        try:
            with patch.object(self.store,'snapshot',side_effect=after_commit):
                result = self.register()
            self.assertEqual(result['lease_id'],'lease-new')
            self.assertFalse(result['fresh_receipt'])
        finally:
            sibling.close()

    def test_public_store_and_symlink_refuse(self):
        self.store.close()
        self.path.chmod(0o644)
        with self.assertRaises(PermissionError):
            self.open()
        self.path.chmod(0o600)
        linked = self.path.with_name('linked.sqlite')
        linked.symlink_to(self.path)
        with self.assertRaises(OSError):
            ResidencyStore(linked,registry_owner='user:65',service_id='device/model-service')
        self.store = self.open()

    def test_incidental_receipt_payload_is_not_persisted(self):
        private = 'synthetic-private-context-not-metadata'
        self.register(note=private)
        operation = self.prepare()
        self.store.accept(operation,self.response(provider_detail=private))
        self.assertNotIn(private.encode(),self.path.read_bytes())


if __name__ == '__main__':
    unittest.main()
