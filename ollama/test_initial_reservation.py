"""Initial allocation intent survives lost replies; no provider/network live."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .inference_arbiter import CapacityDeferred, Conflict
from .inference_reservation import InitialReservations
from .manager import ModelManager
from . import manager as manager_module


class InitialReservation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'initial.sqlite'
        self.receipt = {'status_code':200, 'status':'active', 'lease_id':'exact',
                        'expires_at':'2099-01-01T00:00:00Z'}
        self.client = SimpleNamespace(request_lease=Mock(return_value=self.receipt))
        self.client.list_leases = Mock(return_value=[self.lease()])
        self.client.release_lease = Mock(return_value={'status':'released'})
        self.store = self.open()

    def lease(self, **changes):
        return {'id':'exact', 'resource_id':'gpu', 'service_id':'service',
                'agent_id':'owner', 'memory_mb':100, 'exclusive':0, 'status':'active'} | changes

    def open(self, **overrides):
        args = {'registry_owner':'owner', 'service_id':'service', 'models':{'qwen':'gpu'}}
        return InitialReservations(self.path, self.client, **(args | overrides))

    def reserve(self, store=None, **overrides):
        args = {'resource':'gpu', 'service_id':'service', 'memory_mb':100, 'ttl_seconds':3600}
        return (store or self.store).reserve('qwen', **(args | overrides))

    def test_intent_is_committed_before_send_and_exact_receipt_before_provider_handoff(self):
        def registry(**kwargs):
            with closing(sqlite3.connect(self.path)) as db:
                self.assertEqual(db.execute('SELECT state,lease_id FROM initial_reservations').fetchone(),
                                 ('unknown',None))
            return self.receipt
        self.client.request_lease.side_effect = registry
        self.assertEqual(self.reserve(), 'exact')
        saved = self.store.snapshot('qwen')
        self.assertEqual(saved['state'], 'handed_off')
        self.assertEqual(saved['lease_id'], 'exact')
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertTrue(saved['intent'])

    def test_handed_off_receipt_is_not_reused_after_reopen_or_failed_load(self):
        self.reserve()
        with self.assertRaises(Conflict):
            self.reserve(self.open())
        self.client.request_lease.assert_called_once()

    def test_lost_reply_and_process_reopen_cannot_allocate_again(self):
        self.client.request_lease.side_effect = TimeoutError('PRIVATE transport detail')
        with self.assertRaises(Conflict) as caught:
            self.reserve()
        saved = self.store.snapshot('qwen')
        self.assertEqual(saved['state'], 'unknown')
        self.assertNotIn('PRIVATE', str(caught.exception))
        self.client.request_lease.side_effect = None
        with self.assertRaises(Conflict):
            self.reserve(self.open())
        self.assertEqual(self.store.snapshot('qwen')['intent'], saved['intent'])
        self.client.request_lease.assert_called_once()

    def test_received_conflict_can_be_retried_after_capacity_event(self):
        self.client.request_lease.return_value = {'status_code':409}
        with self.assertRaises(CapacityDeferred):
            self.reserve()
        first = self.store.snapshot('qwen')['intent']
        self.assertEqual(self.store.snapshot('qwen')['state'], 'refused')
        self.client.request_lease.return_value = self.receipt
        self.assertEqual(self.reserve(), 'exact')
        self.assertNotEqual(first, self.store.snapshot('qwen')['intent'])
        self.assertEqual(self.client.request_lease.call_count, 2)

    def test_queued_identifier_is_preserved_never_reposted_or_treated_as_granted(self):
        self.client.request_lease.return_value = {'status_code':200,'status':'queued','lease_id':'queued'}
        with self.assertRaises(CapacityDeferred):
            self.reserve()
        self.assertEqual(self.store.snapshot('qwen')['lease_id'], 'queued')
        self.client.list_leases.return_value = [self.lease(id='queued', status='queued')]
        with self.assertRaises(CapacityDeferred):
            self.reserve(self.open())
        self.client.request_lease.assert_called_once()

    def test_manager_consumes_exact_promoted_row_without_cancel_or_new_post(self):
        manager = object.__new__(ModelManager)
        manager.durable_inference = True
        manager.initial_reservations, manager.models = self.store, {}
        manager.ollama = SimpleNamespace(list_running=Mock(return_value=[]))
        self.client.request_lease.return_value = {'status_code':200, 'status':'queued', 'lease_id':'exact'}
        self.client.list_leases.return_value = [self.lease(status='queued')]
        with patch.object(manager_module,'RESOURCE_ID','gpu'), \
             patch.object(manager_module,'SERVICE_ID','service'), \
             patch.object(manager_module,'LEASE_TTL',3600):
            with self.assertRaises(CapacityDeferred):
                manager._request_lease('qwen',100,before_load=True)
            self.client.release_lease.assert_not_called()
            with self.assertRaises(CapacityDeferred):
                manager._request_lease('qwen',100,before_load=True)
            self.client.list_leases.return_value = [self.lease(expires_at='2099-01-01T00:00:00Z')]
            self.assertEqual(manager._request_lease('qwen',100,before_load=True), 'exact')
        self.client.release_lease.assert_not_called()
        self.client.request_lease.assert_called_once()

    def test_stage_holds_metadata_grant_without_consuming_then_checks_before_load(self):
        args = dict(resource='gpu', service_id='service', memory_mb=100, ttl_seconds=3600)
        self.assertEqual(self.store.stage('qwen', **args), 'exact')
        self.assertEqual(self.store.snapshot('qwen')['state'], 'reserved')
        self.client.list_leases.return_value = [self.lease(expires_at='2099-01-01T00:00:00Z')]
        self.assertEqual(self.open().stage('qwen', **args), 'exact')
        self.assertEqual(self.store.snapshot('qwen')['state'], 'reserved')
        self.assertEqual(self.reserve(), 'exact')
        self.assertEqual(self.store.snapshot('qwen')['state'], 'handed_off')
        with self.assertRaises(Conflict):
            self.reserve()
        self.client.request_lease.assert_called_once()
        self.client.release_lease.assert_not_called()

    def test_vlm_load_consumes_staged_exact_row_before_spawn(self):
        """The VLM path must not buy a second claim beside its staged one."""
        manager = object.__new__(ModelManager)
        manager.durable_inference = True
        manager.initial_reservations, manager.models = self.store, {}
        manager.sc = self.client
        manager.refuse_if_device_in_use = Mock()
        manager._kill_stray_vlm = Mock()
        args = dict(resource='gpu', service_id='service', memory_mb=100, ttl_seconds=3600)
        self.store.stage('qwen', **args)
        self.client.list_leases.return_value = [self.lease(expires_at='2099-01-01T00:00:00Z')]
        class SpawnBoundary(Exception):
            pass
        def spawn(*args, **kwargs):
            self.assertEqual(self.store.snapshot('qwen')['state'], 'handed_off')
            self.assertEqual(self.store.snapshot('qwen')['lease_id'], 'exact')
            self.client.request_lease.assert_called_once()
            raise SpawnBoundary()
        with patch.object(manager_module, 'RESOURCE_ID', 'gpu'), \
             patch.object(manager_module, 'SERVICE_ID', 'service'), \
             patch.object(manager_module, 'LEASE_TTL', 3600), \
             patch('ollama.inference_shared_resource.cold_vision_absent', return_value=True), \
             patch.object(manager_module, 'match_vlm_model', return_value=('qwen',100)), \
             patch.object(manager_module.subprocess, 'Popen', side_effect=spawn):
            with self.assertRaises(SpawnBoundary):
                manager.load_vlm_model('vision-alias')

    def test_staged_queue_preserves_position_through_promotion_and_process_reopen(self):
        args = dict(resource='gpu', service_id='service', memory_mb=100, ttl_seconds=3600)
        self.client.request_lease.return_value = {'status_code':200,'status':'queued','lease_id':'exact'}
        with self.assertRaises(CapacityDeferred):
            self.store.stage('qwen', **args)
        saved = self.store.snapshot('qwen')
        self.client.list_leases.return_value = [self.lease(status='queued')]
        with self.assertRaises(CapacityDeferred):
            self.open().stage('qwen', **args)
        self.client.list_leases.return_value = [self.lease(expires_at='2099-01-01T00:00:00Z')]
        self.assertEqual(self.open().stage('qwen', **args), 'exact')
        self.assertEqual(self.store.snapshot('qwen')['intent'], saved['intent'])
        self.assertEqual(self.store.snapshot('qwen')['state'], 'reserved')
        self.assertEqual(self.reserve(self.open()), 'exact')
        self.assertEqual(self.store.snapshot('qwen')['state'], 'handed_off')
        self.client.release_lease.assert_not_called()
        self.client.request_lease.assert_called_once()

    def test_staged_grant_expiry_or_lost_read_does_not_authorize_load_or_new_post(self):
        args = dict(resource='gpu', service_id='service', memory_mb=100, ttl_seconds=3600)
        self.store.stage('qwen', **args)
        for change in ({'status':'expired'}, {'status':'cancelled'},
                       {'expires_at':'2000-01-01T00:00:00Z'}):
            self.client.list_leases.return_value = [self.lease(expires_at='2099-01-01T00:00:00Z') | change]
            with self.assertRaises(Conflict):
                self.reserve(self.open())
            self.assertEqual(self.store.snapshot('qwen')['state'], 'reserved')
        self.client.list_leases.side_effect = TimeoutError('synthetic')
        with self.assertRaises(Conflict):
            self.reserve(self.open())
        self.client.request_lease.assert_called_once()
        self.client.release_lease.assert_not_called()

    def test_promoted_receipt_unknown_or_wrong_cannot_load_cancel_or_repost(self):
        self.client.request_lease.return_value = {'status_code':200,'status':'queued','lease_id':'exact'}
        with self.assertRaises(CapacityDeferred):
            self.reserve()
        grant = self.lease(expires_at='2099-01-01T00:00:00Z')
        invalid = [[], [grant,grant], [grant|{'id':'other'}]]
        invalid += [[grant|change] for change in (
            {'agent_id':'sibling'}, {'service_id':'other'}, {'resource_id':'other'},
            {'exclusive':1}, {'memory_mb':True}, {'memory_mb':101},
            {'status':'cancelled'}, {'status':'released'}, {'status':'expired'},
            {'expires_at':None}, {'expires_at':'2000-01-01T00:00:00Z'},
            {'expires_at':'2099-01-01T00:00:00'})]
        for rows in invalid:
            with self.subTest(rows=rows):
                self.client.list_leases.return_value = rows
                with self.assertRaises(Conflict):
                    self.reserve(self.open())
                self.assertEqual(self.store.snapshot('qwen')['state'], 'queued')
        self.client.list_leases.side_effect = TimeoutError('PRIVATE')
        with self.assertRaises(Conflict) as caught:
            self.reserve()
        self.assertNotIn('PRIVATE', str(caught.exception))
        self.client.request_lease.assert_called_once()
        self.client.release_lease.assert_not_called()

    def test_explicit_absent_reconciliation_accepts_cancelled_queue_row(self):
        self.client.request_lease.return_value = {'status_code':200,'status':'queued','lease_id':'exact'}
        with self.assertRaises(CapacityDeferred):
            self.reserve()
        self.client.list_leases.return_value = [self.lease(status='queued')]
        self.client.release_lease.return_value = {'status':'cancelled'}
        self.assertEqual(self.store.reconcile_absent('qwen', confirm_absent=lambda _:True)['state'], 'retired')
        self.client.release_lease.assert_called_once_with('exact')

    def test_cancelled_row_is_not_a_new_grant_without_absence_reconciliation(self):
        self.client.request_lease.return_value = {'status_code':200,'status':'queued','lease_id':'exact'}
        with self.assertRaises(CapacityDeferred):
            self.reserve()
        self.client.list_leases.return_value = [self.lease(status='cancelled')]
        with self.assertRaises(Conflict):
            self.reserve()
        self.assertEqual(self.store.reconcile_absent('qwen', confirm_absent=lambda _:True)['state'], 'retired')
        self.client.release_lease.assert_not_called()

    def test_expired_hold_settles_only_after_fresh_provider_absence(self):
        self.reserve()
        self.client.list_leases.return_value = [self.lease(status='expired',
            expiry_reason='ttl_expired', settled_at=None)]
        self.client.release_lease.return_value = {'status':'settled'}
        with self.assertRaises(Conflict):
            self.store.reconcile_absent('qwen', confirm_absent=lambda _:False)
        self.client.release_lease.assert_not_called()
        absent = Mock(return_value=True)
        self.store.reconcile_absent('qwen', confirm_absent=absent)
        self.assertEqual(absent.call_count, 2)
        self.client.release_lease.assert_called_once_with('exact')
        self.assertEqual(self.store.snapshot('qwen')['state'], 'retired')

    def test_lost_settlement_reply_reopens_same_expired_row_without_new_post(self):
        self.reserve()
        expired = self.lease(status='expired', expiry_reason='ttl_expired', settled_at=None)
        self.client.list_leases.return_value = [expired]
        self.client.release_lease.side_effect = TimeoutError('synthetic lost settlement reply')
        with self.assertRaises(Conflict):
            self.store.reconcile_absent('qwen', confirm_absent=lambda _:True)
        self.assertEqual(self.store.snapshot('qwen')['state'], 'retiring')
        with self.assertRaises(Conflict):
            self.reserve(self.open())
        self.client.list_leases.return_value = [expired | {'settled_at':'2099-01-01T00:00:00Z'}]
        self.open().reconcile_absent('qwen', confirm_absent=lambda _:True)
        self.client.release_lease.assert_called_once_with('exact')
        self.client.request_lease.assert_called_once()

    def test_invalid_receipts_never_authorize_provider_load_or_retry(self):
        for change in ({'expires_at':None}, {'expires_at':'2000-01-01T00:00:00Z'},
                       {'expires_at':'2099-01-01T00:00:00'}, {'status':'released'},
                       {'lease_id':None}, {'status_code':503}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                self.path = Path(folder)/'initial.sqlite'
                store = self.open()
                self.client.request_lease.return_value = self.receipt | change
                with self.assertRaises(Conflict):
                    self.reserve(store)
                with self.assertRaises(Conflict):
                    self.reserve(store)

    def test_other_identity_and_changed_dimensions_refuse_without_post(self):
        with self.assertRaises(PermissionError):
            self.open(registry_owner='sibling')
        for change in ({'resource':'elsewhere'}, {'service_id':'other'}, {'memory_mb':True},
                       {'ttl_seconds':0}):
            with self.assertRaises(ValueError):
                self.reserve(**change)
        self.client.request_lease.assert_not_called()
        self.reserve()
        with self.assertRaises(Conflict):
            self.reserve(memory_mb=200)
        self.client.request_lease.assert_called_once()

    def test_competing_owner_cannot_send_through_same_journal(self):
        with self.store._owned():
            with self.assertRaises(BlockingIOError):
                self.reserve()
        self.client.request_lease.assert_not_called()

    def test_manager_uses_bound_controller_before_load_not_for_post_load_resize(self):
        manager = object.__new__(ModelManager)
        manager.sc = self.client
        manager.durable_inference = True
        manager.initial_reservations = self.store
        with patch.object(manager_module,'RESOURCE_ID','gpu'), \
             patch.object(manager_module,'SERVICE_ID','service'), \
             patch.object(manager_module,'LEASE_TTL',3600):
            self.assertEqual(manager._request_lease('qwen',100,before_load=True), 'exact')
        self.assertEqual(self.store.snapshot('qwen')['state'], 'handed_off')
        self.client.request_lease.assert_called_once()
        with patch.object(self.store, 'reserve', side_effect=AssertionError('post-load must not reuse initial intent')):
            self.assertEqual(manager._request_lease('qwen',100), 'exact')
        self.assertEqual(self.client.request_lease.call_count, 2)

    def test_real_cold_load_saves_receipt_before_provider_and_cannot_repeat_after_lost_load_reply(self):
        manager = object.__new__(ModelManager)
        manager.sc, manager.models = self.client, {}
        manager.durable_inference = True
        manager.initial_reservations = self.store
        def load(*args, **kwargs):
            saved = self.store.snapshot('qwen')
            self.assertEqual((saved['state'], saved['lease_id']), ('handed_off','exact'))
            raise TimeoutError('synthetic unknown load outcome')
        manager.ollama = SimpleNamespace(memory_estimate_mb=Mock(return_value=100),
            list_models=Mock(return_value=[{'name':'qwen'}]),load_model=Mock(side_effect=load))
        with patch.object(manager_module,'RESOURCE_ID','gpu'), \
             patch.object(manager_module,'SERVICE_ID','service'), \
             patch.object(manager_module,'LEASE_TTL',3600), \
             patch.object(manager,'refuse_if_device_in_use'), \
             patch.object(manager_module.capacity,'collect',return_value={'memory_available_mb':10000}), \
             patch.object(manager_module.capacity,'fits',return_value=True):
            with self.assertRaises(TimeoutError):
                manager.load_ollama_model('qwen')
            manager.initial_reservations = self.open()  # journal reopen is not permission
            with self.assertRaises(Conflict):
                manager.load_ollama_model('qwen')
        manager.ollama.load_model.assert_called_once()
        self.client.request_lease.assert_called_once()
        self.assertEqual(manager.models, {})

    def test_reconcile_exact_absent_provider_retires_before_new_allocation(self):
        self.reserve()
        absent = Mock(return_value=True)
        self.assertEqual(self.store.reconcile_absent('qwen', confirm_absent=absent),
                         {'lease_id':'exact', 'state':'retired'})
        self.assertEqual(absent.call_count, 2)
        self.client.release_lease.assert_called_once_with('exact')
        self.client.list_leases.assert_called_once_with(resource_id='gpu', service_id='service')
        self.client.request_lease.return_value = self.receipt | {'lease_id':'next'}
        self.assertEqual(self.reserve(), 'next')
        self.assertEqual(self.client.request_lease.call_count, 2)

    def test_unknown_initial_id_cannot_be_guessed_from_one_matching_registry_lease(self):
        self.client.request_lease.side_effect = TimeoutError('synthetic')
        with self.assertRaises(Conflict):
            self.reserve()
        absent = Mock(return_value=True)
        with self.assertRaises(Conflict):
            self.store.reconcile_absent('qwen', confirm_absent=absent)
        absent.assert_not_called()
        self.client.list_leases.assert_not_called()
        self.client.release_lease.assert_not_called()

    def test_missing_changed_or_duplicated_lease_identity_never_releases(self):
        self.reserve()
        for rows in ([], [self.lease(id='other')], [self.lease(agent_id='sibling')],
                     [self.lease(service_id='other')], [self.lease(resource_id='other')],
                     [self.lease(memory_mb=101)], [self.lease(exclusive=1)],
                     [self.lease(status='unknown')], [self.lease(),self.lease()]):
            with self.subTest(rows=rows):
                self.client.list_leases.return_value = rows
                with self.assertRaises(Conflict):
                    self.store.reconcile_absent('qwen', confirm_absent=lambda model:True)
                self.client.release_lease.assert_not_called()
                self.assertEqual(self.store.snapshot('qwen')['state'], 'handed_off')

    def test_absence_unknown_or_changed_during_read_cannot_release(self):
        self.reserve()
        for callback in (Mock(return_value=False), Mock(return_value=None),
                         Mock(side_effect=TimeoutError('PRIVATE')), Mock(side_effect=[True,False])):
            with self.subTest(callback=callback):
                with self.assertRaises(Conflict) as caught:
                    self.store.reconcile_absent('qwen', confirm_absent=callback)
                self.assertNotIn('PRIVATE', str(caught.exception))
                self.client.release_lease.assert_not_called()

    def test_lost_release_reply_reopens_and_reads_same_id_without_double_allocation(self):
        self.reserve()
        self.client.release_lease.side_effect = TimeoutError('PRIVATE')
        with self.assertRaises(Conflict):
            self.store.reconcile_absent('qwen', confirm_absent=lambda model:True)
        self.assertEqual(self.store.snapshot('qwen')['state'], 'retiring')
        with self.assertRaises(Conflict):
            self.reserve()
        self.client.list_leases.return_value = [self.lease(status='released')]
        self.assertEqual(self.open().reconcile_absent('qwen', confirm_absent=lambda model:True)['state'], 'retired')
        self.client.release_lease.assert_called_once_with('exact')
        self.client.request_lease.assert_called_once()

    def test_manager_uses_provider_list_not_merely_empty_manager_inventory(self):
        self.reserve()
        manager = object.__new__(ModelManager)
        manager.durable_inference = True
        manager.initial_reservations = self.store
        manager.models = {}
        manager.ollama = SimpleNamespace(list_running=Mock())
        for rows in ([{'name':'qwen:latest'}], None, [{}]):
            manager.ollama.list_running.return_value = rows
            with self.assertRaises(Conflict):
                manager.reconcile_initial_reservation('qwen')
            self.client.release_lease.assert_not_called()
        manager.ollama.list_running.return_value = []
        manager.models['qwen'] = object()
        with self.assertRaises(Conflict):
            manager.reconcile_initial_reservation('qwen')
        manager.models.clear()
        self.assertEqual(manager.reconcile_initial_reservation('qwen')['state'], 'retired')
        self.client.release_lease.assert_called_once_with('exact')

    def test_post_load_shortfall_preserves_manager_lease_without_unjournaled_resize(self):
        manager = object.__new__(ModelManager)
        manager.sc, manager.models = self.client, {}
        manager.durable_inference = True
        manager.initial_reservations = self.open(budgets={'qwen':100})
        manager.ollama = SimpleNamespace(memory_estimate_mb=Mock(return_value=100),
            list_models=Mock(return_value=[{'name':'qwen'}]), load_model=Mock(),
            list_running=Mock(return_value=[{'name':'qwen', 'size':110*1024*1024,
                                            'context_length':65536}]),
            num_ctx_for=Mock(return_value=65536))
        with patch.object(manager_module,'RESOURCE_ID','gpu'), \
             patch.object(manager_module,'SERVICE_ID','service'), \
             patch.object(manager_module,'LEASE_TTL',3600), \
             patch.object(manager,'refuse_if_device_in_use'), \
             patch.object(manager_module.capacity,'collect',return_value={'memory_available_mb':10000}), \
             patch.object(manager_module.capacity,'fits',return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'exceeds confirmed initial reservation'):
                manager.load_ollama_model('qwen')
        self.assertEqual(manager.models['qwen'].lease_id, 'exact')
        self.assertEqual(manager.models['qwen'].memory_mb, 110)
        self.assertEqual(self.store.snapshot('qwen')['lease_id'], 'exact')
        self.assertEqual(self.store.snapshot('qwen')['state'], 'handed_off')
        manager.ollama.load_model.assert_called_once()
        self.client.request_lease.assert_called_once()
        self.client.release_lease.assert_not_called()

    def test_existing_sizing_keeps_threshold_without_a_fixed_budget(self):
        manager = object.__new__(ModelManager)
        manager.durable_inference = True
        manager.initial_reservations = self.store
        self.reserve()
        with patch.object(manager,'_release_lease_quietly') as release:
            for actual in (100,110,125):
                self.assertEqual(manager._reconcile_lease('qwen','exact',100,actual),'exact')
            release.assert_not_called()
        self.client.request_lease.assert_called_once()

    def test_existing_resize_grants_before_predecessor_release(self):
        manager = object.__new__(ModelManager)
        manager.durable_inference = True
        manager.initial_reservations = self.store
        self.reserve()
        order = []
        self.client.request_lease.side_effect = lambda **kw: (order.append(('acquire',kw['memory_mb'])) or
            self.receipt|{'lease_id':'larger'})
        with patch.object(manager_module,'LEASE_TTL',3600), \
             patch.object(manager,'_release_lease_quietly',side_effect=lambda lease:order.append(('release',lease))):
            self.assertEqual(manager._reconcile_lease('qwen','exact',100,150),'larger')
        self.assertEqual(order,[('acquire',150),('release','exact')])

    def test_lost_resize_reply_is_not_reissued_after_process_reopen(self):
        self.reserve()
        self.client.request_lease.side_effect = TimeoutError('PRIVATE')
        for store in (self.store,self.open()):
            with self.assertRaises(Conflict) as caught:
                store.resize('qwen','exact',150,3600)
            self.assertNotIn('PRIVATE',str(caught.exception))
        self.assertEqual(self.client.request_lease.call_count,2)  # initial + one sizing attempt
        self.client.release_lease.assert_not_called()

    def test_confirmed_resize_refusal_keeps_predecessor_and_does_not_repeat(self):
        self.reserve()
        self.client.request_lease.return_value = {'status_code':409}
        self.assertIsNone(self.store.resize('qwen','exact',150,3600))
        self.assertIsNone(self.open().resize('qwen','exact',150,3600))
        self.assertEqual(self.client.request_lease.call_count,2)
        self.client.release_lease.assert_not_called()

    def test_queued_resize_retires_only_successor_not_predecessor(self):
        self.reserve()
        self.client.request_lease.return_value = {'status_code':200,'status':'queued','lease_id':'waiting'}
        self.assertIsNone(self.store.resize('qwen','exact',150,3600))
        self.client.release_lease.assert_called_once_with('waiting')

    def test_queued_resize_lost_release_stays_unknown(self):
        self.reserve()
        self.client.request_lease.return_value = {'status_code':200,'status':'queued','lease_id':'waiting'}
        self.client.release_lease.side_effect = TimeoutError('PRIVATE')
        for store in (self.store,self.open()):
            with self.assertRaises(Conflict):
                store.resize('qwen','exact',150,3600)
        self.client.release_lease.assert_called_once_with('waiting')
        self.assertEqual(self.client.request_lease.call_count,2)

    def test_ordinary_unload_can_be_followed_by_a_fresh_demand_driven_load(self):
        self.reserve()
        manager = object.__new__(ModelManager)
        manager.durable_inference = True
        manager.initial_reservations,manager.models = self.store,{}
        manager.ollama = SimpleNamespace(list_running=Mock(return_value=[]))
        self.client.list_leases.return_value = [self.lease(status='released')]
        self.client.request_lease.return_value = self.receipt|{'lease_id':'fresh'}
        with patch.object(manager_module,'RESOURCE_ID','gpu'), \
             patch.object(manager_module,'SERVICE_ID','service'), \
             patch.object(manager_module,'LEASE_TTL',3600):
            self.assertEqual(manager._request_lease('qwen',100,before_load=True),'fresh')
        self.assertEqual(self.client.request_lease.call_count,2)
        self.assertEqual(self.store.snapshot('qwen')['lease_id'],'fresh')
        self.client.release_lease.assert_not_called()

    def test_lost_resize_without_identity_cannot_be_retired_as_an_empty_residency(self):
        self.reserve()
        self.client.request_lease.side_effect = TimeoutError()
        with self.assertRaises(Conflict):
            self.store.resize('qwen','exact',150,3600)
        with self.assertRaisesRegex(Conflict,'sizing allocation identity unknown'):
            self.open().reconcile_absent('qwen',confirm_absent=lambda model:True)
        self.client.release_lease.assert_not_called()

    def test_unload_reconciliation_retires_exact_sizing_successor_before_initial_row(self):
        self.reserve()
        self.client.request_lease.return_value = self.receipt|{'lease_id':'larger'}
        self.store.resize('qwen','exact',150,3600)
        self.client.list_leases.return_value = [self.lease(status='released'),
                                               self.lease(id='larger',memory_mb=150)]
        self.assertEqual(self.store.reconcile_absent('qwen',confirm_absent=lambda model:True)['state'],'retired')
        self.client.release_lease.assert_called_once_with('larger')

    def test_durable_sizing_does_not_use_legacy_twenty_five_percent_tolerance(self):
        manager = object.__new__(ModelManager)
        manager.durable_inference = True
        with patch.object(manager,'_request_lease') as allocate, \
             patch.object(manager,'_release_lease_quietly') as release:
            self.assertEqual(manager._reconcile_lease('qwen','exact',100,100), 'exact')
            for actual in (101, 126):
                with self.assertRaises(RuntimeError):
                    manager._reconcile_lease('qwen','exact',100,actual)
            allocate.assert_not_called()
            release.assert_not_called()

    def budget_manager(self):
        manager = object.__new__(ModelManager)
        manager.sc, manager.models = self.client, {}
        manager.durable_inference = True
        manager.initial_reservations = self.open(budgets={'qwen':200})
        manager.ollama = SimpleNamespace(memory_estimate_mb=Mock(return_value=100),
            list_models=Mock(return_value=[{'name':'qwen'}]), load_model=Mock(),
            list_running=Mock(return_value=[{'name':'qwen','size':110*1024*1024,'context_length':65536}]),
            num_ctx_for=Mock(return_value=65536))
        return manager

    def test_configured_budget_drives_fit_and_reservation_not_only_weight_estimate(self):
        manager = self.budget_manager()
        with patch.object(manager_module,'RESOURCE_ID','gpu'), \
             patch.object(manager_module,'SERVICE_ID','service'), \
             patch.object(manager_module,'LEASE_TTL',3600), \
             patch.object(manager,'refuse_if_device_in_use'), \
             patch.object(manager_module.capacity,'collect',return_value={'memory_available_mb':1000}), \
             patch.object(manager_module.capacity,'fits',return_value=True) as fits:
            mm = manager.load_ollama_model('qwen')
        self.assertEqual(mm.memory_mb,110)
        self.assertEqual(mm.lease_id,'exact')
        fits.assert_called_once_with(200,1000)
        self.assertEqual(self.client.request_lease.call_args.kwargs['memory_mb'],200)
        self.assertEqual(manager.initial_reservations.snapshot('qwen')['memory_mb'],200)
        self.client.release_lease.assert_not_called()

    def test_larger_post_pull_estimate_rechecks_fit_before_reservation_or_load(self):
        manager = self.budget_manager()
        manager.ollama.memory_estimate_mb.side_effect = [100,400]
        with patch.object(manager,'refuse_if_device_in_use'), \
             patch.object(manager,'catalogue_mb',return_value={}), \
             patch.object(manager_module.capacity,'collect',return_value={'memory_available_mb':1000}), \
             patch.object(manager_module.capacity,'fits',side_effect=[True,False]) as fits:
            with self.assertRaises(manager_module.capacity.InsufficientCapacity):
                manager.load_ollama_model('qwen')
        self.assertEqual([call.args for call in fits.call_args_list],[(200,1000),(400,1000)])
        self.client.request_lease.assert_not_called()
        manager.ollama.load_model.assert_not_called()
        self.assertEqual(manager.models,{})


if __name__ == '__main__':
    unittest.main()
