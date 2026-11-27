"""An enabled durable manager cannot load after an unconfirmed byte grant."""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from .manager import ModelManager
from . import capacity
from .inference_arbiter import CapacityDeferred


class Reservation(unittest.TestCase):
    def setUp(self):
        self.manager = object.__new__(ModelManager)
        self.manager.durable_inference = True
        self.manager.sc = SimpleNamespace(request_lease=Mock())

    def response(self, **extra):
        return {'status_code':200,'status':'active','lease_id':'exact-new-id',
                'expires_at':'2099-01-01T00:00:00+00:00',**extra}

    def request(self, response):
        self.manager.sc.request_lease.return_value = response
        return self.manager._request_lease('qwen',100)

    def test_confirmed_current_finite_grant_returns_only_its_exact_id(self):
        self.assertEqual(self.request(self.response()),'exact-new-id')

    def test_invalid_or_ended_receipt_never_becomes_permission(self):
        for changes in ({'expires_at':None},{'expires_at':'2000-01-01T00:00:00+00:00'},
                        {'expires_at':'2099-01-01T00:00:00'}, {'expires_at':'bad'},
                        {'status':'released'},{'status':None},{'lease_id':None}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                self.request(self.response(**changes))

    def test_queued_reservation_is_released_but_never_used_to_load(self):
        with patch.object(self.manager,'_release_lease_quietly') as release, \
             patch.object(self.manager,'note_lease_contention'):
            with self.assertRaises(RuntimeError):
                self.request(self.response(status='queued',queue_position=1))
            release.assert_called_once_with('exact-new-id')

    def test_unknown_request_outcome_is_not_returned_as_unleased_permission(self):
        self.manager.sc.request_lease.side_effect = TimeoutError('PRIVATE transport body')
        with self.assertRaises(RuntimeError) as caught:
            self.manager._request_lease('qwen',100)
        self.assertNotIn('PRIVATE',str(caught.exception))

    def test_received_conflict_is_capacity_wait_without_an_allocation_to_release(self):
        with patch.object(self.manager, 'note_lease_contention'), \
             patch.object(self.manager, '_release_lease_quietly') as release:
            with self.assertRaises(CapacityDeferred) as caught:
                self.manager.sc.request_lease.return_value = {'status_code':409, 'detail':'resource is held'}
                self.manager._request_lease('qwen', 100, before_load=True)
            self.assertEqual(caught.exception.reason, 'capacity_unavailable')
            release.assert_not_called()
        self.manager.sc.request_lease.assert_called_once()

    def test_post_load_resize_conflict_is_not_a_pre_provider_wait(self):
        with patch.object(self.manager, 'note_lease_contention'):
            with self.assertRaises(RuntimeError):
                self.request({'status_code':409, 'detail':'resource is held'})

    def test_queued_and_error_outcomes_are_not_safe_capacity_retries(self):
        for response in (self.response(status='queued', queue_position=1),
                         {'status_code':503}, {'status_code':200},
                         self.response(status='released')):
            with self.subTest(response=response), \
                 patch.object(self.manager, 'note_lease_contention'), \
                 patch.object(self.manager, '_release_lease_quietly'):
                with self.assertRaises(RuntimeError):
                    self.request(response)

    def test_real_load_path_cannot_start_backend_after_queued_reservation(self):
        self.manager.models = {}
        self.manager.ollama = SimpleNamespace(memory_estimate_mb=Mock(return_value=100),
            list_models=Mock(return_value=[{'name':'qwen'}]),load_model=Mock())
        self.manager.sc.request_lease.return_value = self.response(status='queued',queue_position=1)
        with patch.object(self.manager,'refuse_if_device_in_use'), \
             patch.object(self.manager,'note_lease_contention'), \
             patch.object(self.manager,'_release_lease_quietly'), \
             patch('ollama.manager.capacity.collect',return_value={'memory_available_mb':10000}), \
             patch('ollama.manager.capacity.fits',return_value=True):
            with self.assertRaises(RuntimeError):
                self.manager.load_ollama_model('qwen')
        self.manager.ollama.load_model.assert_not_called()
        self.assertEqual(self.manager.models,{})

    def test_real_fit_refusal_precedes_pull_reservation_and_provider_load(self):
        self.manager.models = {}
        self.manager.ollama = SimpleNamespace(memory_estimate_mb=Mock(return_value=100),
            list_models=Mock(), pull_model=Mock(), load_model=Mock())
        with patch.object(self.manager, 'refuse_if_device_in_use'), \
             patch.object(self.manager, 'catalogue_mb', return_value={}), \
             patch('ollama.manager.capacity.collect', return_value={'memory_available_mb':0}), \
             patch('ollama.manager.capacity.fits', return_value=False):
            with self.assertRaises(capacity.InsufficientCapacity):
                self.manager.load_ollama_model('qwen')
        self.manager.ollama.list_models.assert_not_called()
        self.manager.ollama.pull_model.assert_not_called()
        self.manager.ollama.load_model.assert_not_called()
        self.manager.sc.request_lease.assert_not_called()
        self.assertEqual(self.manager.models, {})

    def test_missing_exact_residency_size_does_not_fall_back_to_estimate(self):
        for running in ([], [{'name':'qwen-other','size':100*1024*1024}],
                        [{'name':'qwen:latest','size':0}]):
            with self.subTest(running=running):
                self.manager.models = {}
                self.manager.ollama = SimpleNamespace(memory_estimate_mb=Mock(return_value=100),
                    list_models=Mock(return_value=[{'name':'qwen'}]), load_model=Mock(),
                    list_running=Mock(return_value=running), num_ctx_for=Mock(return_value=None))
                self.manager.sc.request_lease.return_value = self.response()
                with patch.object(self.manager,'refuse_if_device_in_use'), \
                     patch('ollama.manager.capacity.collect',return_value={'memory_available_mb':10000}), \
                     patch('ollama.manager.capacity.fits',return_value=True):
                    with self.assertRaisesRegex(RuntimeError, 'size is unconfirmed'):
                        self.manager.load_ollama_model('qwen')
                self.assertEqual(self.manager.models['qwen'].lease_id, 'exact-new-id')
                self.manager.ollama.load_model.assert_called_once()

    def test_disabled_legacy_contract_is_unchanged(self):
        self.manager.durable_inference = False
        self.assertEqual(self.request(self.response(expires_at=None)),'exact-new-id')
        self.manager.sc.request_lease.side_effect = TimeoutError('synthetic')
        self.assertIsNone(self.manager._request_lease('qwen',100))


class Shutdown(unittest.TestCase):
    def setUp(self):
        self.manager = object.__new__(ModelManager)
        self.manager.durable_inference = True
        self.manager.sc = SimpleNamespace(release_lease=Mock())
        self.manager._pool_lock = threading.Lock()
        self.manager._pool_leases = {'pool'}
        self.manager.release_pool = Mock()
        self.manager.unload = Mock()
        self.manager.models = {}
        for name in ('_cooldown_task', '_health_task', '_renewal_task',
                     '_pool_renewal_task', '_stats_task', '_offering_task'):
            setattr(self.manager, name, Mock(done=Mock(return_value=False)))

    def assert_untouched(self):
        self.manager.release_pool.assert_not_called()
        self.manager.sc.release_lease.assert_not_called()
        self.manager.unload.assert_not_called()
        self.manager._renewal_task.cancel.assert_not_called()

    def test_any_active_claim_refuses_before_partial_releases_or_renewal_stop(self):
        # Put an idle model first: shutdown must preflight the WHOLE set.
        for managed in (True, False):
            with self.subTest(managed=managed):
                self.manager.models = {
                    'idle': SimpleNamespace(in_flight=0, managed=True, lease_id='idle'),
                    'busy': SimpleNamespace(in_flight=1, managed=managed, lease_id='busy')}
                with self.assertRaisesRegex(RuntimeError, 'Owned work remains'):
                    self.manager.shutdown()
                self.assert_untouched()

    def test_idle_adopted_residency_cannot_release_bytes_and_keep_provider_loaded(self):
        self.manager.models = {'adopted':SimpleNamespace(
            in_flight=0, managed=False, lease_id='resident')}
        with self.assertRaisesRegex(RuntimeError, 'confirmed unload required'):
            self.manager.shutdown()
        self.assert_untouched()

    def test_legacy_shutdown_still_releases_adopted_lease_without_stopping_provider(self):
        self.manager.durable_inference = False
        self.manager.models = {'adopted':SimpleNamespace(
            in_flight=1, managed=False, lease_id='resident')}
        self.manager.shutdown()
        self.manager.sc.release_lease.assert_called_once_with('resident')
        self.manager.unload.assert_not_called()

    def test_explicitly_cleared_residency_allows_shutdown(self):
        self.manager.shutdown()
        self.manager.release_pool.assert_called_once_with('pool')
        self.manager._renewal_task.cancel.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
