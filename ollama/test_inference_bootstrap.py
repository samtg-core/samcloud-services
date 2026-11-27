"""Offline installation, capacity-event ordering and restart admission gates."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel

from .inference_bootstrap import CapacityEvents, QueueRuntime, QueueSettings, cold_capacity
from .inference_arbiter import StintPolicy


class Chat(BaseModel):
    model: str
    messages: list[dict]


def service():
    app = FastAPI()

    @app.middleware('http')
    async def verified_identity(request, call_next):
        # Test-only identity injection. Production keeps its auth middleware.
        if request.headers.get('x-test-owner'):
            request.state.caller = {'username': request.headers['x-test-owner']}
        return await call_next(request)

    host = SimpleNamespace(app=app, config=SimpleNamespace(WHISPER_MAX_UPLOAD_MB=1,
                           SC_SERVICE_ID='service', SC_RESOURCE_ID='gpu'),
                           ChatRequest=Chat, inference_dispatcher=None)

    @app.post('/v1/chat/completions')
    async def compatibility(body: Chat, request: Request):
        if host.inference_dispatcher is None:
            raise AssertionError('test must never reach legacy direct dispatch')
        return await host.inference_dispatcher.compat('chat', body, request)

    return host


class SettingsTests(unittest.TestCase):
    def test_install_refuses_existing_manager_ownership_before_journals_or_routes(self):
        from . import config
        from .manager import ModelManager
        for state in ({'models':{'resident':object()}}, {'_background_started':True},
                      {'_pool_leases':{'lease'}}, {'_unreleased':{'lease':(1,2)}},
                      {'residency_renewal':object()}, {'initial_reservations':object()},
                      {'durable_inference':True}):
            with self.subTest(state=tuple(state)), tempfile.TemporaryDirectory() as folder:
                host = service()
                host.config = config
                host.mgr = ModelManager(sc=Mock())
                for name,value in state.items():
                    setattr(host.mgr,name,value)
                with self.assertRaisesRegex(RuntimeError,'fresh manager'):
                    QueueRuntime.install(host,QueueSettings(Path(folder)/'queue.sqlite',
                        {'qwen':config.SC_RESOURCE_ID},reservation_mb={'qwen':100}),registry_owner='owner')
                self.assertIsNone(host.inference_dispatcher)
                self.assertFalse(list(Path(folder).iterdir()))
                host.mgr.sc.request_lease.assert_not_called()
                self.assertFalse(any(r.path.startswith('/v1/inference') for r in host.app.routes))

    def test_durable_manager_cannot_use_legacy_discovery_or_adoption_allocations(self):
        from .manager import ModelManager
        manager = ModelManager(sc=Mock(),durable_inference=True)
        manager.ollama = Mock()
        manager.llama = Mock()
        for operation in (manager.discover,manager.claim_leases):
            with self.assertRaisesRegex(RuntimeError,'Durable startup'):
                operation()
        self.assertEqual(manager.models,{})
        manager.ollama.list_running.assert_not_called()
        manager.llama.discover_running.assert_not_called()
        manager.sc.request_lease.assert_not_called()

    def test_file_loader_records_real_target_and_exact_loaded_bytes_without_creating_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = {'journal':str(root/'queue.sqlite'), 'models':{'model':'gpu'},
                    'reservation_mb':{'model':100}}
            target = root/'actual.json'
            target.write_text(json.dumps(data))
            alias = root/'profile.json'
            alias.symlink_to(target)
            settings = QueueSettings.from_file(alias)
            self.assertEqual(settings.source_path,target.resolve())
            self.assertEqual(settings.source_sha256,hashlib.sha256(target.read_bytes()).hexdigest())
            self.assertEqual(settings.reservation_mb['model'],100)
            self.assertIsNone(settings.stint_policy.idle_grace_seconds)
            target.write_text('{}')
            self.assertEqual(settings.models,{'model':'gpu'})
            self.assertFalse((root/'queue.sqlite').exists())

    def test_blank_relative_or_missing_configuration_cannot_silently_disable_queue(self):
        for path in ('', 'relative.json', '/nonexistent/samcloud-queue-configuration.json'):
            with self.assertRaises(ValueError):
                QueueSettings.from_file(path)

    def test_file_loader_rejects_duplicate_keys_including_nested_maps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            for raw in ('{"journal":"/q","journal":"/other","models":{},"reservation_mb":{}}',
                        '{"journal":"/q","models":{"m":"g","m":"other"},"reservation_mb":{"m":100}}'):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    QueueSettings.from_file(path)

    def test_file_loader_refuses_extra_fields_partial_budgets_and_invalid_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            base = {'journal':str(Path(folder)/'queue.sqlite'),'models':{'m':'gpu'},'reservation_mb':{'m':100}}
            for change in ({'idle_grace_seconds':300}, {'password':'PRIVATE'},
                           {'reservation_mb':{'m':True}},
                           {'models':[['m','gpu']]}, {'journal':'relative.sqlite'}):
                path.write_text(json.dumps(base|change))
                with self.assertRaises(ValueError) as caught:
                    QueueSettings.from_file(path)
                self.assertNotIn('PRIVATE',str(caught.exception))

    def test_real_manager_binding_requires_explicit_registry_owner_before_routes_or_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            host = service()
            host.mgr = SimpleNamespace(sc=Mock(), durable_inference=False)
            settings = QueueSettings(Path(folder)/'queue.sqlite', {'model':'gpu'},
                                     reservation_mb={'model':100})
            with self.assertRaises(ValueError):
                QueueRuntime.install(host, settings)
            self.assertFalse(host.mgr.durable_inference)
            self.assertIsNone(host.inference_dispatcher)
            self.assertFalse(any(route.path.startswith('/v1/inference') for route in host.app.routes))

    def test_manager_and_initial_journal_bind_as_one_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            host = service()
            host.mgr = SimpleNamespace(sc=Mock(), durable_inference=False)
            runtime = QueueRuntime.install(host,
                QueueSettings(Path(folder)/'queue.sqlite', {'model':'gpu'},
                              reservation_mb={'model':100}), registry_owner='owner')
            try:
                self.assertTrue(host.mgr.durable_inference)
                self.assertEqual(host.mgr.initial_reservations.models, {'model':'gpu'})
                self.assertEqual(host.mgr.initial_reservations.budgets, {'model':100})
                self.assertIs(host.mgr.initial_reservations.client, host.mgr.sc)
                self.assertIsNone(host.mgr.initial_reservations.snapshot('model'))
                host.mgr.sc.request_lease.assert_not_called()
            finally:
                runtime.queue.supervisor.close()

    def test_existing_sizing_installs_without_a_fixed_reservation_or_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            host = service()
            host.mgr = SimpleNamespace(sc=Mock(), durable_inference=False)
            runtime = QueueRuntime.install(host,
                QueueSettings(Path(folder)/'queue.sqlite', {'model':'gpu'}), registry_owner='owner')
            try:
                self.assertIsNotNone(host.inference_dispatcher)
                self.assertEqual(host.mgr.initial_reservations.budgets,{})
                host.mgr.sc.request_lease.assert_not_called()
                self.assertIsNone(host.mgr.initial_reservations.snapshot('model'))
            finally:
                runtime.queue.supervisor.close()

    def test_file_loader_existing_sizing_needs_no_fixed_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            base = {'journal':str(Path(folder)/'queue.sqlite'),'models':{'qwen':'gpu'}}
            for document in (base,base|{'reservation_mb':{}}):
                path.write_text(json.dumps(document))
                self.assertEqual(dict(QueueSettings.from_file(path).reservation_mb),{})
                self.assertFalse((Path(folder)/'queue.sqlite').exists())

    def test_file_loader_pins_between_request_grace_not_a_compute_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'queue.json'
            document={'journal':str(Path(folder)/'queue.sqlite'),'models':{'qwen':'gpu'},
                      'stint_policy':{'idle_grace_seconds':300}}
            path.write_text(json.dumps(document))
            settings=QueueSettings.from_file(path)
            self.assertEqual(settings.stint_policy.idle_grace_seconds,300)
            self.assertIsNone(settings.stint_policy.fairness_seconds)
            self.assertEqual(dict(settings.reservation_mb),{})
            for policy in ({'idle_grace_seconds':0},{'idle_grace_seconds':True},
                           {'compute_timeout_seconds':300},None):
                path.write_text(json.dumps(document|{'stint_policy':policy}))
                with self.assertRaises(ValueError):QueueSettings.from_file(path)

    def test_actual_production_config_names_bind_manager_allocation_identity(self):
        from . import config
        from .manager import ModelManager, SERVICE_ID, RESOURCE_ID
        with tempfile.TemporaryDirectory() as folder:
            host = service()
            host.config = config  # not a fixture alias of a production field
            host.mgr = object.__new__(ModelManager)
            host.mgr.sc = Mock()
            runtime = QueueRuntime.install(host, QueueSettings(Path(folder)/'queue.sqlite',
                {'qwen':RESOURCE_ID}, reservation_mb={'qwen':100}), registry_owner='owner')
            try:
                self.assertEqual(host.mgr.initial_reservations.service, SERVICE_ID)
                self.assertEqual(host.mgr.initial_reservations.models, {'qwen':RESOURCE_ID})
                host.mgr.sc.request_lease.assert_not_called()
            finally:
                runtime.queue.supervisor.close()

    def test_foreign_resource_mapping_refuses_before_installing_or_mutating_manager(self):
        with tempfile.TemporaryDirectory() as folder:
            host = service()
            host.mgr = SimpleNamespace(sc=Mock(), durable_inference=False)
            with self.assertRaisesRegex(ValueError, 'local allocation target'):
                QueueRuntime.install(host, QueueSettings(Path(folder)/'queue.sqlite',
                    {'model':'other-host/gpu'}, reservation_mb={'model':100}), registry_owner='owner')
            self.assertFalse(host.mgr.durable_inference)
            self.assertIsNone(host.inference_dispatcher)

    def test_reservation_budget_configuration_is_exact_positive_and_copied(self):
        budgets = {'model':100}
        settings = QueueSettings(Path('/not-opened/queue.sqlite'), {'model':'gpu'}, reservation_mb=budgets)
        budgets['model'] = 200
        self.assertEqual(settings.reservation_mb['model'],100)
        with self.assertRaises(TypeError):
            settings.reservation_mb['model'] = 200
        for invalid in ({'other':1}, {'model':True}, {'model':0}, {'model':1.5}):
            with self.assertRaises(ValueError):
                QueueSettings(Path('/not-opened/queue.sqlite'), {'model':'gpu'}, reservation_mb=invalid)

    def test_exact_explicit_mapping_and_event_only_default(self):
        mapping = {'model': 'gpu'}
        config = QueueSettings(Path('/not-opened/journal.sqlite'), mapping)
        mapping['model'] = 'different'
        self.assertEqual(config.models, {'model': 'gpu'})
        with self.assertRaises(TypeError):
            config.models['model'] = 'different'
        self.assertIsNone(config.stint_policy.idle_grace_seconds)
        self.assertIsNone(config.stint_policy.fairness_seconds)

    def test_ambiguous_configuration_refuses(self):
        for journal, models, policy in [
                ('relative.sqlite', {'model': 'gpu'}, StintPolicy()),
                ('/not-opened/journal.sqlite', {}, StintPolicy()),
                ('/not-opened/journal.sqlite', {'model': ''}, StintPolicy()),
                ('/not-opened/journal.sqlite', {'model': 'gpu'}, None)]:
            with self.subTest(journal=journal, models=models, policy=policy):
                with self.assertRaises(ValueError):
                    QueueSettings(Path(journal), models, policy)

    def test_permission_starts_unknown_and_stale_events_cannot_restore_it(self):
        gate = CapacityEvents({'gpu'})
        self.assertFalse(gate.permitted('gpu'))
        self.assertFalse(gate.permitted('other'))
        self.assertTrue(gate.observe('gpu', 1, 'available'))
        self.assertTrue(gate.permitted('gpu'))
        self.assertTrue(gate.observe('gpu', 3, 'unknown'))
        self.assertFalse(gate.observe('gpu', 2, 'available'))
        self.assertFalse(gate.observe('gpu', 3, 'available'))
        self.assertFalse(gate.permitted('gpu'))
        self.assertTrue(gate.observe('gpu', 4, 'available'))

    def test_untyped_or_unconfigured_capacity_evidence_refuses(self):
        gate = CapacityEvents({'gpu'})
        for resource, revision, state in [('other', 1, 'available'),
                ('gpu', True, 'available'), ('gpu', 0, 'available'),
                ('gpu', 1, True), ('gpu', 1, 'healthy')]:
            with self.subTest(resource=resource, revision=revision, state=state):
                with self.assertRaises(ValueError):
                    gate.observe(resource, revision, state)
        self.assertFalse(gate.permitted('gpu'))


class ColdCapacityTests(unittest.TestCase):
    def decide(self, *, need=100, work=True, physical=None, registry=None):
        return cold_capacity('gpu', need, work_permitted=work,
            physical={'memory_available_mb':10000} if physical is None else physical,
            registry=({'id':'gpu','allocation_mode':'shared','available_memory_mb':10000,'leases':[]}
                      if registry is None else registry))

    def test_both_independent_capacity_reads_and_permission_required(self):
        self.assertEqual(self.decide()['state'],'available')
        self.assertEqual(self.decide(work=False)['reason'],'work_mode')
        self.assertEqual(self.decide(work=None)['state'],'unknown')
        self.assertEqual(self.decide(physical={})['state'],'unknown')
        self.assertEqual(self.decide(registry={})['state'],'unknown')

    def test_large_physical_machine_does_not_override_registered_resource_cap(self):
        result = self.decide(need=55645, physical={'memory_available_mb':100000},
            registry={'id':'gpu','allocation_mode':'shared','available_memory_mb':49152,'leases':[]})
        self.assertEqual(result['state'],'unavailable')
        self.assertEqual(result['reason'],'registry_capacity')
        self.assertEqual(result['registry_available_mb'],49152)

    def test_registered_capacity_does_not_override_busy_physical_memory(self):
        result = self.decide(need=55645, physical={'memory_available_mb':42324},
            registry={'id':'gpu','allocation_mode':'shared','available_memory_mb':100000,'leases':[]})
        self.assertEqual(result['state'],'unavailable')
        self.assertEqual(result['reason'],'physical_capacity')

    def test_exclusive_owner_and_foreign_or_partial_resource_never_permit(self):
        base = {'id':'gpu','allocation_mode':'shared','available_memory_mb':10000,'leases':[]}
        for changes, reason in (({'id':'other'},'registry_resource_unknown'),
                                ({'leases':None},'registry_leases_unknown'),
                                ({'leases':[{}]},'registry_leases_unknown'),
                                ({'leases':[{'exclusive':1,'status':'active'}]},'exclusive_holder')):
            self.assertEqual(self.decide(registry=base|changes)['reason'],reason)

    def test_missing_boolean_or_invalid_numbers_are_unknown_not_zero_or_permission(self):
        base = {'id':'gpu','allocation_mode':'shared','available_memory_mb':10000,'leases':[]}
        for value in (None, True, -1, '10000'):
            self.assertEqual(self.decide(physical={'memory_available_mb':value})['state'],'unknown')
            self.assertEqual(self.decide(registry=base|{'available_memory_mb':value})['state'],'unknown')


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_background_start_is_sticky_after_its_tasks_are_stopped(self):
        from . import config
        from .manager import ModelManager
        manager = ModelManager(sc=Mock())
        self.assertFalse(manager._background_started)
        with patch.object(config,'OFFERING_ENABLED',False), patch.object(config,'EXO_ENABLED',False):
            manager.start_background_tasks()
        tasks = [value for value in vars(manager).values() if isinstance(value,asyncio.Task)]
        self.assertTrue(tasks)
        for task in tasks:
            task.cancel()  # before yielding: no background backend/registry I/O
        await asyncio.gather(*tasks,return_exceptions=True)
        self.assertTrue(manager._background_started)
        with tempfile.TemporaryDirectory() as folder:
            host = service()
            host.mgr,host.config = manager,config
            with self.assertRaisesRegex(RuntimeError,'fresh manager'):
                QueueRuntime.install(host,QueueSettings(Path(folder)/'queue.sqlite',
                    {'qwen':config.SC_RESOURCE_ID},reservation_mb={'qwen':100}),registry_owner='owner')
            self.assertFalse(list(Path(folder).iterdir()))
        manager.sc.request_lease.assert_not_called()

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = QueueSettings(Path(self.temp.name)/'queue.sqlite', {'model': 'gpu'})
        self.host = service()
        self.runtime = QueueRuntime.install(self.host, self.settings)
        self.queue = self.runtime.queue
        self.entered = []
        self.release = asyncio.Event()

        async def execute(grant, body):
            self.entered.append(grant['owner'])
            await self.release.wait()
            return {'kind': 'json', 'body': {'ok': True}}

        self.queue.supervisor.execute = execute
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.host.app),
                                       base_url='http://test', headers={'x-test-owner': 'owner'})

    async def asyncTearDown(self):
        self.release.set()
        await asyncio.gather(*list(self.queue.supervisor.workers.values()))
        await self.client.aclose()
        self.queue.supervisor.close()
        self.temp.cleanup()

    async def enqueue(self, key='work', owner='owner'):
        headers = {'x-test-owner': owner}
        grant = await self.client.post('/v1/inference/grants', headers=headers,
            json={'key': key, 'model': 'model', 'lifetime': 'execution'})
        self.assertEqual(grant.status_code, 202)
        call = await self.client.post('/v1/inference/grants/'+grant.json()['id']+'/executions',
            headers=headers, json={'key': 'call', 'request': {'model': 'model', 'messages': []}})
        self.assertEqual(call.status_code, 202)
        return call.json()

    async def test_install_mounts_owner_scoped_api_and_dispatcher_together(self):
        self.assertIs(self.host.inference_dispatcher, self.queue)
        self.assertIs(self.runtime.settings, self.settings)
        caps = await self.client.get('/v1/inference/capabilities')
        self.assertTrue(caps.json()['stints']['enabled'])
        self.assertIsNone(caps.json()['stints']['idle_grace_seconds'])
        self.assertEqual((await self.client.get('/v1/inference/capabilities',
            headers={'x-test-owner': ''})).status_code, 401)
        self.assertEqual((await self.client.post('/v1/inference/grants',
            json={'key': 'x', 'model': 'unmapped'})).status_code, 503)
        with self.assertRaisesRegex(RuntimeError, 'already installed'):
            QueueRuntime.install(self.host, self.settings)

    async def test_capacity_event_pumps_saved_work_without_new_request(self):
        job = await self.enqueue()
        await asyncio.sleep(0)
        self.assertEqual(self.entered, [])
        self.assertEqual(job['state'], 'queued')
        self.runtime.capacity.observe('gpu', 1, 'available')
        await asyncio.sleep(0)
        self.assertEqual(self.entered, ['owner'])
        self.assertEqual((await self.client.get('/v1/inference/executions/'+job['id'],
            headers={'x-test-owner': 'other'})).status_code, 404)

    async def test_withdrawal_blocks_next_worker_but_does_not_cancel_active(self):
        await self.enqueue('first')
        self.runtime.capacity.observe('gpu', 1, 'available')
        await asyncio.sleep(0)
        second = await self.enqueue('second', 'peer')
        self.runtime.capacity.observe('gpu', 3, 'unknown')
        self.runtime.capacity.observe('gpu', 2, 'available')
        self.assertEqual(self.entered, ['owner'])
        self.release.set()
        await asyncio.gather(*list(self.queue.supervisor.workers.values()))
        self.assertEqual(self.queue.supervisor.journal.execution(second['id'], 'peer')['state'], 'queued')
        self.runtime.capacity.observe('gpu', 4, 'available')
        await asyncio.sleep(0)
        self.assertEqual(self.entered, ['owner', 'peer'])

    async def test_compatibility_missing_intent_cannot_bypass_queue(self):
        response = await self.client.post('/v1/chat/completions',
            json={'model': 'model', 'messages': []})
        self.assertEqual(response.status_code, 428)
        self.assertEqual(self.entered, [])

    async def test_quiesce_drains_active_work_without_dispatching_waiting_peer(self):
        await self.enqueue('first')
        self.runtime.capacity.observe('gpu', 1, 'available')
        await asyncio.sleep(0)
        await self.enqueue('second', 'peer')
        drain = asyncio.create_task(self.runtime.quiesce())
        await asyncio.sleep(0)
        self.assertFalse(drain.done())
        self.assertEqual(self.entered, ['owner'])
        # New source events cannot undo a deliberate dispatch pause.
        self.runtime.capacity.observe('gpu', 2, 'available')
        self.release.set()
        result = await asyncio.wait_for(drain, 2)
        self.assertEqual(result, {'drained':True, 'running':0, 'uncertain':0, 'queued':1,
                                  'held_claims':0, 'manager_claims':0})
        self.assertEqual(self.entered, ['owner'])
        self.runtime.capacity.resume()
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertFalse(self.runtime.capacity.observe('gpu', 2, 'available'))
        self.runtime.capacity.observe('gpu', 3, 'available')
        await asyncio.sleep(0)
        self.assertEqual(self.entered, ['owner', 'peer'])

    async def test_cancelling_drain_observation_does_not_cancel_worker(self):
        job = await self.enqueue()
        self.runtime.capacity.observe('gpu', 1, 'available')
        await asyncio.sleep(0)
        worker = self.queue.supervisor.workers[job['id']]
        drain = asyncio.create_task(self.runtime.quiesce())
        await asyncio.sleep(0)
        drain.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await drain
        self.assertFalse(worker.done())
        self.assertEqual(self.queue.supervisor.journal.execution(job['id'], 'owner')['state'], 'running')
        self.release.set()
        await worker
        self.assertEqual((await self.runtime.quiesce())['drained'], True)

    async def test_quiesce_does_not_call_uncertain_backend_ownership_drained(self):
        async def lost_contact(grant, body):
            raise RuntimeError('synthetic transport loss, not termination')
        self.queue.supervisor.execute = lost_contact
        await self.enqueue()
        self.runtime.capacity.observe('gpu', 1, 'available')
        await asyncio.sleep(0)
        result = await self.runtime.quiesce()
        self.assertEqual(result, {'drained':False, 'running':0, 'uncertain':1, 'queued':0,
                                  'held_claims':0, 'manager_claims':0})

    async def test_quiesce_requires_resolving_stranded_worker_claims(self):
        self.queue.held_claims['synthetic-prior-worker'] = 'model'
        result = await self.runtime.quiesce()
        self.assertEqual(result, {'drained':False, 'running':0, 'uncertain':0,
                                  'queued':0, 'held_claims':1, 'manager_claims':0})
        # This fixture cleanup is not backend termination evidence or recovery.
        self.queue.held_claims.clear()

    async def test_quiesce_cannot_ignore_claims_outside_queue_bookkeeping(self):
        self.runtime.manager = SimpleNamespace(models={'model':SimpleNamespace(in_flight=2)})
        result = await self.runtime.quiesce()
        self.assertFalse(result['drained'])
        self.assertEqual(result['manager_claims'],2)

    async def test_quiesce_waits_for_owned_input_preparation_too(self):
        preparation_release = asyncio.Event()
        prepared = []
        async def prepare():
            await preparation_release.wait()
            prepared.append(True)
        self.queue.supervisor.prepare(prepare())
        drain = asyncio.create_task(self.runtime.quiesce())
        await asyncio.sleep(0)
        self.assertFalse(drain.done())
        self.assertEqual(prepared, [])
        preparation_release.set()
        self.assertTrue((await asyncio.wait_for(drain, 2))['drained'])
        self.assertEqual(prepared, [True])

    async def test_reopen_starts_unknown_and_quarantines_prior_running_work(self):
        job = await self.enqueue()
        # A crash fixture, not a claim that closing a real worker is safe.
        self.queue.supervisor.journal.dispatch_next('gpu')
        self.queue.supervisor.close()
        self.host = service()
        self.runtime = QueueRuntime.install(self.host, self.settings)
        self.queue = self.runtime.queue
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.queue.supervisor.journal.execution(job['id'], 'owner')['state'], 'uncertain')
        self.runtime.capacity.observe('gpu', 1, 'available')
        self.assertEqual(self.queue.supervisor.workers, {})


if __name__ == '__main__':
    unittest.main()
