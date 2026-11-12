"""Actual gateway startup ordering; synthetic registry/provider, no live fleet."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
import httpx

from . import server
from .inference_bootstrap import QueueSettings, QueueStartup
from .ollama_client import OllamaClient
from .samcloud import SamcloudClient


class Startup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.registry = SimpleNamespace(verify_token=Mock(return_value={
            'valid':True,'owner_id':'user:65','username':'own-service'}),
            api_index=Mock(return_value={'endpoints':{'leases':{'rollover':{
                'method':'POST','path':'/api/v1/leases/{id}/rollover'}}}}))
        self.manager = SimpleNamespace(sc=self.registry,models={},durable_inference=False,
            residency_renewal=None,ollama=SimpleNamespace(list_running=Mock(return_value=[])),
            discover=Mock(),claim_leases=Mock(),start_background_tasks=Mock(),
            shutdown=Mock(return_value=[]))
        self.app = FastAPI()
        for name,value in (('app',self.app),('mgr',None),('inference_dispatcher',None),
                           ('inference_runtime',None),('inference_startup',None),
                           ('SamcloudClient',Mock(return_value=self.registry)),
                           ('ModelManager',Mock(return_value=self.manager))):
            self.stack.enter_context(patch.object(server,name,value))
        self.stack.enter_context(patch.object(server.config,'EXO_ENABLED',False))
        self.stack.enter_context(patch.object(server.config,'SC_TOKEN',''))
        self.settings = QueueSettings(self.root/'queue.sqlite',
            {'qwen':server.config.SC_RESOURCE_ID},reservation_mb={'qwen':100})
        self.context = None

    async def asyncTearDown(self):
        if self.context is not None:
            await self.context.__aexit__(None,None,None)
        self.stack.close()
        self.temp.cleanup()

    async def boot(self):
        context = server.lifespan(self.app)
        await context.__aenter__()
        self.context = context

    async def test_vision_startup_refuses_an_existing_or_unknown_listener(self):
        vision='mlx-community/Qwen2.5-VL-7B-Instruct-4bit'
        settings=QueueSettings(self.root/'vision.sqlite',{'qwen':server.config.SC_RESOURCE_ID,
                                                        vision:server.config.SC_RESOURCE_ID})
        from . import inference_bootstrap
        server.mgr=self.manager
        for verdict in (False,None):
            with self.subTest(verdict=verdict),patch.object(inference_bootstrap,
                    'vision_backend_absent',return_value=verdict),self.assertRaisesRegex(RuntimeError,'vision backend'):
                await QueueStartup(settings).start(server)
        self.assertFalse(settings.journal.exists())

    async def test_real_lifespan_installs_before_maintenance_without_legacy_adoption(self):
        def maintenance():
            self.assertTrue(self.manager.durable_inference)
            self.assertIsNotNone(server.inference_runtime)
        self.manager.start_background_tasks.side_effect = maintenance
        server.inference_startup = QueueStartup(self.settings)
        await self.boot()
        self.manager.discover.assert_not_called()
        self.manager.claim_leases.assert_not_called()
        self.registry.verify_token.assert_called_once_with()
        self.manager.ollama.list_running.assert_called_once_with()
        runtime = server.inference_runtime
        self.assertEqual(runtime.manager.initial_reservations.owner,'user:65')
        self.assertFalse(runtime.capacity.permitted(server.config.SC_RESOURCE_ID))
        grant = runtime.queue.supervisor.journal.submit('worker','turn',
            server.config.SC_RESOURCE_ID,'qwen')
        call = runtime.queue.supervisor.journal.enqueue(grant['id'],'worker','call',{})
        self.assertIsNone(runtime.queue.supervisor.pump(server.config.SC_RESOURCE_ID))
        self.assertEqual(runtime.queue.supervisor.journal.execution(call['id'],'worker')['state'],'queued')

    async def test_shared_queue_installs_only_with_verified_hold_contract_and_takes_no_idle_lease(self):
        resource=server.config.SC_RESOURCE_ID
        self.registry.api_index.return_value['endpoints']['leases']['wait']={
            'method':'GET','path':'/api/v1/leases/{id}/wait?timeout_s=30'}
        self.registry.get_resource=Mock(return_value={'id':resource,'expiry_policy':'hold'})
        self.registry.request_lease=Mock()
        self.manager.status=Mock(return_value={})
        server.inference_startup=QueueStartup(QueueSettings(self.root/'shared.sqlite',{'qwen':resource},
            reservation_mb={'qwen':100},shared_queue_enabled=True),work_permitted=True)
        await self.boot()
        runtime=server.inference_runtime
        self.assertIsNotNone(runtime.shared_resource)
        self.assertIsNone(runtime.manager.initial_reservations.wait_s)
        self.assertTrue(runtime.manager.initial_reservations.wait)
        self.registry.request_lease.assert_not_called()
        self.assertFalse(runtime.capacity.permitted(resource))
        report=await server.status()
        self.assertEqual(report['resource_queue']['handoff'],'idle')

    async def test_shared_queue_refuses_free_policy_or_wrong_resource_before_install(self):
        resource=server.config.SC_RESOURCE_ID
        self.registry.api_index.return_value['endpoints']['leases']['wait']={
            'method':'GET','path':'/api/v1/leases/{id}/wait?timeout_s=30'}
        settings=QueueSettings(self.root/'shared.sqlite',{'qwen':resource},shared_queue_wait_s=3600)
        server.mgr=self.manager
        for view in ({'id':resource,'expiry_policy':'free'}, {'id':'other','expiry_policy':'hold'}):
            self.registry.get_resource=Mock(return_value=view)
            with self.assertRaisesRegex(RuntimeError,'queue/hold'):
                await QueueStartup(settings).start(server)
        self.assertFalse(settings.journal.exists())
        self.assertIsNone(server.inference_runtime)

    async def test_indefinite_shared_queue_refuses_free_policy_or_wrong_resource(self):
        resource=server.config.SC_RESOURCE_ID
        self.registry.api_index.return_value['endpoints']['leases']['wait']={
            'method':'GET','path':'/api/v1/leases/{id}/wait?timeout_s=30'}
        settings=QueueSettings(self.root/'shared.sqlite',{'qwen':resource},shared_queue_enabled=True)
        server.mgr=self.manager
        for view in ({'id':resource,'expiry_policy':'free'}, {'id':'other','expiry_policy':'hold'}):
            self.registry.get_resource=Mock(return_value=view)
            with self.assertRaisesRegex(RuntimeError,'queue/hold'):
                await QueueStartup(settings).start(server)
        self.assertFalse(settings.journal.exists())
        self.assertIsNone(server.inference_runtime)

    async def test_startup_binds_real_capacity_and_residency_lifecycle(self):
        resource = server.config.SC_RESOURCE_ID
        self.registry.get_resource = Mock(return_value={'id':resource,
            'allocation_mode':'shared','available_memory_mb':10000,'leases':[]})
        server.inference_startup = QueueStartup(self.settings,work_permitted=True)
        await self.boot()
        runtime = server.inference_runtime
        source = runtime.capacity_source
        self.assertIs(self.manager.inference_capacity_source,source)
        self.assertIs(self.manager.residency_renewal,source.binding)
        await source.refresh({'memory_available_mb':20000,'memory_device_inuse_mb':0})
        self.assertTrue(runtime.capacity.permitted(resource))
        self.assertTrue((self.root/'residency.sqlite').exists())
        result = await runtime.quiesce()
        self.assertTrue(result['drained'])
        self.assertTrue(source.paused)
        self.assertTrue(source.binding.paused)
        self.assertFalse(runtime.capacity.permitted(resource))
        runtime.resume()
        self.assertFalse(runtime.capacity.permitted(resource))
        await source.refresh({'memory_available_mb':20000,'memory_device_inuse_mb':0})
        self.assertTrue(runtime.capacity.permitted(resource))

    async def test_invalid_permission_refuses_before_mounting_or_creating_journals(self):
        server.inference_startup = QueueStartup(self.settings,work_permitted='yes')
        with self.assertRaisesRegex(ValueError,'explicit work permission'):
            await self.boot()
        self.registry.verify_token.assert_not_called()
        self.assert_refused()

    async def test_failed_post_execution_collection_cannot_leave_old_admission_enabled(self):
        resource = server.config.SC_RESOURCE_ID
        self.registry.get_resource = Mock(return_value={'id':resource,
            'allocation_mode':'shared','available_memory_mb':10000,'leases':[]})
        self.manager._collect_stats = Mock(side_effect=RuntimeError('synthetic collector failure'))
        server.inference_startup = QueueStartup(self.settings,work_permitted=True)
        await self.boot()
        runtime = server.inference_runtime
        await runtime.capacity_source.refresh({'memory_available_mb':20000,'memory_device_inuse_mb':0})
        self.assertTrue(runtime.capacity.permitted(resource))
        journal = runtime.queue.supervisor.journal
        grant = journal.submit('worker','turn',resource,'qwen')
        execution = journal.enqueue(grant['id'],'worker','call',{})
        with self.assertRaisesRegex(RuntimeError,'synthetic collector failure'):
            await runtime.queue.supervisor.on_finished(grant,execution['id'])
        self.assertFalse(runtime.capacity.permitted(resource))

    async def test_unverified_identity_refuses_before_inventory_journals_or_fallback(self):
        self.registry.verify_token.side_effect = RuntimeError('PRIVATE secret detail')
        server.inference_startup = QueueStartup(self.settings)
        with self.assertRaisesRegex(RuntimeError,'identity is unverified') as error:
            await self.boot()
        self.assertNotIn('PRIVATE',str(error.exception))
        self.manager.ollama.list_running.assert_not_called()
        self.assert_refused()

    async def test_missing_identity_is_not_borrowed_from_an_inference_caller(self):
        self.registry.verify_token.return_value = {'role':'service'}
        server.inference_startup = QueueStartup(self.settings)
        with self.assertRaisesRegex(RuntimeError,'identity is unverified'):
            await self.boot()
        self.assert_refused()

    async def test_display_name_is_not_used_as_canonical_registry_lease_owner(self):
        self.registry.verify_token.return_value={'valid':True,'owner_id':'user:65',
                                                'username':'a-different-display-name'}
        server.inference_startup=QueueStartup(self.settings,work_permitted=True)
        await self.boot()
        runtime=server.inference_runtime
        self.assertEqual(runtime.manager.initial_reservations.owner,'user:65')
        self.assertEqual(runtime.residency_actor.store.owner,'user:65')
        resource=server.config.SC_RESOURCE_ID
        receipt={'id':'exact-owned','agent_id':'user:65','service_id':server.config.SC_SERVICE_ID,
            'resource_id':resource,'memory_mb':100,'max_total_s':3600,'exclusive':0,
            'status':'active','expires_at':'2099-01-01T00:00:00+00:00'}
        self.registry.get_resource=Mock(return_value={'id':resource,'allocation_mode':'shared',
            'available_memory_mb':0,'leases':[receipt]})
        self.registry.list_leases=Mock(return_value=[receipt])
        self.manager.models['qwen']=SimpleNamespace(name='qwen',lease_id='exact-owned',
                                                   memory_mb=80,in_flight=0)
        await runtime.capacity_source.refresh({'memory_available_mb':20000,'memory_device_inuse_mb':80})
        self.assertTrue(runtime.capacity.permitted(resource))

    async def test_username_alone_or_invalid_bearer_never_binds_a_lease_owner(self):
        for document in ({'username':'own-service'},
                         {'valid':False,'owner_id':'user:65','username':'own-service'},
                         {'valid':True,'owner_id':None,'username':'own-service'}):
            with self.subTest(document=document):
                self.registry.verify_token.return_value=document
                server.inference_startup=QueueStartup(self.settings)
                with self.assertRaisesRegex(RuntimeError,'identity is unverified'):
                    await self.boot()
                self.assert_refused()

    async def test_existing_or_unknown_inventory_never_gets_adopted_or_reloaded(self):
        server.inference_startup = QueueStartup(self.settings)
        for inventory in ([{'name':'resident'}],None,{},False):
            with self.subTest(inventory=inventory):
                self.manager.ollama.list_running.return_value = inventory
                with self.assertRaisesRegex(RuntimeError,'confirmed empty Ollama inventory'):
                    await self.boot()
                self.assert_refused()

    async def test_provider_observation_failure_is_not_an_empty_inventory(self):
        server.inference_startup = QueueStartup(self.settings)
        self.manager.ollama.list_running.side_effect = TimeoutError('PRIVATE backend detail')
        with self.assertRaisesRegex(RuntimeError,'confirmed empty Ollama inventory') as error:
            await self.boot()
        self.assertNotIn('PRIVATE',str(error.exception))
        self.assert_refused()

    async def test_undeployed_or_unknown_rollover_refuses_before_inventory_or_journals(self):
        for document in ({},{'endpoints':{}},{'endpoints':{'leases':{'rollover':{
                'method':'GET','path':'/api/v1/leases/{id}/rollover'}}}}):
            with self.subTest(document=document):
                self.registry.api_index.return_value=document
                server.inference_startup=QueueStartup(self.settings)
                with self.assertRaisesRegex(RuntimeError,'deployed registry rollover contract'):
                    await self.boot()
                self.manager.ollama.list_running.assert_not_called()
                self.assert_refused()

    async def test_registry_contract_read_failure_does_not_use_legacy_renewal(self):
        self.registry.api_index.side_effect=TimeoutError('PRIVATE registry detail')
        server.inference_startup=QueueStartup(self.settings)
        with self.assertRaisesRegex(RuntimeError,'deployed registry rollover contract') as error:
            await self.boot()
        self.assertNotIn('PRIVATE',str(error.exception))
        self.assert_refused()

    async def test_invalid_startup_binding_never_silently_uses_legacy_mode(self):
        server.inference_startup = object()
        with self.assertRaisesRegex(TypeError,'Trusted QueueStartup'):
            await self.boot()
        self.registry.verify_token.assert_not_called()
        self.assert_refused()

    def assert_refused(self):
        self.manager.discover.assert_not_called()
        self.manager.claim_leases.assert_not_called()
        self.manager.start_background_tasks.assert_not_called()
        self.manager.shutdown.assert_not_called()  # no lease cleanup on failed boot
        self.assertIsNone(server.inference_dispatcher)
        self.assertFalse(list(self.root.iterdir()))


class Inventory(unittest.TestCase):
    def test_registry_index_uses_own_client_and_exact_api_root(self):
        def response(request):
            self.assertEqual(str(request.url),'https://synthetic/api/v1/')
            self.assertEqual(request.headers['Authorization'],'Bearer synthetic')
            return httpx.Response(200,json={'endpoints':{}})
        with httpx.Client(base_url='https://synthetic/api/v1',
                headers={'Authorization':'Bearer synthetic'},transport=httpx.MockTransport(response)) as http:
            client=SamcloudClient(token='synthetic',_http=http)
            client._http.close()  # dataclass __post_init__ replaces the provided client
            client._http=http
            self.assertEqual(client.api_index(),{'endpoints':{}})

    def test_real_client_requires_an_explicit_models_list(self):
        client = OllamaClient()
        client._http.close()
        try:
            for document in ({},[],{'models':None},{'models':{}},{'models':[]}):
                with self.subTest(document=document):
                    with httpx.Client(base_url='http://synthetic',transport=httpx.MockTransport(
                            lambda request:httpx.Response(200,json=document))) as http:
                        client._http = http
                        if document == {'models':[]}:
                            self.assertEqual(client.list_running(),[])
                        else:
                            with self.assertRaisesRegex(ValueError,'inventory is unknown'):
                                client.list_running()
        finally:
            client._http.close()


if __name__ == '__main__':
    unittest.main()
