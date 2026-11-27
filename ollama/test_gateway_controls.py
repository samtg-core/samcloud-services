"""Explicit load/unload share admission with real inference handlers.

Fake backend/manager operations only; no registry, model load or live service.
"""
import asyncio
import threading
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from . import server
from . import test_gateway_queue as fixture


class GatewayControls(unittest.IsolatedAsyncioTestCase):
    request = fixture.GatewayLifetime.request
    call = fixture.GatewayLifetime.call
    wait_for = fixture.GatewayLifetime.wait_for
    asyncTearDown = fixture.GatewayLifetime.asyncTearDown

    async def asyncSetUp(self):
        await fixture.GatewayLifetime.asyncSetUp(self)
        self.model.port = 0
        self.model.memory_mb = 100
        self.model.lease_id = 'synthetic-lease'

    async def control(self, operation, *, model='qwen', key='control', force=False, ctx=12288):
        request = self.request('operator', key)
        if operation=='load':
            return await server.load_model(server.LoadRequest(model=model, backend='ollama', ctx_size=ctx),
                                           request)
        return await server.unload_model(server.UnloadRequest(model=model, force=force), request)

    async def check_control_waits(self, operation):
        order = []
        enter = self.backend.enter
        def generation(messages):
            order.append('infer:'+messages[-1]['content'])
            enter(messages)
        def mutate(*args, **kwargs):
            self.assertEqual(self.model.in_flight, 0)
            order.append(operation)
            return self.model if operation=='load' else {'status':'unloaded'}
        jobs = []
        method = 'load_ollama_model' if operation=='load' else 'unload'
        with patch.object(self.manager, method, mutate), patch.object(self.backend, 'enter', generation):
            try:
                jobs.append(asyncio.create_task(self.call()))
                await self.wait_for(lambda: self.backend.entered==['creative'])
                jobs.append(asyncio.create_task(self.control(operation, force=True)))
                await self.wait_for(lambda: self.queue.supervisor.journal.db.execute(
                    'SELECT COUNT(*) FROM grants').fetchone()[0]==2)
                jobs.append(asyncio.create_task(self.call('assistant')))
                await self.wait_for(lambda: self.queue.supervisor.journal.db.execute(
                    'SELECT COUNT(*) FROM grants').fetchone()[0]==3)
                self.assertEqual(order, ['infer:creative'])
                self.backend.gate.set()
                results = await asyncio.wait_for(asyncio.gather(*jobs), 2)
                self.assertEqual(order, ['infer:creative', operation, 'infer:assistant'])
                self.assertEqual(results[1]['status'], 'loaded' if operation=='load' else 'unloaded')
                replay = await self.control(operation, force=True)
                self.assertEqual(replay, results[1])
                self.assertEqual(order.count(operation), 1)
            finally:
                self.backend.gate.set()
                for job in jobs:
                    if not job.done():
                        job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)

    async def test_explicit_load_waits_between_inference_and_reopens_saved_result(self):
        await self.check_control_waits('load')

    async def test_force_unload_cannot_jump_inflight_or_queued_inference(self):
        await self.check_control_waits('unload')

    async def test_control_key_cannot_be_reused_for_changed_request(self):
        calls = []
        def load(name):
            calls.append(name)
            return self.model
        with patch.object(self.manager, 'load_ollama_model', load):
            await self.control('load')
            with self.assertRaises(HTTPException) as caught:
                await self.control('load', ctx=9999)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(calls, ['qwen'])

    async def test_unmapped_control_has_no_mutation_or_grant(self):
        with patch.object(self.manager, 'unload') as unload:
            with self.assertRaises(HTTPException) as caught:
                await self.control('unload', model='unconfigured', force=True)
        self.assertEqual(caught.exception.status_code, 503)
        unload.assert_not_called()
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            'SELECT COUNT(*) FROM grants').fetchone()[0], 0)

    async def test_cancelled_control_observer_does_not_cancel_mutation(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def load(name):
            calls.append(name)
            entered.set()
            if not release.wait(3):
                raise AssertionError('control cleanup did not release load')
            return self.model
        jobs = []
        with patch.object(self.manager, 'load_ollama_model', load):
            try:
                jobs.append(asyncio.create_task(self.control('load')))
                await self.wait_for(entered.is_set)
                self.assertEqual((await asyncio.wait_for(server.health(), .5))['status'], 'ok')
                jobs[0].cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await jobs[0]
                jobs.append(asyncio.create_task(self.call('assistant')))
                await asyncio.sleep(.03)
                self.assertEqual(self.backend.entered, [])
                release.set()
                self.backend.gate.set()
                await asyncio.wait_for(jobs[1], 2)
                result = await self.control('load')
                self.assertEqual(result['status'], 'loaded')
                self.assertEqual(calls, ['qwen'])
            finally:
                release.set()
                self.backend.gate.set()
                for job in jobs:
                    if not job.done():
                        job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)

    async def test_control_transport_error_stays_unknown_and_holds_queue(self):
        def lost_load(name):
            raise TimeoutError('PRIVATE backend detail')
        with patch.object(self.manager, 'load_ollama_model', lost_load):
            with self.assertRaises(HTTPException) as caught:
                await self.control('load')
        self.assertEqual(caught.exception.detail['state'], 'uncertain')
        self.assertNotIn('PRIVATE', str(caught.exception.detail))
        self.assertIsNone(self.queue.supervisor.pump('gpu'))

    async def test_unload_lost_reply_keeps_model_and_lease_accounted(self):
        self.model.managed = True
        with patch.object(self.backend, 'unload_model',
                side_effect=TimeoutError('PRIVATE unload detail'), create=True), \
             patch.object(self.manager, '_release_lease_quietly') as release:
            with self.assertRaises(HTTPException) as caught:
                await self.control('unload', force=True)
            self.assertEqual(caught.exception.detail['state'], 'uncertain')
            release.assert_not_called()
        self.assertIs(self.manager.models['qwen'], self.model)
        self.assertEqual(self.model.lease_id, 'synthetic-lease')
        self.assertNotIn('PRIVATE', str(caught.exception.detail))
        self.assertIsNone(self.queue.supervisor.pump('gpu'))

    async def test_unload_positive_response_requires_absent_residency_before_release(self):
        self.model.managed = True
        self.model.loaded_at = 0
        with patch.object(self.backend, 'unload_model', return_value={'done':True}, create=True), \
             patch.object(self.backend, 'list_running', return_value=[], create=True), \
             patch.object(self.manager, '_release_lease_quietly') as release:
            result = await self.control('unload', force=True)
            self.assertEqual(result['status'], 'unloaded')
            release.assert_called_once_with('synthetic-lease')
        self.assertNotIn('qwen', self.manager.models)

    async def test_unknown_unload_postcondition_cannot_free_capacity(self):
        self.model.managed = True
        for response, running in (({}, []), ({'done':True}, [{'name':'qwen'}]),
                                  ({'done':True}, [{}])):
            with self.subTest(response=response, running=running), \
                 patch.object(self.backend, 'unload_model', return_value=response, create=True), \
                 patch.object(self.backend, 'list_running', return_value=running, create=True), \
                 patch.object(self.manager, '_release_lease_quietly') as release:
                with self.assertRaises(RuntimeError):
                    self.manager.unload('qwen',force=True,require_confirmed=True)
                release.assert_not_called()
                self.assertIs(self.manager.models['qwen'], self.model)

    async def test_installed_mode_also_protects_background_unload(self):
        self.model.managed = True
        self.manager.durable_inference = True
        self.model.in_flight = 1
        with patch.object(self.backend, 'unload_model', create=True) as unload:
            with self.assertRaises(RuntimeError):
                self.manager.unload('qwen', force=True)
            unload.assert_not_called()
        self.model.in_flight = 0
        with patch.object(self.backend, 'unload_model', side_effect=TimeoutError(), create=True), \
             patch.object(self.manager, '_release_lease_quietly') as release:
            with self.assertRaises(RuntimeError):
                self.manager.unload('qwen')
            release.assert_not_called()

    async def test_force_unload_cannot_bypass_uncertain_generation(self):
        self.backend.omit_done = True
        self.backend.gate.set()
        with self.assertRaises(HTTPException) as caught:
            await self.call(stream=False)
        self.assertEqual(caught.exception.detail['state'],'uncertain')
        self.assertGreater(self.model.in_flight,0)
        with patch.object(self.manager,'unload') as unload:
            pending = asyncio.create_task(self.control('unload',force=True))
            try:
                await self.wait_for(lambda: self.queue.supervisor.journal.db.execute(
                    'SELECT COUNT(*) FROM grants').fetchone()[0]==2)
                await asyncio.sleep(.03)
                self.assertFalse(pending.done())
                unload.assert_not_called()
                self.assertIsNone(self.queue.supervisor.pump('gpu'))
            finally:
                pending.cancel()
                await asyncio.gather(pending,return_exceptions=True)

    async def test_control_requires_verified_owner_before_admission(self):
        request = Request({'type':'http','headers':[],'state':{}})
        with patch.object(self.manager,'unload') as unload:
            with self.assertRaises(HTTPException) as caught:
                await server.unload_model(server.UnloadRequest(model='qwen',force=True),request)
        self.assertEqual(caught.exception.status_code,401)
        unload.assert_not_called()
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            'SELECT COUNT(*) FROM grants').fetchone()[0],0)


if __name__=='__main__':
    unittest.main()
