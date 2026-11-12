"""Real loopback HTTP shutdown, fake compute, private temporary journal."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest

from fastapi import FastAPI
import httpx
import uvicorn

from .inference_bootstrap import CapacityEvents, QueueRuntime
from .inference_http_server import QueueServer, observer_closed_response
from .inference_supervisor import ObserverClosed, Supervisor


class HTTPShutdown(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.gate, self.entered, self.observing = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.calls = 0
        self.runtime = QueueRuntime()
        self.runtime.capacity = CapacityEvents({'gpu'})
        self.runtime.capacity_source = None

        async def execute(grant, request):
            self.calls += 1
            self.entered.set()
            await self.gate.wait()
            return {'answer': 'saved'}

        self.s = Supervisor(Path(self.temp.name)/'queue.sqlite', execute,
                            self.runtime.capacity.permitted)
        self.runtime.queue = SimpleNamespace(supervisor=self.s, held_claims={})
        self.grant = self.s.journal.submit('owner', 'turn', 'gpu', 'model')
        self.call = self.s.journal.enqueue(self.grant['id'], 'owner', 'exact-call', {'prompt':'private'})

    async def asyncTearDown(self):
        self.gate.set()
        await asyncio.gather(*list(self.s.workers.values()))
        self.s.close()
        self.temp.cleanup()

    async def test_shutdown_closes_queued_http_observer_and_preserves_exact_request(self):
        before = dict(self.s.journal.execution(self.call['id'], 'owner'))
        lifespan_done = asyncio.Event()

        @asynccontextmanager
        async def lifespan(app):
            yield
            await asyncio.gather(*list(self.s.workers.values()))
            lifespan_done.set()

        app = FastAPI(lifespan=lifespan)
        app.add_exception_handler(ObserverClosed, observer_closed_response)

        @app.get('/wait')
        async def wait():
            self.observing.set()
            return await self.s.observe(self.call['id'], 'owner')

        config = uvicorn.Config(app, host='127.0.0.1', port=0, log_config=None,
                                log_level='critical', timeout_graceful_shutdown=None)
        baseline = os.environ.get('BASELINE_HTTP_SERVER') == '1'
        server = uvicorn.Server(config) if baseline else QueueServer(config, get_runtime=lambda:self.runtime)
        sock = socket.socket()
        sock.bind(('127.0.0.1',0))
        sock.setblocking(False)
        port = sock.getsockname()[1]
        task = asyncio.create_task(server.serve(sockets=[sock]))
        request = None
        try:
            async with asyncio.timeout(2):  # test harness only; no production deadline
                while not server.started:
                    await asyncio.sleep(.001)
            async with httpx.AsyncClient(trust_env=False) as client:
                request = asyncio.create_task(client.get(f'http://127.0.0.1:{port}/wait'))
                await asyncio.wait_for(self.observing.wait(),2)
                server.should_exit = True  # no OS signal or process stop
                await asyncio.wait_for(asyncio.shield(task),2)
                response = await request
            self.assertEqual(response.status_code,503)
            self.assertEqual(response.headers['X-Inference-Execution'],self.call['id'])
            self.assertEqual(response.json()['detail']['code'],'service_restarting')
            self.assertTrue(lifespan_done.is_set())
            self.assertEqual(dict(self.s.journal.execution(self.call['id'],'owner')),before)
            self.assertEqual(self.calls,0)
            same = self.s.journal.enqueue(self.grant['id'],'owner','exact-call',{'prompt':'private'})
            self.assertEqual(same['id'],self.call['id'])
            # Reopen the saved queue as a new scheduler, without resubmitting.
            execute, path = self.s.execute, Path(self.temp.name)/'queue.sqlite'
            self.s.close()
            self.s = Supervisor(path,execute,lambda _:False)
            self.runtime.queue.supervisor = self.s
            self.assertEqual(dict(self.s.journal.execution(self.call['id'],'owner')),before)
        finally:
            server.force_exit = True
            for waiter in list(server.server_state.tasks):
                waiter.cancel()  # test HTTP cleanup only, never a supervisor worker
            if request is not None:
                request.cancel()
                await asyncio.gather(request,return_exceptions=True)
            if not task.done():
                task.cancel()
            await asyncio.gather(task,return_exceptions=True)
            if not lifespan_done.is_set() and hasattr(server,'lifespan'):
                await server.lifespan.shutdown()
            sock.close()

    async def test_detaching_running_observer_does_not_cancel_or_finish_compute(self):
        self.runtime.capacity.observe('gpu',1,'available')
        self.s.pump('gpu')
        await asyncio.wait_for(self.entered.wait(),2)
        before = dict(self.s.journal.execution(self.call['id'],'owner'))
        observer = asyncio.create_task(self.s.observe(self.call['id'],'owner'))
        self.runtime.begin_http_shutdown()
        with self.assertRaises(ObserverClosed) as caught:
            await observer
        self.assertEqual(caught.exception.execution_id,self.call['id'])
        self.assertEqual(dict(self.s.journal.execution(self.call['id'],'owner')),before)
        self.assertTrue(any(not t.done() for t in self.s.workers.values()))
        self.assertEqual(self.calls,1)
        self.gate.set()
        await asyncio.gather(*list(self.s.workers.values()))
        # An already saved terminal result wins over observation shutdown.
        saved = await self.s.observe(self.call['id'],'owner')
        self.assertEqual(saved['state'],'complete')
        self.assertEqual(json.loads(saved['result']),{'answer':'saved'})
        self.assertEqual(self.calls,1)

    async def test_shutdown_fences_dispatch_before_detaching_observers(self):
        self.runtime.capacity.observe('gpu',1,'available')
        self.runtime.begin_http_shutdown()
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertIsNone(self.s.pump('gpu'))
        self.assertEqual(self.s.journal.execution(self.call['id'],'owner')['state'],'queued')
        self.assertEqual(self.calls,0)

    async def test_physical_handoff_keeps_http_observers_open(self):
        self.runtime.manager = SimpleNamespace(models={})
        self.assertTrue((await self.runtime.quiesce(_resource_handoff=True))['drained'])
        self.assertFalse(self.s.observers_closed)
        observer = asyncio.create_task(self.s.observe(self.call['id'],'owner',interval=.001))
        try:
            await asyncio.sleep(.01)
            self.assertFalse(observer.done())
            self.assertEqual(self.s.journal.execution(self.call['id'],'owner')['state'],'queued')
        finally:
            observer.cancel()
            await asyncio.gather(observer,return_exceptions=True)
