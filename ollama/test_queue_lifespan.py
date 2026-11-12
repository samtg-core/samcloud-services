"""Real gateway graceful-shutdown path, synthetic provider and registry only."""
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI

from . import server
from .inference_bootstrap import QueueRuntime, QueueSettings


class Lifespan(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry = SimpleNamespace(verify_token=lambda:{'username':'owner','role':'service'})
        self.manager = SimpleNamespace(sc=self.registry,models={},durable_inference=False,
            residency_renewal=None,discover=Mock(return_value=[]),claim_leases=Mock(return_value=[]),
            start_background_tasks=Mock(),shutdown=Mock(return_value=[]))
        self.app = FastAPI()
        self.patches = [patch.object(server,'app',self.app),
            patch.object(server,'mgr',None),patch.object(server,'inference_dispatcher',None),
            patch.object(server,'inference_runtime',None),
            patch.object(server,'SamcloudClient',return_value=self.registry),
            patch.object(server,'ModelManager',return_value=self.manager),
            patch.object(server.config,'EXO_ENABLED',False)]
        for item in self.patches:
            item.start()
        self.context = server.lifespan(self.app)
        await self.context.__aenter__()
        self.runtime = None
        self.exited = False
        self.release = asyncio.Event()

    async def asyncTearDown(self):
        self.release.set()
        if self.runtime is not None:
            await asyncio.gather(*list(self.runtime.queue.supervisor.workers.values()))
        if not self.exited:
            try:
                await self.context.__aexit__(None,None,None)
            except RuntimeError:
                pass
        if self.runtime is not None:
            # Unknown-work fixture cleanup only. No real reservation exists.
            try:
                self.runtime.queue.supervisor.journal.db.execute('SELECT 1')
            except Exception:
                pass  # normal lifecycle already closed it
            else:
                self.runtime.queue.supervisor.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def install(self):
        self.runtime = QueueRuntime.install(server,QueueSettings(
            Path(self.temp.name)/'queue.sqlite',{'qwen':server.config.SC_RESOURCE_ID},
            reservation_mb={'qwen':100}),registry_owner='owner')
        return self.runtime

    async def start_work(self, *, unknown=False):
        runtime = self.install()
        entered = asyncio.Event()
        async def execute(grant, body):
            entered.set()
            await self.release.wait()
            if unknown:
                raise TimeoutError('synthetic unknown backend outcome')
            return {'answer':'done'}
        runtime.queue.supervisor.execute = execute
        journal = runtime.queue.supervisor.journal
        grant = journal.submit('owner','turn',server.config.SC_RESOURCE_ID,'qwen','execution')
        call = journal.enqueue(grant['id'],'owner','call',{})
        runtime.capacity.observe(server.config.SC_RESOURCE_ID,1,'available')
        await asyncio.wait_for(entered.wait(),2)
        return call

    async def exit(self):
        try:
            await self.context.__aexit__(None,None,None)
        finally:
            self.exited = True

    async def test_real_lifespan_drains_owned_worker_before_manager_releases(self):
        call = await self.start_work()
        self.assertIs(server.inference_runtime,self.runtime)
        task = asyncio.create_task(self.exit())
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.manager.shutdown.assert_not_called()
        self.assertEqual(self.runtime.queue.supervisor.journal.execution(call['id'],'owner')['state'],'running')
        self.release.set()
        await asyncio.wait_for(task,2)
        self.manager.shutdown.assert_called_once_with()

    async def test_real_lifespan_unknown_work_refuses_manager_shutdown_and_keeps_journal(self):
        await self.start_work(unknown=True)
        self.release.set()
        await asyncio.gather(*list(self.runtime.queue.supervisor.workers.values()))
        with self.assertRaisesRegex(RuntimeError,'requires reconciliation'):
            await self.exit()
        self.manager.shutdown.assert_not_called()
        self.assertEqual(self.runtime.queue.supervisor.journal.db.execute(
            "SELECT COUNT(*) FROM executions WHERE state='uncertain'").fetchone()[0],1)

    async def test_mismatched_runtime_cannot_release_different_manager_leases(self):
        runtime = self.install()
        runtime.manager = object()
        with self.assertRaisesRegex(RuntimeError,'ownership differs'):
            await self.exit()
        self.manager.shutdown.assert_not_called()

    async def test_legacy_uninstalled_lifecycle_is_unchanged(self):
        await self.exit()
        self.manager.shutdown.assert_called_once_with()
        self.assertIsNone(server.inference_dispatcher)

    async def test_dispatcher_without_lifecycle_owner_cannot_fall_back_to_legacy_shutdown(self):
        server.inference_dispatcher = object()
        with self.assertRaisesRegex(RuntimeError,'no matching lifecycle owner'):
            await self.exit()
        self.manager.shutdown.assert_not_called()


if __name__ == '__main__':
    unittest.main()
