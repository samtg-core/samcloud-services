"""Actual maintenance drain ordering; controlled threads, no providers/network."""
import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from .manager import ModelManager
from .inference_events import source_call


class MaintenanceShutdown(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        m=ModelManager.__new__(ModelManager)
        m.root_binding=None;m.durable_inference=False;m.models={}
        m._pool_lock=threading.Lock();m._pool_leases=set();m.release_pool=Mock()
        for name in ('_cooldown_task','_health_task','_renewal_task',
                     '_pool_renewal_task','_stats_task','_offering_task'):
            setattr(m,name,None)
        return m

    async def held_thread(self,m):
        started=threading.Event();release=threading.Event();finished=threading.Event()
        def work():
            started.set();release.wait();finished.set()
        m._stats_task=asyncio.create_task(source_call(work))
        await asyncio.wait_for(asyncio.to_thread(started.wait),2)
        return release,finished

    async def test_async_stop_does_not_return_while_cancelled_maintenance_thread_owns_work(self):
        m=self.manager();release,finished=await self.held_thread(m)
        stop=asyncio.create_task(m.shutdown_async())
        try:
            await asyncio.sleep(0);await asyncio.sleep(0)
            self.assertFalse(stop.done());self.assertFalse(finished.is_set())
            self.assertFalse(m._stats_task.done())
        finally:release.set()
        self.assertEqual(await asyncio.wait_for(stop,2),[])
        self.assertTrue(finished.is_set());self.assertTrue(m._stats_task.done())

    async def test_observer_cancel_does_not_cancel_drain_and_retry_joins_same_stop(self):
        m=self.manager();release,finished=await self.held_thread(m)
        m.shutdown=Mock(wraps=m.shutdown)
        observer=asyncio.create_task(m.shutdown_async())
        try:
            await asyncio.sleep(0);await asyncio.sleep(0)
            observer.cancel()
            with self.assertRaises(asyncio.CancelledError):await observer
            original=m._shutdown_drain_task
            self.assertFalse(original.done());self.assertFalse(finished.is_set())
            retry=asyncio.create_task(m.shutdown_async());await asyncio.sleep(0)
            self.assertIs(original,m._shutdown_drain_task);m.shutdown.assert_called_once_with()
        finally:release.set()
        self.assertEqual(await asyncio.wait_for(retry,2),[])
        self.assertTrue(finished.is_set());m.shutdown.assert_called_once_with()

    async def test_refused_owned_work_does_not_cancel_maintenance_and_can_retry_after_reconciliation(self):
        m=self.manager();m.durable_inference=True
        m.models={'fixture':SimpleNamespace(in_flight=1,managed=True)}
        gate=asyncio.Event();m._health_task=asyncio.create_task(gate.wait())
        try:
            with self.assertRaisesRegex(RuntimeError,'Owned work remains'):
                await m.shutdown_async()
            self.assertFalse(m._health_task.done());self.assertEqual(m._health_task.cancelling(),0)
            # Fixture supplies explicit known completion; shutdown never does.
            m.models.clear()
            self.assertEqual(await m.shutdown_async(),[])
            self.assertTrue(m._health_task.cancelled())
        finally:
            gate.set();await asyncio.gather(m._health_task,return_exceptions=True)
