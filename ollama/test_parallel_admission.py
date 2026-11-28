"""Two accepted calls still belong to the same physical resource and journal."""
from pathlib import Path
import tempfile
import unittest

from .inference_arbiter import Arbiter, StintPolicy


class ParallelAdmission(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'queue.sqlite'
        self.policy = StintPolicy(parallel_models=('qwen','inkling'))
        self.journal = Arbiter(self.path, stint_policy=self.policy)
        self.addCleanup(lambda: self.journal.close())

    def enqueue(self, model, key=None, operation='chat'):
        grant = self.journal.submit('seat', key or model, 'gpu-0', model)
        execution = self.journal.enqueue(grant['id'], 'seat', 'call', {'operation':operation})
        return grant['id'],execution['id']

    def dispatch(self, permission=True):
        return self.journal.dispatch_next('gpu-0', permitted_model=lambda *_: True,
                                          permitted_pair=lambda *_: permission)

    def test_pair_runs_only_with_confirmed_permission_and_cleans_up_both_orders(self):
        for primary_first in (True,False):
            with self.subTest(primary_first=primary_first):
                a,ax = self.enqueue('qwen', 'q'+str(primary_first))
                b,bx = self.enqueue('inkling','i'+str(primary_first))
                self.assertEqual(ax, self.dispatch()['execution']['id'])
                self.assertIsNone(self.dispatch(False))
                self.assertIsNone(self.dispatch(None))
                self.assertEqual(bx, self.dispatch()['execution']['id'])
                self.assertIsNone(self.dispatch())
                for gid,eid in ([(a,ax),(b,bx)] if primary_first else [(b,bx),(a,ax)]):
                    self.journal.finish(eid,{'ok':True})
                    self.journal.request_release(gid,'seat')
                    self.assertEqual(self.journal.get(gid,'seat')['state'], 'released')

    def test_surviving_owner_can_pair_again_after_primary_releases(self):
        first, x = self.enqueue('qwen')
        survivor, y = self.enqueue('inkling')
        self.dispatch()
        self.dispatch()
        self.journal.request_release(first, 'seat')
        self.journal.finish(x, {'ok': True})
        peer, z = self.enqueue('qwen', 'next-turn')
        self.assertIsNone(self.dispatch(None))
        self.assertEqual(self.journal.get(survivor, 'seat')['parallel_slot'], 1)
        self.assertEqual(self.dispatch()['execution']['id'], z)
        self.assertEqual(self.journal.execution(y, 'seat')['state'], 'running')
        self.assertEqual(self.journal.get(survivor, 'seat')['parallel_slot'], 0)
        self.assertEqual(self.journal.get(peer, 'seat')['parallel_slot'], 1)
        for gid, eid in ((survivor, y), (peer, z)):
            self.journal.request_release(gid, 'seat')
            self.journal.finish(eid, {'ok': True})
            self.assertEqual(self.journal.get(gid, 'seat')['state'], 'released')

    def test_fifo_head_cannot_be_skipped(self):
        self.enqueue('qwen')
        self.enqueue('large')
        self.enqueue('inkling')
        self.dispatch()
        self.assertIsNone(self.dispatch())

    def test_model_lifecycle_operation_cannot_share(self):
        self.enqueue('qwen')
        self.enqueue('inkling',operation='model_unload')
        self.dispatch()
        self.assertIsNone(self.dispatch())

    def test_restart_preserves_both_uncertain_owners(self):
        a,ax = self.enqueue('qwen')
        b,bx = self.enqueue('inkling')
        self.dispatch()
        self.dispatch()
        self.journal.close()
        self.journal = Arbiter(self.path, stint_policy=self.policy)
        self.journal.quarantine_running()
        self.enqueue('inkling','later')
        self.assertIsNone(self.dispatch())
        for gid,eid in ((a,ax),(b,bx)):
            self.assertEqual(self.journal.get(gid,'seat')['state'], 'uncertain')
            self.assertEqual(self.journal.execution(eid,'seat')['state'], 'uncertain')


class ParallelWorkers(unittest.IsolatedAsyncioTestCase):
    async def test_one_wake_starts_both_workers_while_their_grants_remain_owned(self):
        import asyncio
        from .inference_supervisor import Supervisor

        entered = []
        both = asyncio.Event()
        finish = asyncio.Event()
        async def execute(grant, call):
            entered.append(grant['id'])
            if len(entered) == 2:
                both.set()
            await finish.wait()
            return {'ok': True}

        with tempfile.TemporaryDirectory() as tmp:
            s = Supervisor(Path(tmp)/'q', execute, lambda _: True,
                           stint_policy=StintPolicy(parallel_models=('a','b')))
            try:
                s.permitted_pair = lambda *_: True
                for name in ('a','b'):
                    g = s.journal.submit(name, name, 'gpu', name, lifetime='execution')
                    s.journal.enqueue(g['id'], name, 'ask', {'operation':'chat'})
                s.pump('gpu')
                await asyncio.wait_for(both.wait(),2)
                self.assertEqual(len(s.workers),2)
                self.assertEqual(len(set(entered)),2)
            finally:
                finish.set()
                await asyncio.gather(*list(s.workers.values()))
                s.close()


class PairCapacityEvidence(unittest.TestCase):
    def test_requires_two_current_residencies_distinct_leases_and_backends(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from .inference_capacity_source import ManagerCapacitySource

        client = SimpleNamespace(_backend=lambda model: 'http://'+model)
        runtime = SimpleNamespace(manager=SimpleNamespace(ollama=client),capacity=SimpleNamespace())
        source = ManagerCapacitySource(runtime, None)
        source.permitted_model = Mock(return_value=True)
        a = ('qwen', SimpleNamespace(backend='ollama'), 'lease-1')
        b = ('inkling', SimpleNamespace(backend='ollama'), 'lease-2')
        source.observed_residency = {('gpu','qwen'):a, ('gpu','inkling'):b}
        allowed = lambda: source.permitted_pair('gpu','qwen','inkling')
        self.assertTrue(allowed())
        for bad in (('inkling',None,None), ('inkling',b[1],'lease-1'),
                    ('inkling',SimpleNamespace(backend='mlx-vlm'),'lease-2')):
            source.observed_residency['gpu','inkling'] = bad
            self.assertFalse(allowed())
        source.observed_residency['gpu','inkling'] = b
        source.permitted_model.return_value = None
        self.assertFalse(allowed())
        source.permitted_model.return_value = True
        client._backend = lambda _: 'http://same'
        self.assertFalse(allowed())
