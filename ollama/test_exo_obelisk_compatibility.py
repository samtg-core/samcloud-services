"""Live Ollama reasoning guards coexist with the native registered EXO route.

The real QueueRuntime, provider binding, arbiter and compatibility enqueue are
used. Physical registry/backend effects stay injected; no model or lease runs.
"""
import unittest

from fastapi import HTTPException
from starlette.requests import Request

from . import server
from . import test_exo_group_install as fixtures


class ProviderReasoningControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = fixtures.InstalledControls()
        await self.context.asyncSetUp()
        self.addAsyncCleanup(self.context.asyncTearDown)
        self.queue = self.context.queue

    def request(self):
        return Request({'type': 'http', 'headers': [(b'idempotency-key', b'compatibility')],
                        'state': {'caller': {'username': 'caller'}}})

    async def test_registered_exo_native_reasoning_without_ollama_budget_stays_queued(self):
        model = self.context.parts.plan.model_id
        body = {'model': model, 'messages': [{'role': 'user', 'content': 'hello'}],
                'reasoning_effort': 'high'}
        self.queue.validate_call(model, body)
        execution, owner = self.queue.enqueue_compat('chat', server.ChatRequest(**body), self.request())
        self.assertEqual(self.queue.supervisor.journal.execution(execution['id'], owner)['state'], 'queued')
        self.assertEqual(self.context.created, 0)
        self.assertEqual(self.context.parts.posts, 0)
        self.context.manager.sc.request_lease.assert_not_called()

    async def test_local_ollama_retains_raw_budget_refusal_before_enqueue(self):
        journal = self.queue.supervisor.journal
        before = journal.db.execute('SELECT COUNT(*) FROM grants').fetchone()[0]
        for budget in (None, '64', 64.0, True):
            body = {'model': 'local', 'messages': [], 'reasoning_effort': 'high'}
            if budget is not None:
                body['max_tokens'] = budget
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.queue.validate_call('local', body)
            with self.subTest(budget=budget), self.assertRaises(HTTPException) as caught:
                self.queue.enqueue_compat('chat', server.ChatRequest(**body), self.request())
            self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(journal.db.execute('SELECT COUNT(*) FROM grants').fetchone()[0], before)
        self.context.manager.sc.request_lease.assert_not_called()


if __name__ == '__main__':
    unittest.main()
