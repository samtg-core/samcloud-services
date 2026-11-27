"""Card sampling survives both queue intakes; local Ollama stays unchanged."""
import json
import unittest

from fastapi import HTTPException
from starlette.requests import Request

from . import server
from . import test_exo_group_install as fixtures
from .inference_arbiter import encoded


class NativeSamplingControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = fixtures.InstalledControls()
        await self.context.asyncSetUp()
        self.addAsyncCleanup(self.context.asyncTearDown)
        self.queue = self.context.queue
        self.model = self.context.parts.plan.model_id
        self.body = {'model': self.model, 'messages': [{'role': 'user', 'content': 'hello'}],
                     'temperature': 0.7, 'max_tokens': 16384}
        self.sampling = {'top_p': 0.8, 'top_k': 20, 'min_p': 0}

    def request(self):
        return Request({'type': 'http', 'headers': [(b'idempotency-key', b'card-sampling')],
                        'state': {'caller': {'username': 'caller'}}})

    async def test_durable_intake_preserves_card_options_in_persisted_request(self):
        call = self.queue.validate_call(self.model, {**self.body, **self.sampling})
        journal = self.queue.supervisor.journal
        grant = journal.submit('caller', 'card', 'slice/gpu', self.model)
        execution = journal.enqueue(grant['id'], 'caller', 'call', call)
        persisted = json.loads(journal.execution(execution['id'], 'caller')['request'])
        self.assertEqual({k: persisted['body'][k] for k in self.sampling}, self.sampling)
        self.assertEqual(persisted['body']['temperature'], 0.7)
        self.assertEqual(persisted['body']['max_tokens'], 16384)
        self.assertEqual(self.context.parts.posts, 0)

    async def test_compatibility_intake_preserves_the_same_options(self):
        raw = {**self.body, **self.sampling}
        execution, owner = self.queue.enqueue_compat('chat', server.ChatRequest(**raw), self.request())
        persisted = json.loads(self.queue.supervisor.journal.execution(execution['id'], owner)['request'])
        self.assertEqual(persisted['body'], self.queue.validate_call(self.model, raw)['body'])
        self.assertEqual({k: persisted['body'][k] for k in self.sampling}, self.sampling)
        self.assertEqual(self.context.parts.posts, 0)

    async def test_omitted_options_preserve_the_existing_request_digest(self):
        self.assertEqual(encoded(self.queue.validate_call(self.model, self.body)['body']),
                         encoded(server.ChatRequest(**self.body).model_dump()))
        self.assertFalse(set(self.sampling) & set(server.ChatRequest(**self.body).model_dump()))

    async def test_explicit_null_is_preserved_without_new_defaults(self):
        raw = {**self.body, 'top_p': None}
        body = self.queue.validate_call(self.model, raw)['body']
        self.assertIsNone(body['top_p'])
        self.assertNotIn('top_k', body)
        self.assertNotIn('min_p', body)

    async def test_invalid_native_options_refuse_before_any_grant_is_created(self):
        journal = self.queue.supervisor.journal
        before = journal.db.execute('SELECT COUNT(*) FROM grants').fetchone()[0]
        invalid = [('top_p', True), ('top_p', '0.8'), ('top_p', float('nan')),
                   ('top_p', float('inf')), ('top_p', 1.1), ('top_p', -0.1),
                   ('top_p', 10**1000), ('min_p', False), ('min_p', -1),
                   ('top_k', True), ('top_k', 20.0), ('top_k', '20'), ('top_k', -1)]
        for key, value in invalid:
            raw = {**self.body, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.queue.validate_call(self.model, raw)
            with self.subTest(compatibility=key, value=value), self.assertRaises(HTTPException) as caught:
                self.queue.enqueue_compat('chat', server.ChatRequest(**raw), self.request())
            self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(journal.db.execute('SELECT COUNT(*) FROM grants').fetchone()[0], before)

    async def test_local_ollama_body_is_unchanged_even_with_native_options(self):
        raw = {**self.body, 'model': 'local', 'top_p': 'not-a-native-number'}
        self.assertEqual(self.queue.validate_call('local', raw)['body'],
                         server.ChatRequest(**raw).model_dump())
        self.assertNotIn('top_p', self.queue.validate_call('local', raw)['body'])

    async def test_endpoint_values_remain_explicit(self):
        options = {'top_p': 1, 'min_p': 0.0, 'top_k': 0}
        body = self.queue.validate_call(self.model, {**self.body, **options})['body']
        self.assertEqual({k: body[k] for k in options}, options)


if __name__ == '__main__':
    unittest.main()
