"""Local source refusals stay fail-closed without a network retry/fallback loop."""
import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
from openai import APIConnectionError
from agent.error_classifier import classify_api_error
import providers
from .hermes_queue_identity import HostBindings
from .hermes_queue_provider import register_profiles
from .inference_identity import LocalSourceRefused
from .inference_sdk import QueuedClient


class LocalSourceClassification(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.environment = patch.dict(os.environ, HERMES_HOME=self.temp.name)
        self.environment.start(); self.addCleanup(self.environment.stop)
        self.registry = patch.object(providers, '_REGISTRY', dict(providers._REGISTRY))
        self.registry.start(); self.addCleanup(self.registry.stop)
        register_profiles()
        self.host = HostBindings(Path(self.temp.name)/'identity.sqlite', 'synthetic-profile')

    async def test_real_unbound_host_and_sdk_fail_locally_on_both_transports(self):
        for sync in (False, True):
            for stream in (False, True):
                calls = []
                def forbidden(request):
                    calls.append(request.url.path)
                    raise AssertionError('Unbound source contacted service')
                client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                    http_factory=lambda:httpx.AsyncClient(base_url='https://queue.invalid',
                        transport=httpx.MockTransport(forbidden)),
                    key_provider=self.host.main_stint_keys, state_frames=True)
                try:
                    body = {'model':'qwen', 'messages':[{'role':'user','content':'fixture'}], 'stream':stream}
                    with self.assertRaises(APIConnectionError) as caught:
                        if sync:
                            await asyncio.to_thread(lambda:client.create(**body))
                        else:
                            await client.create(**body)
                    verdict = classify_api_error(caught.exception, provider='samcloud-queue-stint', model='qwen')
                    self.assertEqual(verdict.reason.value, 'format_error')
                    self.assertFalse(verdict.retryable)
                    self.assertFalse(verdict.should_fallback)
                    self.assertFalse(verdict.should_compress)
                    self.assertEqual(verdict.error_context['origin'], 'local_queue_adapter')
                    self.assertEqual(calls, [])
                finally:
                    await client.aclose()

    async def test_genuine_transport_failure_remains_retryable_timeout(self):
        request = httpx.Request('GET','https://queue.invalid')
        error = APIConnectionError(request=request)
        error.__cause__ = httpx.ConnectError('connection refused', request=request)
        verdict = classify_api_error(error, provider='samcloud-queue-stint', model='qwen')
        self.assertEqual(verdict.reason.value, 'timeout')
        self.assertTrue(verdict.retryable)

    async def test_prose_does_not_authorize_local_override(self):
        profile = providers.get_provider_profile('samcloud-queue-stint')
        error = ValueError('no owning foreground source is bound for a stint')
        self.assertIsNone(profile.classify_api_error(error))
        error.__cause__ = error
        self.assertIsNone(profile.classify_api_error(error))
        other = classify_api_error(LocalSourceRefused('fixture'), provider='unrelated-provider', model='qwen')
        self.assertNotEqual(other.error_context.get('origin'), 'local_queue_adapter')


if __name__ == '__main__':
    unittest.main()
