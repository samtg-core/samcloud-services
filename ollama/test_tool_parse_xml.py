"""Ollama 0.33.0 XML tool-call parse errors (qwen3-coder / qwen3.5 renderers): terminal, content-free.

LIVE is the exact in-stream error Obelisk's backend sent for execution db354f7c on
11 Nov 2026 (ollama.log 10:45:26 AEDT: "qwen3.5 tool call parsing failed"). The
others are Go encoding/xml's own message forms for xml.Unmarshal into a struct:
SyntaxError ("XML syntax error on line %d: %s") and UnmarshalError. A reset, an
abrupt EOF or a read timeout carries no such frame and must stay uncertain.
"""
import asyncio
import json
from unittest.mock import patch
import unittest

import httpx

from .ollama_client import _backend_error
from .test_gateway_queue import GatewayLifetime

LIVE = 'XML syntax error on line 8: element <parameter> closed by </function>'
PRIVATE = 'PRIVATE_' + 'x' * 8
FIXTURES = {
    'mismatched_close': LIVE,
    'eof': 'XML syntax error on line 3: unexpected EOF',
    'unexpected_end': 'XML syntax error on line 2: unexpected end element </parameter>',
    'bad_entity': 'XML syntax error on line 5: invalid character entity &nbsp;',
    'syntax_other': 'XML syntax error on line 1: expected attribute name in element',
    'element_type': 'expected element type <function> but have <parameter>',
}


class XmlToolParseDiagnostics(unittest.TestCase):
    def test_xml_errors_are_completed_generation_with_a_content_free_shape(self):
        for kind, detail in FIXTURES.items():
            with self.subTest(kind=kind):
                error = _backend_error(200, 'in_stream', 18, detail, parsed=True)
                self.assertEqual(error.subtype, 'tool_parse_xml')
                self.assertTrue(error.completed_generation)
                self.assertEqual(error.diagnostics()['parse_shape'], {'parser': 'xml', 'xml_error': kind})

    def test_element_names_never_reach_diagnostics(self):
        detail = 'XML syntax error on line 4: element <' + PRIVATE + '> closed by </function>'
        error = _backend_error(200, 'in_stream', 2, detail, parsed=True)
        text = json.dumps(error.diagnostics()) + str(error)
        self.assertNotIn(PRIVATE, text)
        self.assertEqual(error.diagnostics()['parse_shape']['xml_error'], 'mismatched_close')

    def test_non_stream_status_body_is_classified_the_same(self):
        error = _backend_error(500, 'status', 0, json.dumps({'error': LIVE}).encode())
        self.assertEqual((error.subtype, error.completed_generation), ('tool_parse_xml', True))

    def test_other_errors_are_not_completed_generation(self):
        for detail in ('EOF', 'unexpected EOF', 'context canceled',
                       'model runner has unexpectedly stopped, this may be due to resource limitations',
                       # Prefix must be at the start; model text is never a tag.
                       'note: XML syntax error on line 1: unexpected EOF',
                       'see expected element type <function> but have <x>'):
            with self.subTest(detail=detail[:24]):
                error = _backend_error(200, 'in_stream', 1, detail, parsed=True)
                self.assertFalse(error.completed_generation)
                self.assertNotIn('parse_shape', error.diagnostics())

    def test_truncated_detail_is_not_classified(self):
        error = _backend_error(200, 'in_stream', 1, LIVE, parsed=True, truncated=True)
        self.assertFalse(error.completed_generation)


class XmlToolParseQueue(GatewayLifetime):
    async def _provider(self, end, *, before=2):
        async def provider(model, messages, **kwargs):
            self.backend.enter(messages)
            for i in range(before):
                yield {'message': {'content': '', 'thinking': 'step %d' % i}, 'done': False}
            raise end
        return provider

    async def _run(self, end, then):
        # The stream is read inside the patch: the provider runs when the body is iterated.
        with patch.object(self.backend, 'chat_stream', await self._provider(end)):
            response = await self.call(stream=True)
            return await then(response, response.headers['X-Inference-Execution'])

    async def test_in_stream_xml_parse_fails_and_frees_the_gpu_for_the_next_call(self):
        async def read(response, execution):
            async with asyncio.timeout(5):
                return execution, ''.join([f async for f in response.body_iterator])
        execution, frames = await self._run(_backend_error(200, 'in_stream', 2, LIVE, parsed=True), read)
        row = self.queue.supervisor.journal.execution(execution, 'creative')
        self.assertEqual(row['state'], 'failed')
        result = json.loads(row['result']) if isinstance(row['result'], str) else row['result']
        self.assertEqual(result['code'], 'backend_tool_parse')
        self.assertEqual(result['diagnostics']['parse_shape'], {'parser': 'xml', 'xml_error': 'mismatched_close'})
        self.assertIn('inference_failed', frames)
        self.assertEqual(self.model.in_flight, 0)
        self.assertEqual(self.queue.held_claims, {})
        # The next caller is dispatched on the freed resource and completes.
        self.backend.gate.set()
        nxt = await self.call('next', stream=True)
        async with asyncio.timeout(5):
            ''.join([f async for f in nxt.body_iterator])
        row = self.queue.supervisor.journal.execution(nxt.headers['X-Inference-Execution'], 'next')
        self.assertEqual(row['state'], 'complete')

    async def _stays_uncertain(self, end):
        async def settle(response, execution):
            await self.wait_for(lambda: self.queue.supervisor.journal.execution(
                execution, 'creative')['state'] == 'uncertain')
            await response.body_iterator.aclose()
        await self._run(end, settle)
        self.assertGreater(self.model.in_flight, 0)
        self.assertIsNone(self.queue.supervisor.pump('gpu'))

    async def test_connection_reset_stays_uncertain(self):
        await self._stays_uncertain(httpx.ReadError('[Errno 54] Connection reset by peer'))

    async def test_abrupt_eof_stays_uncertain(self):
        await self._stays_uncertain(httpx.RemoteProtocolError(
            'peer closed connection without sending complete message body'))

    async def test_read_timeout_stays_uncertain(self):
        await self._stays_uncertain(httpx.ReadTimeout('timed out'))

    async def test_clean_completion_stays_complete(self):
        self.backend.gate.set()
        response = await self.call(stream=True)
        async with asyncio.timeout(5):
            ''.join([f async for f in response.body_iterator])
        row = self.queue.supervisor.journal.execution(response.headers['X-Inference-Execution'], 'creative')
        self.assertEqual(row['state'], 'complete')


del GatewayLifetime  # collected via XmlToolParseQueue: its inherited tests re-run here

if __name__ == '__main__':
    unittest.main()
