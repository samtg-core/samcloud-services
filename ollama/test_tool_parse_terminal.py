"""Ollama 0.33.0 tool-call parse errors: terminal, content-free; no live backend.

FIXTURES are the exact strings v0.33.0's own harmony handler returned for
synthetic completions (harmony.NewHarmonyMessageHandler().Add(..., done=true)).
"""
import asyncio
import json
from unittest.mock import patch
import unittest

from fastapi import HTTPException
import httpx

from .ollama_client import OllamaBackendError, OllamaClient, _backend_error, ERROR_BODY_LIMIT
from .test_gateway_queue import GatewayLifetime

FIXTURES = {
 "array": "error parsing tool call: raw='[\"ls\"]', err=Value looks like object, but can't find closing '}' symbol",
 "bad_escape": "error parsing tool call: raw='{\"path\":\"C:\\q\"}', err=invalid character 'q' in string escape code",
 "bad_sep": "error parsing tool call: raw='{\"command\" \"ls\"}', err=invalid character '\"' after object key",
 "code": "error parsing tool call: raw='import csv\nprint(1)', err=invalid character 'i' looking for beginning of value",
 "empty": "error parsing tool call: raw='', err=unexpected end of JSON input",
 "eof": "error parsing tool call: raw='{\"command\":\"ls\"', err=unexpected end of JSON input",
 "newline_in_str": "error parsing tool call: raw='{\"content\":\"a\nb\"}', err=invalid character '\\n' in string literal",
 "single_quote": "error parsing tool call: raw='{'command':'ls'}', err=invalid character '\\'' looking for beginning of object key string",
 "string": "error parsing tool call: raw='\"ls\"', err=Value looks like object, but can't find closing '}' symbol",
 "trailing": "error parsing tool call: raw='{\"command\":\"ls\"} extra', err=invalid character 'e' after top-level value"
}

EXPECTED = {'array': ('not_object', 'none', 'bracket', 'bracket'), 'string': ('not_object', 'none', 'quote', 'quote'), 'bad_escape': ('bad_escape', 'letter', 'brace', 'brace'), 'bad_sep': ('bad_separator', 'quote', 'brace', 'brace'), 'code': ('not_json', 'letter', 'letter', 'punct'), 'empty': ('eof', 'none', 'none', 'none'), 'eof': ('eof', 'none', 'brace', 'quote'), 'newline_in_str': ('control_in_string', 'newline_or_tab', 'brace', 'brace'), 'single_quote': ('bad_key', 'apostrophe', 'brace', 'brace'), 'trailing': ('trailing_data', 'letter', 'brace', 'letter')}

PRIVATE = 'PRIVATE_' + 'x' * 8


class ToolParseDiagnostics(unittest.TestCase):
    def test_real_backend_strings_classify_without_content(self):
        self.assertEqual(set(FIXTURES), set(EXPECTED))
        for case, detail in FIXTURES.items():
            with self.subTest(case=case):
                error = _backend_error(200, 'in_stream', 3, detail, parsed=True)
                kind, char, first, last = EXPECTED[case]
                shape = error.diagnostics()['parse_shape']
                self.assertEqual(error.subtype, 'tool_parse')
                self.assertTrue(error.completed_generation)
                self.assertEqual((shape['json_error'], shape['char'], shape['raw_first'], shape['raw_last']),
                                 (kind, char, first, last))
                raw = detail[len("error parsing tool call: raw='"):detail.rindex("', err=")]
                self.assertEqual(shape['raw_len'], len(raw))

    def test_private_arguments_never_reach_diagnostics(self):
        detail = "error parsing tool call: raw='{\"path\":\"" + PRIVATE + "\" junk', err=invalid character 'j' after object key:value pair"
        error = _backend_error(200, 'in_stream', 0, detail, parsed=True)
        text = json.dumps(error.diagnostics()) + str(error)
        self.assertNotIn(PRIVATE, text)
        self.assertNotIn('junk', text)
        self.assertEqual(error.diagnostics()['parse_shape']['json_error'], 'bad_separator')

    def test_oversized_arguments_keep_the_tag_but_not_a_shape(self):
        detail = "error parsing tool call: raw='{\"content\":\"" + 'a' * ERROR_BODY_LIMIT + "\n\"}', err=invalid character '\\n' in string literal"
        error = _backend_error(200, 'in_stream', 9, detail, parsed=True)
        self.assertTrue(error.detail_truncated)
        self.assertEqual(error.subtype, 'tool_parse')
        self.assertTrue(error.completed_generation)
        self.assertEqual(error.diagnostics()['parse_shape']['json_error'], 'unknown')

    def test_status_bodies_complete_and_truncated(self):
        body = json.dumps({'error': FIXTURES['code']}).encode()
        error = _backend_error(500, 'status', 0, body)
        self.assertEqual((error.subtype, error.diagnostics()['parse_shape']['json_error']), ('tool_parse', 'not_json'))
        # gin's c.JSON spelling: no space after the colon.
        cut = json.dumps({'error': "error parsing tool call: raw='" + 'b' * ERROR_BODY_LIMIT},
                         separators=(',', ':')).encode()[:ERROR_BODY_LIMIT]
        error = _backend_error(500, 'status', 0, cut, truncated=True)
        self.assertEqual(error.subtype, 'tool_parse')
        self.assertEqual(error.diagnostics()['parse_shape']['json_error'], 'unknown')

    def test_other_errors_are_not_completed_generation(self):
        for detail in ('model runner has unexpectedly stopped, this may be due to resource limitations',
                       'an error was encountered while running the model: GGML_ASSERT',
                       'context canceled', 'PRIVATE something else',
                       # Prefix must be at the start; model text is never a tag.
                       'unexpected: error parsing tool call'):
            with self.subTest(detail=detail[:20]):
                error = _backend_error(200, 'in_stream', 1, detail, parsed=True)
                self.assertFalse(error.completed_generation)
                self.assertNotIn('parse_shape', error.diagnostics())
        error = _backend_error(500, 'status', 0, b'{"detail":"error parsing tool call"}')
        self.assertFalse(error.completed_generation)

    def test_sync_status_500_from_the_client(self):
        client = OllamaClient(base_url='http://fake')
        client._http.close()
        client._http = httpx.Client(base_url='http://fake', transport=httpx.MockTransport(
            lambda req: httpx.Response(500, json={'error': FIXTURES['trailing']})))
        try:
            with self.assertRaises(OllamaBackendError) as caught:
                list(client.chat('gpt-oss', []))
        finally:
            client._http.close()
        self.assertTrue(caught.exception.completed_generation)


class ToolParseQueue(GatewayLifetime):
    async def _failing(self, detail, *, before=1):
        async def provider(model, messages, **kwargs):
            self.backend.enter(messages)
            for i in range(before):
                yield {'message': {'content': '', 'thinking': 'step %d' % i}, 'done': False}
            raise _backend_error(200, 'in_stream', before, detail, parsed=True)
        return provider

    async def test_tool_parse_is_a_terminal_failure_that_frees_the_resource(self):
        detail = FIXTURES['newline_in_str'].replace('a\nb', PRIVATE + '\nb')
        self.assertIn(PRIVATE, detail)
        with patch.object(self.backend, 'chat_stream', await self._failing(detail)):
            response = await self.call(stream=True)
            execution = response.headers['X-Inference-Execution']
            async with asyncio.timeout(5):
                frames = ''.join([f async for f in response.body_iterator])
        row = self.queue.supervisor.journal.execution(execution, 'creative')
        page = self.queue.supervisor.journal.events(execution, 'creative')
        self.assertEqual(row['state'], 'failed')
        result = json.loads(row['result']) if isinstance(row['result'], str) else row['result']
        self.assertEqual(result['code'], 'backend_tool_parse')
        self.assertEqual(result['diagnostics']['parse_shape']['json_error'], 'control_in_string')
        self.assertIn('inference_failed', frames)
        self.assertNotIn(PRIVATE, frames + json.dumps(page) + json.dumps(row, default=str))
        self.assertEqual(self.model.in_flight, 0)
        self.assertEqual(self.queue.held_claims, {})
        # Same key replays the saved failure; it is not re-executed.
        entered = len(self.backend.entered)
        replay = await self.call(stream=True)
        async with asyncio.timeout(5):
            ''.join([f async for f in replay.body_iterator])
        self.assertEqual(len(self.backend.entered), entered)
        # The resource is free for the next caller.
        self.backend.gate.set()
        await self.call('next')

    async def test_non_stream_status_500_tool_parse_is_terminal(self):
        def provider(model, messages, **kwargs):
            raise _backend_error(500, 'status', 0, json.dumps({'error': FIXTURES['code']}).encode())
            yield
        with patch.object(self.backend, 'chat', provider):
            with self.assertRaises(HTTPException) as caught:
                async with asyncio.timeout(5):
                    await self.call()
        self.assertEqual(caught.exception.detail['state'], 'failed')
        self.assertEqual(self.model.in_flight, 0)
        self.assertEqual(self.queue.held_claims, {})

    async def test_other_backend_errors_stay_uncertain(self):
        with patch.object(self.backend, 'chat_stream', await self._failing(
                'model runner has unexpectedly stopped, this may be due to resource limitations')):
            response = await self.call(stream=True)
            execution = response.headers['X-Inference-Execution']
            await self.wait_for(lambda: self.queue.supervisor.journal.execution(
                execution, 'creative')['state'] == 'uncertain')
            await response.body_iterator.aclose()
        self.assertGreater(self.model.in_flight, 0)
        self.assertIsNone(self.queue.supervisor.pump('gpu'))


del GatewayLifetime  # collected via ToolParseQueue: its inherited tests re-run here

if __name__ == '__main__':
    unittest.main()
