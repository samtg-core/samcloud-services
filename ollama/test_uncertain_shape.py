"""The uncertain-shape diagnostic: content-free, and it never changes uncertainty semantics."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import inference_supervisor
from .inference_supervisor import Supervisor, exception_shape

SECRET = 'SYNTHETIC-PROMPT-AND-TOKEN-DO-NOT-LOG'


def inner_parse(body):
    raise ValueError(SECRET + ' in ' + json.dumps(body))


def adapter_step(body):
    try:
        inner_parse(body)
    except ValueError as exc:
        raise RuntimeError(SECRET) from exc


class UncertainShape(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()

        async def execute(grant, body):
            await asyncio.sleep(0)
            adapter_step(body)

        self.s = Supervisor(Path(self.temp.name) / 'jobs.sqlite', execute, lambda _: True)

    async def asyncTearDown(self):
        await asyncio.gather(*list(self.s.workers.values()), return_exceptions=True)
        self.s.close()
        self.temp.cleanup()

    async def run_one(self):
        grant = self.s.journal.submit('owner', 'key', 'slice', 'qwen')
        call = self.s.journal.enqueue(grant['id'], 'owner', 'key', {'prompt': SECRET})
        self.s.pump('slice')
        return await self.s.observe(call['id'], 'owner'), grant

    async def test_shape_names_types_and_frames_without_content(self):
        with self.assertLogs('ollama.inference_supervisor', 'WARNING') as logs:
            row, grant = await self.run_one()
        self.assertEqual(row['state'], 'uncertain')
        self.assertIsNone(row['result'])
        self.assertEqual(self.s.journal.get(grant['id'], 'owner')['state'], 'uncertain')
        text = '\n'.join(logs.output)
        self.assertNotIn('SYNTHETIC', text)
        self.assertIn('error_type=RuntimeError', logs.output[0])  # existing line unchanged
        shape = json.loads(logs.output[1].split('shape=', 1)[1])
        self.assertEqual([x['type'] for x in shape], ['builtins.RuntimeError', 'builtins.ValueError'])
        self.assertEqual(shape[0]['frames'][-1][2], 'adapter_step')
        self.assertEqual(shape[1]['frames'][-1][2], 'inner_parse')
        self.assertTrue(all(f[0].endswith('test_uncertain_shape.py') or '/' in f[0]
                            for x in shape for f in x['frames']))
        self.assertTrue(all(type(f[1]) is int for x in shape for f in x['frames']))

    async def test_failing_diagnostic_cannot_change_uncertain_outcome(self):
        with patch.object(inference_supervisor, 'exception_shape', side_effect=RuntimeError('boom')):
            with self.assertLogs('ollama.inference_supervisor', 'WARNING') as logs:
                row, _ = await self.run_one()
        self.assertEqual(row['state'], 'uncertain')
        self.assertEqual(len(logs.output), 1)
        self.assertIn('error_type=RuntimeError', logs.output[0])

    def test_shape_bounds_and_suppressed_context(self):
        try:
            try:
                raise KeyError(SECRET)
            except KeyError:
                raise RuntimeError(SECRET) from None
        except RuntimeError as exc:
            shape = exception_shape(exc)
        self.assertEqual([x['type'] for x in shape], ['builtins.RuntimeError'])
        self.assertNotIn('SYNTHETIC', json.dumps(shape))
        self.assertEqual(exception_shape(None), [])


if __name__ == '__main__':
    unittest.main()
