"""Desired single-slot admission contract; initially RED on f5833a2.

Runs the real chat handler and real ModelManager claim/release methods with a
blocked fake backend. No network, model loading, registry writes or lifespan.
This deliberately does not let the fake backend serialize requests: admission
must happen in model-service, not be delegated to Ollama. It makes no choice
about whether a consumer's lease spans one request or several.

Run: python -m unittest -v ollama.test_service_admission
"""
import asyncio
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from starlette.requests import Request

from . import server
from .manager import Backend, ModelManager
from .inference_gateway import GatewayQueue


class BlockedBackend:
    def __init__(self):
        self.entered = []
        self.gate = threading.Event()
        self.lock = threading.Lock()

    def enter(self, messages):
        with self.lock:
            self.entered.append(messages[-1]["content"])

    @staticmethod
    def answer():
        return {"message": {"content": "done"}, "done": True}

    def chat(self, model, messages, **kwargs):
        self.enter(messages)
        if not self.gate.wait(5):
            raise AssertionError("test cleanup failed to release fake provider")
        yield self.answer()

    async def chat_stream(self, model, messages, **kwargs):
        self.enter(messages)
        while not self.gate.is_set():
            await asyncio.sleep(0.005)
        yield self.answer()


class ServiceAdmission(unittest.IsolatedAsyncioTestCase):
    async def check_exclusive_dispatch(self, streaming, control_gate=False, enabled=True):
        backend = BlockedBackend()
        model = SimpleNamespace(name="test-model", backend=Backend.OLLAMA,
                                in_flight=0, last_used=0)
        # Avoid constructors that create real provider/registry clients; the
        # production claim, release and serving implementations remain intact.
        manager = object.__new__(ModelManager)
        manager.models = {model.name: model}
        manager.ollama = backend
        resolved = []

        async def resolve(name, **kwargs):
            resolved.append(name)
            return model

        async def dispatch(label):
            request = Request({'type': 'http', 'headers': [(b'idempotency-key',label.encode())],
                               'state': {'caller': {'username': label}}})
            result = await server.chat_completions(server.ChatRequest(
                model=model.name, messages=[{"role": "user", "content": label}],
                stream=streaming, max_tokens=2), request)
            if streaming:
                return [chunk async for chunk in result.body_iterator]
            return result

        gate = asyncio.Lock()

        async def call(label):
            if control_gate:
                # Known-good control for the detector, not a production fix:
                # the gate spans response-body consumption, not just headers.
                async with gate:
                    return await dispatch(label)
            return await dispatch(label)

        async def wait_until(predicate):
            async with asyncio.timeout(2):
                while not predicate():
                    await asyncio.sleep(0.005)

        jobs = []
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(server, "mgr", manager), \
             patch.object(server, "_resolve_model", resolve), \
             patch.object(server.prompt_size, "count", return_value=(10, "stub")):
            queue = GatewayQueue(Path(directory) / 'queue.sqlite', server,
                                 {model.name: 'slice-gpu-0'}, lambda _: True)
            previous = server.inference_dispatcher
            server.inference_dispatcher = queue if enabled else None
            try:
                jobs.append(asyncio.create_task(call("first")))
                await wait_until(lambda: len(backend.entered) == 1)
                jobs.append(asyncio.create_task(call("second")))
                if not control_gate and enabled:
                    await wait_until(lambda: queue.supervisor.journal.db.execute(
                        'SELECT COUNT(*) FROM grants').fetchone()[0] == 2)
                    self.assertEqual(len(resolved), 1,
                                     'queued work must not load/resolve a model before admission')
                elif not control_gate:
                    await wait_until(lambda: len(resolved) == 2)
                # Give the second route scheduling opportunities after its
                # model resolution. The first provider is still held by a gate.
                await asyncio.sleep(0.1)
                self.assertEqual(
                    backend.entered, ["first"],
                    "second consumer reached backend before first released: "
                    "in_flight accounting is not exclusive service admission")
            finally:
                backend.gate.set()
                await asyncio.wait_for(asyncio.gather(*jobs), timeout=3)
                await asyncio.gather(*list(queue.supervisor.workers.values()))
                queue.supervisor.close()
                server.inference_dispatcher = previous
        self.assertEqual(backend.entered, ["first", "second"])
        self.assertEqual(model.in_flight, 0)

    async def test_nonstreaming_consumers_wait_in_service(self):
        await self.check_exclusive_dispatch(False)

    async def test_streaming_consumers_wait_in_service(self):
        await self.check_exclusive_dispatch(True)

    async def test_detector_accepts_nonstreaming_serial_control(self):
        await self.check_exclusive_dispatch(False, control_gate=True)

    async def test_detector_accepts_streaming_serial_control(self):
        await self.check_exclusive_dispatch(True, control_gate=True)

    async def test_detector_rejects_unarbitrated_nonstreaming_route(self):
        with self.assertRaisesRegex(AssertionError, 'second consumer reached backend'):
            await self.check_exclusive_dispatch(False, enabled=False)

    async def test_detector_rejects_unarbitrated_streaming_route(self):
        with self.assertRaisesRegex(AssertionError, 'second consumer reached backend'):
            await self.check_exclusive_dispatch(True, enabled=False)


if __name__ == "__main__":
    unittest.main()
