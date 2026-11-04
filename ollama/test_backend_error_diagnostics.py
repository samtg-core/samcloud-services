"""Offline client-to-SSE error evidence; no backend, inference or registry."""
import ast
import asyncio
import io
import json
import logging
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

import httpx
from . import ollama_client as client

CANARY = "SECRET-CANARY"

class Content:
    def __init__(self, body=b"", chunks=(), hang=False):
        self.body, self.position, self.chunks, self.hang = body, 0, list(chunks), hang
    async def read(self, size):
        if self.hang:
            await asyncio.sleep(60)
        value = self.body[self.position:self.position+size]
        self.position += len(value)
        return value
    def __aiter__(self):
        return self
    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return json.dumps(self.chunks.pop(0)).encode()

class Response:
    def __init__(self, status=200, body=b"", chunks=(), hang=False):
        self.status, self.content = status, Content(body, chunks, hang)
    def raise_for_status(self):
        if self.status >= 400:
            raise AssertionError("Status detail was discarded")
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False

class Session:
    def __init__(self, response):
        self.response, self.closed = response, False
    def post(self, *args, **kwargs):
        return self.response
    async def close(self):
        self.closed = True

def fake_aiohttp(response):
    session = Session(response)
    module = types.SimpleNamespace(ClientSession=lambda:session,
                                   ClientTimeout=lambda **kwargs:None)
    return module, session

def bare_client():
    value = client.OllamaClient.__new__(client.OllamaClient)
    value.base_url = "http://offline"
    value._with_num_ctx = lambda model, payload:payload
    return value

class Diagnostics(unittest.IsolatedAsyncioTestCase):
    async def frames(self, response):
        """Execute the exact server stream body with the real client iterator."""
        tree = ast.parse((Path(__file__).parent/"server.py").read_text())
        stream = next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef)
                      and n.name=="stream" and any(isinstance(v,ast.Name) and v.id=="saw_tool_calls"
                                                  for v in ast.walk(n)))
        helpers = [next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
                   for name in ("_failure_frames","_fix_tool_calls")]
        helpers.insert(0,next(n for n in tree.body if isinstance(n,ast.Assign)
                              and any(isinstance(t,ast.Name) and t.id=="_FAILURE_FINISH_REASON"
                                      for t in n.targets)))
        value = bare_client()
        async def chat_stream(*args, **kwargs):
            async for chunk in value._stream_with_deadlines("/api/chat", {}, 1):
                yield chunk
        namespace = dict(json=json, asyncio=asyncio, time=time, log=logging.getLogger("offline-server"),
                         mgr=types.SimpleNamespace(ollama=types.SimpleNamespace(chat_stream=chat_stream)),
                         mm=types.SimpleNamespace(name="fixture"),ollama_messages=[],n_prompt=1,
                         ollama_kwargs={},queue_owned=True,req=types.SimpleNamespace(stream_options=None),
                         _log_ollama_timings=lambda *args:None,
                         OllamaBackendError=client.OllamaBackendError,
                         OllamaStalled=client.OllamaStalled,OllamaRequestTimeout=client.OllamaRequestTimeout)
        exec(compile(ast.Module(body=helpers+[stream],type_ignores=[]),"exact-server-stream","exec"),namespace)
        fake, session = fake_aiohttp(response)
        with patch.dict(sys.modules, {"aiohttp":fake}):
            frames = [frame async for frame in namespace["stream"]()]
        self.assertTrue(session.closed)
        self.assertEqual(frames[-1], "data: [DONE]\n\n")
        payloads = [json.loads(frame[6:]) for frame in frames[:-1]]
        failures = [frame["error"] for frame in payloads if frame.get("error")]
        self.assertEqual(len(failures),1)
        self.assertEqual(failures[0]["type"],"backend_error")
        self.assertFalse(any(choice.get("finish_reason")=="stop"
                             for frame in payloads for choice in frame.get("choices",[])))
        return payloads, failures[0]

    async def test_status_known_prefix_is_safe(self):
        response = Response(500,json.dumps({"error":"error parsing tool call: "+CANARY}).encode())
        with self.assertLogs(level="WARNING") as logs:
            frames, error = await self.frames(response)
        self.assertEqual(error["subtype"],"tool_parse")
        self.assertEqual(error["phase"],"status")
        self.assertEqual(error["chunks_before"],0)
        self.assertNotIn(CANARY,json.dumps(frames)+" ".join(logs.output))

    async def test_unknown_status_message_cannot_escape(self):
        with self.assertLogs(level="WARNING") as logs:
            frames, error = await self.frames(Response(500,json.dumps({"error":CANARY}).encode()))
        self.assertEqual(error["subtype"],"unclassified")
        self.assertNotIn(CANARY,json.dumps(frames)+" ".join(logs.output))

    async def test_runner_crash_prefix_is_safe(self):
        with self.assertLogs(level="WARNING") as logs:
            frames, error = await self.frames(Response(500,json.dumps({
                "error":"model runner has unexpectedly stopped: "+CANARY}).encode()))
        self.assertEqual(error["subtype"],"runner_stopped")
        self.assertNotIn(CANARY,json.dumps(frames)+" ".join(logs.output))

    async def test_inband_error_after_content_is_failure(self):
        with self.assertLogs(level="WARNING") as logs:
            frames, error = await self.frames(Response(chunks=[
                {"message":{"content":"partial"}},{"error":"context canceled: "+CANARY}]))
        self.assertEqual(frames[0]["choices"][0]["delta"]["content"],"partial")
        self.assertEqual(error["phase"],"in_stream")
        self.assertEqual(error["chunks_before"],1)
        self.assertNotIn(CANARY,json.dumps(frames)+" ".join(logs.output))

    async def test_malformed_and_oversized_bodies_fail_safely(self):
        for body in (b"not JSON "+CANARY.encode(),
                     json.dumps({"error":CANARY+"x"*6000}).encode()):
            with self.subTest(size=len(body)), self.assertLogs(level="WARNING") as logs:
                response = Response(500,body)
                frames, error = await self.frames(response)
                self.assertEqual(error["subtype"],"unclassified")
                self.assertLessEqual(response.content.position,client.ERROR_BODY_LIMIT)
                self.assertNotIn(CANARY,json.dumps(frames)+" ".join(logs.output))
                if len(body)>client.ERROR_BODY_LIMIT:
                    self.assertTrue(error["detail_truncated"])

    async def test_error_body_silence_has_a_separate_short_read_bound(self):
        with patch.object(client,"ERROR_BODY_TIMEOUT",.001), self.assertLogs(level="WARNING"):
            _, error = await self.frames(Response(500,hang=True))
        self.assertTrue(error["detail_truncated"])
        self.assertEqual(error["subtype"],"unclassified")

class SyncDiagnostics(unittest.TestCase):
    def test_sync_status_error_has_the_same_safe_type(self):
        for operation in ("chat","generate"):
            with self.subTest(operation=operation), self.assertLogs(level="WARNING") as logs:
                value = bare_client()
                with httpx.Client(base_url="http://offline",transport=httpx.MockTransport(
                    lambda request:httpx.Response(500,json={"error":"error parsing tool call: "+CANARY}))) as http:
                    value._http = http
                    with self.assertRaises(client.OllamaBackendError) as raised:
                        list(value.chat("fixture",[]) if operation=="chat" else value.generate("fixture",""))
                self.assertEqual(raised.exception.phase,"status")
                self.assertEqual(raised.exception.subtype,"tool_parse")
                self.assertNotIn(CANARY,str(raised.exception)+repr(raised.exception)+" ".join(logs.output))

    def test_sync_inband_error_preserves_partial_count(self):
        value = bare_client()
        body=b'{"message":{"content":"partial"}}\n{"error":"context canceled"}\n'
        with httpx.Client(base_url="http://offline",transport=httpx.MockTransport(
            lambda request:httpx.Response(200,content=body))) as http, self.assertLogs(level="WARNING"):
            value._http=http
            stream=value.chat("fixture",[])
            self.assertEqual(next(stream)["message"]["content"],"partial")
            with self.assertRaises(client.OllamaBackendError) as raised:
                next(stream)
            self.assertEqual(raised.exception.chunks_before,1)


class CompletionDiagnostics(unittest.IsolatedAsyncioTestCase):
    async def frames(self, status, body):
        """Run the exact completions route and its real sync client iterator."""
        tree = ast.parse((Path(__file__).parent/"server.py").read_text())
        route = next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef)
                     and n.name=="completions")
        route.decorator_list = []
        value = bare_client()
        async def resolve(model):
            return types.SimpleNamespace(name=model,backend="ollama")
        namespace = dict(json=json,CompletionRequest=types.SimpleNamespace,
                         _resolve_model=resolve,Backend=types.SimpleNamespace(EXO="exo",OLLAMA="ollama"),
                         mgr=types.SimpleNamespace(ollama=value),
                         OllamaBackendError=client.OllamaBackendError,
                         StreamingResponse=lambda iterator,**kwargs:iterator)
        exec(compile(ast.Module(body=[route],type_ignores=[]),"exact-completions-route","exec"),namespace)
        with httpx.Client(base_url="http://offline",transport=httpx.MockTransport(
                lambda request:httpx.Response(status,content=body))) as http:
            value._http=http
            response = await namespace["completions"](
                types.SimpleNamespace(model="fixture",prompt="fixture",stream=True))
            return [json.loads(line) for line in response]

    async def test_inband_error_ends_ndjson_with_safe_failure(self):
        partial = {"response":"partial","done":False}
        body = (json.dumps(partial)+"\n"+json.dumps({"error":
                "an error was encountered while running the model: "+CANARY})+"\n").encode()
        with self.assertLogs(level="WARNING") as logs:
            frames = await self.frames(200,body)
        self.assertEqual(frames[0],partial)
        self.assertEqual(len(frames),2)
        self.assertEqual(frames[-1]["error"],"backend_error")
        self.assertTrue(frames[-1]["done"])
        self.assertEqual(frames[-1]["done_reason"],"error")
        self.assertEqual(frames[-1]["diagnostics"]["chunks_before"],1)
        self.assertEqual(frames[-1]["diagnostics"]["phase"],"in_stream")
        self.assertNotIn(CANARY,json.dumps(frames)+" ".join(logs.output))

    async def test_status_error_has_a_final_safe_ndjson_line(self):
        with self.assertLogs(level="WARNING") as logs:
            frames = await self.frames(500,json.dumps({"error":CANARY}).encode())
        self.assertEqual(len(frames),1)
        self.assertEqual(frames[0]["error"],"backend_error")
        self.assertTrue(frames[0]["done"])
        self.assertEqual(frames[0]["done_reason"],"error")
        self.assertEqual(frames[0]["diagnostics"]["phase"],"status")
        self.assertNotIn(CANARY,json.dumps(frames)+" ".join(logs.output))

    async def test_successful_ndjson_stays_unchanged(self):
        chunks = [{"response":"partial","done":False},{"done":True}]
        frames = await self.frames(200,("\n".join(json.dumps(v) for v in chunks)+"\n").encode())
        self.assertEqual(frames,chunks)

if __name__=="__main__":
    unittest.main()
