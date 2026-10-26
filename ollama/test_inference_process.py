"""Real client-process death/reopen against a local durable HTTP job service.

Not native JSON-RPC/host recovery, physical leases, or provider qualification.
The deliberately killed process owns only observation and persisted call keys.
"""
import asyncio
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest

from fastapi import FastAPI, Request
import uvicorn

from .inference_api import build_router
from .inference_supervisor import Supervisor


CLIENT = r'''
import asyncio, json, sys
from ollama.inference_identity import IdentityStore
from ollama.inference_sdk import QueuedClient

body = {'model':'synthetic', 'messages':[{'role':'user','content':'synthetic input'}],
        'stream':False}
store = IdentityStore(sys.argv[1])
try:
    keys = store.bind_execution('isolated-profile/conversation', 'accepted-input:1',
                                'held-tail:1', body)
finally:
    store.close()
client = QueuedClient(api_key='synthetic-only', base_url=sys.argv[2],
                     poll_interval=.02, grant_lifetime='execution')
try:
    answer = client.chat.completions.create(**body, extra_headers=keys.headers())
    print(json.dumps({'answer':answer.choices[0].message.content}), flush=True)
finally:
    asyncio.run(client.aclose())
'''


class ObserverProcessRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_process_kill_and_same_intent_reopen_does_not_redispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            entered, release = asyncio.Event(), asyncio.Event()
            dispatches, submitted = [], []

            async def execute(grant, body):
                dispatches.append(grant['active_execution'])
                entered.set()
                await release.wait()
                return {'kind':'json', 'body':{
                    'id':'synthetic-answer', 'object':'chat.completion', 'created':0,
                    'model':'synthetic', 'choices':[{'index':0, 'finish_reason':'stop',
                        'message':{'role':'assistant','content':'saved answer'}}]}}

            supervisor = Supervisor(Path(directory)/'service.sqlite', execute, lambda _:True)
            app = FastAPI()

            @app.middleware('http')
            async def authenticate(request: Request, call_next):
                # Synthetic, local-only credential; no live auth or credentials.
                if request.headers.get('authorization') == 'Bearer synthetic-only':
                    request.state.caller = {'username':'synthetic-owner'}
                response = await call_next(request)
                if request.method == 'POST' and request.url.path.endswith('/executions'):
                    submitted.append(response.status_code)
                return response

            def resolve(model):
                if model != 'synthetic':
                    raise ValueError('unconfigured test model')
                return 'synthetic-resource', model

            app.include_router(build_router(supervisor, resolve, lambda model, body:body))
            listener = socket.socket()
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            listener.setblocking(False)
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='off',
                                                  access_log=False))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            children = []

            async def wait_until(predicate):
                async with asyncio.timeout(10):
                    while not predicate():
                        if serving.done():
                            await serving
                            raise AssertionError('local test HTTP service stopped')
                        await asyncio.sleep(.01)

            async def start_client():
                process = await asyncio.create_subprocess_exec(sys.executable, '-c', CLIENT,
                    str(Path(directory)/'client.sqlite'), f'http://127.0.0.1:{port}/v1',
                    cwd=Path(__file__).resolve().parent.parent,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                children.append(process)
                return process

            try:
                await wait_until(lambda: server.started)
                first = await start_client()
                await asyncio.wait_for(entered.wait(), 10)
                await wait_until(lambda: len(submitted) == 1)
                execution = dispatches[0]
                self.assertIsNone(first.returncode)
                first.kill()  # Only this test's observer process, never the service.
                await asyncio.wait_for(first.communicate(), 10)
                self.assertNotEqual(first.returncode, 0)
                self.assertEqual(supervisor.journal.execution(execution, 'synthetic-owner')['state'], 'running')
                self.assertFalse(supervisor.workers[execution].done())
                self.assertFalse(release.is_set())

                second = await start_client()
                await wait_until(lambda: len(submitted) == 2)
                self.assertEqual(submitted, [202, 202])
                self.assertEqual(dispatches, [execution])
                self.assertIsNone(second.returncode)
                release.set()
                output, error = await asyncio.wait_for(second.communicate(), 10)
                self.assertEqual(second.returncode, 0, error.decode())
                self.assertEqual(json.loads(output), {'answer':'saved answer'})
                self.assertEqual(dispatches, [execution])
                self.assertEqual(supervisor.journal.db.execute(
                    'SELECT COUNT(*) FROM executions').fetchone()[0], 1)
                self.assertEqual(supervisor.journal.execution(execution, 'synthetic-owner')['state'], 'complete')
            finally:
                release.set()
                for child in children:
                    if child.returncode is None:
                        child.kill()
                    await asyncio.wait_for(child.communicate(), 10)
                await asyncio.gather(*list(supervisor.workers.values()))
                server.should_exit = True
                await asyncio.wait_for(serving, 10)
                listener.close()
                supervisor.close()


if __name__ == '__main__':
    unittest.main()
