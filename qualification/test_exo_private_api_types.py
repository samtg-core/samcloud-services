"""Actual EXO HTTP handlers; command delivery injected, no node/model/server."""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
import httpx
import pytest

from exo.api.main import API
from exo.api.types import ChatCompletionRequest
from exo.shared.types.state import State
from exo.shared.types.worker.runners import RunnerRunning, RunnerReady
from exo.shared.types.events import IndexedEvent, RunnerStatusUpdated, TaskCreated
from exo.shared.types.tasks import TextGeneration as GenerationTask
from exo.utils.leased_commands import LeasedCommands
from exo.utils.leased_runtime import LeasedAPIGuard, LeasedRuntimeProfile


class HandlerAPI(API):
    def __init__(self, profile, state):
        # Use actual request/handler/state code; replace only command transport.
        self._leased_profile = profile
        self._leased_commands = LeasedCommands()
        self.state = state
        self.sent = []
        self.app = FastAPI()
        self.app.add_middleware(LeasedAPIGuard, key=b'a' * 64)
        self.app.get('/leased/commands/{command_id}/end')(self.leased_command_end)

    async def _send(self, command):
        self.sent.append(command)

    async def start_chat(self, stream):
        return await self.chat_completions(ChatCompletionRequest(
            model=next(iter(self.state.instances.values())).shard_assignments.model_id,
            messages=[{'role': 'user', 'content': 'hello'}], stream=stream))

    def task_created(self, index=0):
        command = self.sent[0]
        task = GenerationTask(command_id=command.command_id,
            instance_id=next(iter(self.state.instances)), task_params=command.task_params)
        self._leased_commands.observe(IndexedEvent(idx=index,
            event=TaskCreated(task_id=task.task_id, task=task)))

    def end_ranks(self, after=0):
        for rank, runner in enumerate(self.state.runners):
            self._leased_commands.observe(IndexedEvent(idx=after + rank * 2 + 1,
                event=RunnerStatusUpdated(runner_id=runner, runner_status=RunnerRunning())))
            self._leased_commands.observe(IndexedEvent(idx=after + rank * 2 + 2,
                event=RunnerStatusUpdated(runner_id=runner, runner_status=RunnerReady())))


def make_api(root):
    wire = json.loads((Path(__file__).parent / 'fixtures/exo-instance-v1.json').read_text())
    state = State.model_validate_json(json.dumps(wire['state']))
    profile = LeasedRuntimeProfile('group_' + 'c' * 24, 1, 'slice',
        wire['plan']['rank_manifest_sha256'], root / 'ranks', root / 'state',
        ('192.168.50.11', '192.168.50.12', '192.168.50.13'), root / 'key',
        52515, 52514, 52516, 52518)
    return HandlerAPI(profile, state)


@pytest.mark.parametrize('stream', [True, False])
async def test_actual_chat_handler_exposes_uuid_and_reserves_only_one_command(tmp_path, stream):
    api = make_api(tmp_path)
    result = await api.start_chat(stream)
    assert result.headers['X-EXO-Command-ID'] == str(api.sent[0].command_id)
    with pytest.raises(HTTPException) as error:
        await api.start_chat(stream)
    assert error.value.status_code == 409
    assert len(api.sent) == 1


async def test_actual_private_http_end_waits_for_all_ranks_and_requires_key(tmp_path):
    api = make_api(tmp_path)
    result = await api.start_chat(True)
    command = result.headers['X-EXO-Command-ID']
    api.task_created()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://private') as client:
        path = '/leased/commands/' + command + '/end'
        assert (await client.get(path)).status_code == 401
        waiter = asyncio.create_task(client.get(path, headers={'Authorization': 'Bearer ' + 'a' * 64}))
        await asyncio.sleep(0)
        assert not waiter.done()
        api.end_ranks()
        response = await waiter
        assert response.status_code == 200
        proof = response.json()
        assert proof['command_id'] == command
        assert proof['group_id'] == api._leased_profile.group_id
        assert proof['rank_manifest_sha256'] == api._leased_profile.manifest_sha256
        assert proof['runner_ids'] == sorted(api.state.runners)
        assert proof['end_event_idx'] == 6
        assert len(api.sent) == 1


async def test_ordinary_profile_has_no_private_end_endpoint_or_command_header(tmp_path):
    api = make_api(tmp_path)
    api._leased_profile = None
    result = await api.start_chat(True)
    assert 'X-EXO-Command-ID' not in result.headers
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://private') as client:
        response = await client.get('/leased/commands/unknown/end',
                                    headers={'Authorization': 'Bearer ' + 'a' * 64})
        assert response.status_code == 404
