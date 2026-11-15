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


def creation_payload(api):
    from dataclasses import replace
    from exo.api.types.api import CreateInstanceParams
    from ollama.exo_groups.prepare import canonical
    from test_exo_instance_types import ExplicitInstanceControls
    fixture = ExplicitInstanceControls()
    fixture.setUp()
    try:
        api._leased_profile = replace(api._leased_profile,
            manifest_sha256=fixture.plan.rank_manifest_sha256)
        return CreateInstanceParams.model_validate_json(canonical(fixture.build()))
    finally:
        fixture.doCleanups()


async def test_leased_instance_creation_uses_prepared_card_without_model_name_lookup(tmp_path):
    from unittest.mock import AsyncMock, patch
    from exo.shared.models.model_cards import ModelCard
    from exo.shared.types.memory import Memory
    api = make_api(tmp_path)
    payload = creation_payload(api)
    expected = next(iter(payload.instance.shard_assignments.runner_to_shard.values())).model_card
    lookup = AsyncMock(side_effect=AssertionError('leased request must not look up a model name'))
    with patch.object(ModelCard, 'load', lookup), patch.object(api,
            '_calculate_total_available_memory', return_value=Memory.from_gb(1000)):
        result = await api.create_instance(payload)
    lookup.assert_not_awaited()
    assert result.model_card == expected
    assert api.sent[0].instance == payload.instance


async def test_ordinary_instance_creation_keeps_existing_model_card_lookup(tmp_path):
    from unittest.mock import AsyncMock, patch
    from exo.shared.models.model_cards import ModelCard
    from exo.shared.types.memory import Memory
    api = make_api(tmp_path)
    payload = creation_payload(api)
    api._leased_profile = None
    card = next(iter(payload.instance.shard_assignments.runner_to_shard.values())).model_card
    lookup = AsyncMock(return_value=card)
    with patch.object(ModelCard, 'load', lookup), patch.object(api,
            '_calculate_total_available_memory', return_value=Memory.from_gb(1000)):
        result = await api.create_instance(payload)
    lookup.assert_awaited_once_with(payload.instance.shard_assignments.model_id)
    assert result.model_card == card
    assert len(api.sent) == 1


@pytest.mark.parametrize('change', ['namespace', 'runner', 'endpoint', 'card',
                                  'layers', 'timeout', 'remote_code', 'image'])
async def test_leased_instance_refuses_changed_wire_before_command_or_lookup(tmp_path, change):
    from unittest.mock import AsyncMock, patch
    from exo.api.types.api import CreateInstanceParams
    from exo.shared.models.model_cards import ModelCard
    api = make_api(tmp_path)
    payload = creation_payload(api)
    wire = payload.model_dump(mode='json')
    instance = wire['instance']['MlxRingInstance']
    assignments = instance['shard_assignments']
    shards = list(assignments['runner_to_shard'].values())
    first = shards[0]['PipelineShardMetadata']
    if change == 'namespace':
        instance['instance_id'] = 'f' * 64
    elif change == 'runner':
        runner, shard = assignments['runner_to_shard'].popitem()
        assignments['runner_to_shard']['f' * 64] = shard
        for node, value in assignments['node_to_runner'].items():
            if value == runner:
                assignments['node_to_runner'][node] = 'f' * 64
    elif change == 'endpoint':
        next(iter(instance['hosts_by_node'].values()))[0]['ip'] = '192.168.50.99'
    elif change == 'card':
        first['model_card']['context_length'] += 1
    elif change == 'layers':
        first['end_layer'] += 1
    elif change == 'timeout':
        first['should_timeout'] = 3.0
    elif change == 'remote_code':
        for shard in shards:
            shard['PipelineShardMetadata']['model_card']['trust_remote_code'] = True
    else:
        for shard in shards:
            shard['PipelineShardMetadata']['model_card']['tasks'] = ['TextToImage']
    altered = CreateInstanceParams.model_validate_json(json.dumps(wire))
    lookup = AsyncMock(side_effect=AssertionError('no name lookup on leased profile'))
    with patch.object(ModelCard, 'load', lookup), pytest.raises(HTTPException) as error:
        await api.create_instance(altered)
    assert error.value.status_code == 400
    lookup.assert_not_awaited()
    assert not api.sent


async def test_actual_private_instance_http_reply_preserves_pinned_card_aliases(tmp_path):
    from unittest.mock import AsyncMock, patch
    from exo.shared.models.model_cards import ModelCard
    from exo.shared.types.memory import Memory
    api = make_api(tmp_path)
    payload = creation_payload(api)
    card = next(iter(payload.instance.shard_assignments.runner_to_shard.values())).model_card
    api.app.post('/instance')(api.create_instance)
    lookup = AsyncMock(side_effect=AssertionError('leased path must not resolve a model name'))
    with patch.object(ModelCard, 'load', lookup), patch.object(api,
            '_calculate_total_available_memory', return_value=Memory.from_gb(1000)):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),
                                    base_url='http://private') as client:
            response = await client.post('/instance', json=payload.model_dump(mode='json'),
                headers={'Authorization': 'Bearer ' + 'a' * 64})
    assert response.status_code == 200, response.text
    assert response.json()['model_card'] == card.model_dump(mode='json', by_alias=True)
    assert response.json()['command_id'] == str(api.sent[0].command_id)
    lookup.assert_not_awaited()
