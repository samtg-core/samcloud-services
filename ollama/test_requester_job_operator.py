"""Operator end survives process loss and drains real authenticated API work."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import httpx
import pytest

from .hermes_queue_provider import host_for_home
from .hermes_queue_source import owner_scope
from .inference_arbiter import Conflict
from .inference_consumer import DurableConsumer
from .inference_identity import IdentityStore, LocalSourceRefused
from .requester_job import FinishDelivery, deliver_configured, deliver_finish, local_operation, main
from . import test_inference_jobs_api as fixtures


@pytest.fixture
def owner(tmp_path):
    home = tmp_path / 'profile'; home.mkdir()
    (home / 'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
    host = host_for_home(home)
    scope = owner_scope(host.scope, 'S1')
    store = IdentityStore(host.path)
    store.begin_native_job(scope, 'T1')
    store.observe_native_outcome(scope, 'T1', completed=False, failed=True, interrupted=False)
    store.close()
    return home, host.path, scope


def test_cli_status_and_finish_survive_fresh_process_without_credentials(owner):
    home, path, scope = owner
    env = dict(os.environ)
    env.pop('SAMCLOUD_QUEUE_API_KEY', None); env.pop('SAMCLOUD_QUEUE_BASE_URL', None)
    command = [sys.executable, '-m', 'ollama.requester_job', '--home', str(home), '--session', 'S1']
    cwd = Path(__file__).resolve().parents[1]
    read = subprocess.run([*command, 'status'], env=env, cwd=cwd, capture_output=True, text=True, timeout=10)
    assert read.returncode == 0, read.stderr
    status = json.loads(read.stdout)['parent']
    assert status['state'] == 'recoverable'
    finish = subprocess.run([*command, 'finish', '--revision', status['revision']],
        env=env, cwd=cwd, capture_output=True, text=True, timeout=10)
    assert finish.returncode == 0, finish.stderr
    result = json.loads(finish.stdout)
    assert result['delivery'] == 'pending'
    assert result['parent']['state'] == 'finish_requested'
    assert local_operation(home, 'S1')['parent'] is None
    assert local_operation(home, 'S1', revision=status['revision'])['parent'] == result['parent']
    store = IdentityStore(path)
    try:
        assert store.begin_native_job(scope, 'T1') is None
        assert store.pending_events()[0]['job_key'] == status['job_key']
    finally:
        store.close()


def test_abrupt_exit_after_intent_keeps_exact_finish_and_cannot_end_next_job(owner):
    home, path, scope = owner
    revision = local_operation(home, 'S1')['parent']['revision']
    code = 'import os,sys;from ollama.requester_job import local_operation;local_operation(sys.argv[1],"S1",revision=sys.argv[2]);os._exit(73)'
    process = subprocess.run([sys.executable, '-c', code, str(home), revision],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
    assert process.returncode == 73, process.stderr
    first = local_operation(home, 'S1', revision=revision)['parent']
    store = IdentityStore(path)
    try:
        new = store.begin_native_job(scope, 'T2')
        assert new != first['job_key']
    finally:
        store.close()
    assert local_operation(home, 'S1', revision=revision)['parent'] == first
    assert local_operation(home, 'S1')['parent']['job_key'] == new


def test_stale_operator_revision_cannot_finish_resumed_attempt_or_other_profile(owner, tmp_path):
    home, path, scope = owner
    old = local_operation(home, 'S1')['parent']
    store = IdentityStore(path)
    try:
        assert store.begin_native_job(scope, 'T2') == old['job_key']
    finally:
        store.close()
    with pytest.raises(Conflict):
        local_operation(home, 'S1', revision=old['revision'])
    assert local_operation(home, 'S1')['parent']['turn_id'] == 'T2'
    assert local_operation(home, 'unrelated')['parent'] is None
    assert not (tmp_path / 'unknown').exists()
    with pytest.raises(LocalSourceRefused):
        local_operation(tmp_path / 'unknown', 'S1')
    other = tmp_path / 'other-profile'; other.mkdir()
    (other / 'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
    other_host = host_for_home(other)
    store = IdentityStore(other_host.path)
    try:
        store.begin_native_job(owner_scope(other_host.scope, 'S1'), 'T1')
    finally:
        store.close()
    with pytest.raises(Conflict):
        local_operation(other, 'S1', revision=old['revision'])


def test_compaction_alias_selects_same_parent_and_stale_cli_is_actionable(owner, capsys):
    home, path, scope = owner
    host = host_for_home(home)
    host.link_compaction_session('S1', 'S2', 'T1')
    assert local_operation(home, 'S2')['parent'] == local_operation(home, 'S1')['parent']
    assert main(['--home', str(home), '--session', 'S2', 'finish', '--revision', 'stale']) == 2
    assert json.loads(capsys.readouterr().out)['error'] == 'Conflict'
    assert local_operation(home, 'S2')['parent'] is not None


def test_invalid_endpoint_retains_receipt_without_network_or_secret_output(owner):
    home, path, scope = owner
    status = local_operation(home, 'S1')['parent']
    receipt = local_operation(home, 'S1', revision=status['revision'])['parent']
    source = FinishDelivery(path, receipt['event_id'])
    with patch('httpx.AsyncClient', side_effect=AssertionError('must not connect')):
        for endpoint in ('http://unsafe.invalid/v1', 'https://secret@unsafe.invalid/v1'):
            result = asyncio.run(deliver_configured(source, endpoint, 'credential-never-printed'))
            assert result == {'delivery':'pending', 'observation_error':'LocalSourceRefused'}
    assert source.read()['job_key'] == receipt['job_key']


class OperatorAPI(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.JobAPI.asyncSetUp
    asyncTearDown = fixtures.JobAPI.asyncTearDown
    grant = fixtures.JobAPI.grant
    submit = fixtures.JobAPI.submit
    settle = fixtures.JobAPI.settle

    async def test_operator_end_drains_running_child_and_delivers_only_its_event(self):
        home = Path(self.temp.name) / 'profile'; home.mkdir()
        (home / 'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
        host = host_for_home(home)
        scope = owner_scope(host.scope, 'S1')
        store = IdentityStore(host.path)
        try:
            key = store.begin_native_job(scope, 'T1')
            store.begin_native_job('another-conversation', 'other')
            store.finish_turn('another-conversation', 'other')
        finally:
            store.close()
        grant, child, body = await self.submit('native-main', key)
        _, other, _ = await self.submit('other-requester', owner='assistant')
        await asyncio.sleep(0)
        status = local_operation(home, 'S1')['parent']
        receipt = local_operation(home, 'S1', revision=status['revision'])['parent']
        source = FinishDelivery(host.path, receipt['event_id'])
        consumer = DurableConsumer(self.client)
        real_post = self.client.post
        async def lose_ack(*args, **kwargs):
            await real_post(*args, **kwargs)
            raise httpx.ReadError('lost response after service intake')
        with patch.object(self.client, 'post', side_effect=lose_ack):
            assert (await deliver_finish(consumer, source))['delivery'] == 'pending'
        assert self.s.journal.owns_job('slice-gpu-0')
        assert self.s.journal.execution(child['id'], 'creative')['state'] == 'running'
        assert self.s.journal.execution(other['id'], 'assistant')['state'] == 'queued'
        assert (await deliver_finish(consumer, source))['delivery'] == 'accepted'
        self.gate.set(); await self.settle()
        assert self.s.journal.execution(child['id'], 'creative')['state'] == 'complete'
        assert self.s.journal.execution(other['id'], 'assistant')['state'] == 'complete'
        replay = await self.client.post(f'/v1/inference/grants/{grant}/executions', json=body)
        assert replay.json()['id'] == child['id']
        store = IdentityStore(host.path)
        try:
            assert len(store.pending_events()) == 1
            assert store.pending_events()[0]['job_key'] != key
        finally:
            store.close()
