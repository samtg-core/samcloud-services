"""Operational launcher checks with no listener, credentials or provider I/O."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from . import config, server
from . import inference_root_launch as launch
from .inference_bootstrap import QueueSettings
from .inference_root_bootstrap import RootQueueStartup
from .manager import Backend


@pytest.fixture
def files(tmp_path, monkeypatch):
    queue = tmp_path/'queue.json'
    queue.write_text(json.dumps({'journal': str(tmp_path/'queue.sqlite'),
        'models': {'text': 'host/gpu', 'vision': 'host/gpu'},
        'shared_queue_enabled': True}))
    body = {'version': 1, 'registry': 'https://registry.test/api/v1',
        'actor': 'provider', 'service': 'host/models', 'resource': 'host/gpu',
        'journal': str(tmp_path/'root.sqlite'),
        'backends': {'text': 'ollama', 'vision': 'mlx-vlm'}}
    root = tmp_path/'root.json'
    root.write_text(json.dumps(body))
    monkeypatch.setattr(config, 'SC_BASE', body['registry'])
    monkeypatch.setattr(config, 'SC_SERVICE_ID', body['service'])
    monkeypatch.setattr(config, 'SC_RESOURCE_ID', body['resource'])
    monkeypatch.setattr(config, 'EXO_ENABLED', False)
    monkeypatch.setattr(config, 'OLLAMA_BASE', 'http://127.0.0.1:11434')
    monkeypatch.setattr(config, 'VLM_HOST', '127.0.0.1')
    monkeypatch.setattr(server, 'SC_TOKEN', 'fixture-only-never-sent')
    for name in ('inference_startup', 'inference_runtime', 'inference_dispatcher'):
        monkeypatch.setattr(server, name, None)
    monkeypatch.setattr(server, 'mgr', None)
    run = Mock()
    monkeypatch.setattr('uvicorn.run', run)
    def no_network(*args, **kwargs):
        pytest.fail('launcher attempted network before the service lifespan')
    monkeypatch.setattr('httpx.Client.send', no_network)
    return SimpleNamespace(queue=queue, root=root, body=body, run=run,
        args=['--queue-config', str(queue), '--root-config', str(root), '--work-permission', 'allow'])


def test_offline_check_creates_no_journals_and_opens_no_client(files, monkeypatch, capsys):
    monkeypatch.setattr(launch, 'RootIntentStore', Mock(side_effect=AssertionError('journal opened')))
    monkeypatch.setattr('httpx.Client', Mock(side_effect=AssertionError('HTTP client opened')))
    launch.main(files.args+['--check'])
    result = json.loads(capsys.readouterr().out)
    assert result['checked'] is True and result['activated'] is False
    assert len(result['root_sha256']) == len(result['queue_sha256']) == 64
    assert server.inference_startup is None and not files.run.called
    assert not list(files.root.parent.glob('*.sqlite'))


@pytest.mark.parametrize('allowed', [True, False])
def test_launcher_binds_matching_root_runtime_only(files, allowed):
    args = files.args[:-1]+['allow' if allowed else 'deny']
    launch.main(args+['--port', '8800'])
    startup = server.inference_startup
    assert isinstance(startup, RootQueueStartup)
    assert startup.work_permitted is allowed
    assert dict(startup.models) == {'text': Backend.OLLAMA, 'vision': Backend.VLM}
    assert startup.client.store.actor == 'provider'
    assert startup.client.store.service == 'host/models'
    assert startup.client.store.path == files.root.parent/'root.sqlite'
    # The mocked server returned without attachment, so the launcher closes it.
    assert startup.client._http.is_closed
    assert startup.client.store.prepare('read-only-proof', 'host/gpu')['root_id'] is None
    files.run.assert_called_once_with(server.app, host='127.0.0.1', port=8800, workers=1)
    assert not (files.root.parent/'queue.sqlite').exists()


@pytest.mark.parametrize('field,value', [
    ('version', True), ('version', 2), ('actor', ''), ('journal', 'relative.sqlite'),
    ('backends', {'text': 'ollama'}), ('backends', {'text': 'ollama', 'vision': 'exo'}),
    ('resource', 'other/gpu'), ('registry', 'http://registry.test/api/v1'),
    ('registry', 'https://credential:SECRET@registry.test/api/v1'),
])
def test_invalid_plan_refuses_without_exposing_values_or_mutating(files, field, value):
    files.body[field] = value
    files.root.write_text(json.dumps(files.body))
    with pytest.raises(ValueError, match='^Invalid root startup configuration$'):
        launch.main(files.args)
    assert server.inference_startup is None and not files.run.called
    assert not list(files.root.parent.glob('*.sqlite'))


def test_duplicate_config_and_journal_alias_are_rejected(files):
    raw = files.root.read_text()
    files.root.write_text(raw[:-1]+',"actor":"other"}')
    with pytest.raises(ValueError, match='Invalid root startup'):
        launch.main(files.args)
    files.body['journal'] = str(files.root.parent/'queue.sqlite')
    files.root.write_text(json.dumps(files.body))
    with pytest.raises(ValueError, match='Invalid root startup'):
        launch.main(files.args)
    assert not list(files.root.parent.glob('*.sqlite'))


def test_existing_hardlinked_journal_alias_is_rejected_without_changing_it(files):
    queue_journal = files.root.parent/'queue.sqlite'
    queue_journal.write_bytes(b'existing-journal-fixture')
    root_journal = files.root.parent/'root.sqlite'
    root_journal.hardlink_to(queue_journal)
    with pytest.raises(ValueError, match='Invalid root startup configuration'):
        launch.main(files.args)
    assert root_journal.read_bytes() == queue_journal.read_bytes() == b'existing-journal-fixture'
    assert server.inference_startup is None and not files.run.called


@pytest.mark.parametrize('change', ['registry', 'service', 'resource', 'exo', 'credential', 'live', 'manager'])
def test_service_mismatch_or_existing_owner_cannot_open_root_journal(files, monkeypatch, change):
    if change == 'registry': monkeypatch.setattr(config, 'SC_BASE', 'https://another.test/api/v1')
    if change == 'service': monkeypatch.setattr(config, 'SC_SERVICE_ID', 'other/models')
    if change == 'resource': monkeypatch.setattr(config, 'SC_RESOURCE_ID', 'other/gpu')
    if change == 'exo': monkeypatch.setattr(config, 'EXO_ENABLED', True)
    if change == 'credential': monkeypatch.setattr(server, 'SC_TOKEN', '')
    if change == 'live': monkeypatch.setattr(server, 'inference_runtime', object())
    if change == 'manager': monkeypatch.setattr(server, 'mgr', object())
    with pytest.raises((ValueError, RuntimeError)):
        launch.main(files.args)
    assert not files.run.called and not list(files.root.parent.glob('*.sqlite'))


@pytest.mark.parametrize('rows,vision,managed,expected', [
    ([], True, {}, True), ([], False, {}, False), ([], None, {}, False),
    ([{'name': 'foreign-model'}], True, {}, False),
    ({}, True, {}, False), (None, True, {}, False), ([], True, {'owned': object()}, False),
])
def test_cold_observation_requires_both_complete_backend_absences(monkeypatch, rows, vision, managed, expected):
    monkeypatch.setattr('ollama.inference_shared_resource.cold_vision_absent', lambda: vision)
    manager = SimpleNamespace(models=managed, ollama=SimpleNamespace(list_running=lambda: rows))
    assert launch.confirm_cold(SimpleNamespace(mgr=manager)) is expected


def test_inventory_failure_is_unknown_not_empty(monkeypatch):
    def unavailable(): raise OSError('fixture inventory unavailable')
    manager = SimpleNamespace(models={}, ollama=SimpleNamespace(list_running=unavailable))
    assert launch.confirm_cold(SimpleNamespace(mgr=manager)) is False


@pytest.mark.parametrize('attached', [False, True])
def test_launcher_error_closes_only_unattached_control_client(files, monkeypatch, attached):
    def fail_serve(*args, **kwargs):
        if attached:
            server.inference_runtime = object()
        raise RuntimeError('fixture serve failure')
    files.run.side_effect = fail_serve
    with pytest.raises(RuntimeError, match='fixture serve failure'):
        launch.main(files.args)
    client = server.inference_startup.client
    assert client._http.is_closed is (not attached)
    client.close()  # fixture owns the synthetic attached runtime


@pytest.mark.parametrize('field,value', [('reservation_mb', {'text': 10, 'vision': 20}),
    ('shared_queue_wait_s', 3600)])
def test_root_launch_rejects_legacy_accounting_configuration(files, field, value):
    body = json.loads(files.queue.read_text())
    body[field] = value
    files.queue.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='Invalid root startup configuration'):
        launch.main(files.args)
    assert server.inference_startup is None and not files.run.called
    assert not list(files.root.parent.glob('*.sqlite'))


@pytest.mark.parametrize('url,host,expected', [
    ('http://localhost:11434', '127.0.0.1', True),
    ('http://[::1]:11434', '::1', True),
    ('http://192.0.2.1:11434', '127.0.0.1', False),
    ('http://127.0.0.1:11434', '0.0.0.0', False),
    ('http://user:SECRET@localhost:11434', 'localhost', False),
    ('http://localhost:11434/proxy', 'localhost', False),
    ('http://localhost:bad', 'localhost', False),
])
def test_backend_binding_is_local_and_unambiguous(url, host, expected):
    assert launch.local_backends(SimpleNamespace(OLLAMA_BASE=url, VLM_HOST=host)) is expected


def test_remote_backend_refuses_before_journal_or_listener(files, monkeypatch):
    monkeypatch.setattr(config, 'OLLAMA_BASE', 'http://192.0.2.1:11434')
    with pytest.raises(ValueError, match='differs from this local service'):
        launch.main(files.args)
    assert server.inference_startup is None and not files.run.called
    assert not list(files.root.parent.glob('*.sqlite'))
