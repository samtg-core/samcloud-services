"""Provider reads against the real registry's isolated HTTP/SQLite contract.

Run with the selected registry source on PYTHONPATH and ``-p conftest``.
The registry isolation plugin must load before its app. This file never reads
workspace credentials or contacts the production plane; leases are test rows.
"""

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

import db
from test_registry import _register_device

from ollama import test_exo_group_plan as fixtures
from ollama.exo_groups.acquire import GroupAcquisition
from ollama.exo_groups.authority import AuthorityReadUnavailable, GroupAuthorityReader
from ollama.exo_groups.intents import GroupIntents
from ollama.exo_groups.demand import AcquisitionDemand
from ollama.exo_groups.plane import CoordinatorPlane
from ollama.exo_groups.plan import GroupPlan, bind_admitted_group
from ollama.inference_arbiter import Arbiter
from ollama.inference_supervisor import Supervisor

pytest_plugins = ["test_registry"]


@pytest.fixture
def placement(client):
    fixture = fixtures.GroupBindingControls()
    fixture.setUp()
    raw = fixture.raw
    raw["owner_id"] = client.get("/api/v1/auth/verify").json()["owner_id"]
    plan = GroupPlan.parse(raw)
    for member in plan.members:
        device, resource = member.resource_id.split("/")
        assert _register_device(client, device, device).status_code == 200
        response = client.post("/api/v1/resources", json={
            "device_id": device, "name": resource, "type": "memory",
            "provider": "test", "specs": {"ram_mb": 10000},
            "allocation_mode": "shared",
        })
        assert response.status_code == 200, response.text
        db.execute("UPDATE resources SET expiry_policy='hold' WHERE id=?",
                   (member.resource_id,))
    return plan, fixture.manifest


@pytest.fixture
def admitted(client, placement):
    plan, _ = placement
    response = client.post("/api/v1/lease-groups", json=plan.request())
    assert response.status_code == 200, response.text
    group = response.json()
    assert group["phase"] == "admitted"
    leases = {
        member["lease_id"]: client.get("/api/v1/leases/" + member["lease_id"]).json()
        for member in group["members"]
    }
    launches = bind_admitted_group(plan, group, leases,
        expected_generation=1, now=datetime.now(timezone.utc))
    return plan, group, launches


class ObservedRegistry(httpx.AsyncBaseTransport):
    """Real ASGI app; record the provider's requests, optionally race a fence."""

    def __init__(self, client, before=None):
        self.inner = httpx.ASGITransport(app=client._client.app)
        self.requests = []
        self.before = before

    async def handle_async_request(self, request):
        self.requests.append((request.method, request.url.path))
        if self.before is not None:
            self.before(request)
        return await self.inner.handle_async_request(request)

    async def aclose(self):
        await self.inner.aclose()


def reader_for(client, admitted, *, before=None, header=None):
    plan, _, launches = admitted
    transport = ObservedRegistry(client, before)
    return GroupAuthorityReader(plan, launches,
        header or (lambda: client._headers["Authorization"]),
        transport=transport), transport


def command(client, group, action, **extra):
    response = client.post(f"/api/v1/lease-groups/{group['id']}/{action}",
        json={"generation": 1, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def assert_still_charged(*, expired=0):
    assert db.query_one("SELECT COUNT(*) n FROM leases")["n"] == 3
    assert db.query_one("SELECT COUNT(*) n FROM leases WHERE status='active'")["n"] == 3 - expired
    assert db.query_one("SELECT COUNT(*) n FROM leases WHERE status='expired'")["n"] == expired
    assert db.query_one("SELECT COUNT(*) n FROM leases WHERE settled_at IS NOT NULL")["n"] == 0
    assert db.query_one("SELECT COUNT(*) n FROM lease_group_members "
                        "WHERE release_receipt IS NOT NULL")["n"] == 0


def test_actual_admission_serving_and_fence_never_create_or_release_leases(client, admitted):
    async def run():
        reader, transport = reader_for(client, admitted)
        try:
            await reader.refresh()
            assert reader.may_start_ranks() and not reader.may_serve_requests()
            command(client, admitted[1], "serving", evidence_ref="test:ready",
                    evidence_sha256="b" * 64)
            await reader.refresh()
            assert reader.may_serve_requests()
            command(client, admitted[1], "fence", reason="replace test coordinator")
            notices = client.get("/api/v1/events/recent?limit=100").json()
            assert any(reader.on_event(event) for event in notices)
            assert not reader.may_serve_requests()
            observed = await reader.refresh()
            assert all(decision.state == "stop_required" for decision in observed.decisions)
            assert not reader.may_start_ranks()
            assert all(method == "GET" for method, _ in transport.requests)
            assert len(transport.requests) == 10  # One auth, three fresh reads per observation.
            assert_still_charged()
        finally:
            await reader.close()
    asyncio.run(run())


def test_actual_lost_create_reply_recovers_the_same_group_from_inference_journal(
        client, placement, tmp_path):
    plan, manifest = placement
    path = tmp_path / 'inference.sqlite'
    journal = Arbiter(path)
    try:
        intents = GroupIntents(journal)
        grant = journal.submit('native-caller', 'one', 'claude-services-slice/gpu-0',
                               plan.model_id, lifetime='execution')
        call = journal.enqueue(grant['id'], 'native-caller', 'one', {'operation': 'chat'})
        intents.prepare(call['id'], plan, manifest)
        body = intents.create_request(plan.owner_id, plan.request_key)
        assert body == plan.request()
        first = client.post('/api/v1/lease-groups', json=body)
        assert first.status_code == 200
        # The API committed, but the coordinator didn't persist the reply.
        first_id = first.json()['id']
        journal.close()
        journal = Arbiter(path)
        intents = GroupIntents(journal)
        assert intents.snapshot(plan.owner_id, plan.request_key)['group_id'] is None
        replay = client.post('/api/v1/lease-groups',
            json=intents.create_request(plan.owner_id, plan.request_key))
        assert replay.status_code == 200 and replay.json()['id'] == first_id
        intents.record_group(plan.owner_id, plan.request_key, replay.json())
        assert intents.create_request(plan.owner_id, plan.request_key) is None
        assert journal.execution(call['id'], 'native-caller')['state'] == 'queued'
        assert_still_charged()  # Three member leases, not a duplicate set.
    finally:
        journal.close()


def test_actual_peer_hold_expiry_revokes_the_whole_group_without_settlement(client, admitted):
    async def run():
        reader, _ = reader_for(client, admitted)
        try:
            await reader.refresh()
            assert reader.may_start_ranks()
            peer = admitted[1]["members"][0]["lease_id"]
            db.execute("UPDATE leases SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
                       (peer,))
            await reader.refresh()
            assert not reader.may_start_ranks() and not reader.may_serve_requests()
            # A resource read reaps expired rows, but HOLD still charges the
            # member until exact cleanup and settlement are acknowledged.
            resource = client.get("/api/v1/resources/" +
                admitted[1]["members"][0]["resource_id"]).json()
            assert resource["available_memory_mb"] == 9000  # The 1,000 MiB stays charged.
            assert [row["lease_id"] for row in resource["unsettled"]] == [peer]
            assert_still_charged(expired=1)
        finally:
            await reader.close()
    asyncio.run(run())


def test_actual_fence_between_member_reads_cannot_publish_mixed_authority(client, admitted):
    seen = 0
    def fence_after_first(request):
        nonlocal seen
        if request.url.path.endswith("/authority"):
            seen += 1
            if seen == 2:
                command(client, admitted[1], "fence", reason="race a provider read")

    async def run():
        reader, _ = reader_for(client, admitted, before=fence_after_first)
        try:
            report = await reader.refresh()
            assert report.decisions[0].state == "authorized"
            assert report.decisions[1].state == "stop_required"
            assert not reader.may_start_ranks() and not reader.may_serve_requests()
            assert_still_charged()
        finally:
            await reader.close()
    asyncio.run(run())


def test_actual_rotated_host_credential_cannot_take_over_the_coordinator(client, admitted):
    response = client.post("/api/v1/users", json={
        "username": "host-reader", "password": "test-pw", "role": "user",
        "scopes": ["device:claude-services-slice"],
    })
    assert response.status_code == 200
    login = client.post("/api/v1/auth/login",
        json={"username": "host-reader", "password": "test-pw"})
    assert login.status_code == 200
    header = client._headers["Authorization"]

    async def run():
        nonlocal header
        reader, transport = reader_for(client, admitted, header=lambda: header)
        try:
            await reader.refresh()
            assert reader.may_start_ranks()
            before = len(transport.requests)
            header = "Bearer " + login.json()["token"]
            with pytest.raises(AuthorityReadUnavailable):
                await reader.refresh()
            assert transport.requests[before:] == [("GET", "/api/v1/auth/verify")]
            assert reader.current is None and not reader.may_start_ranks()
            assert_still_charged()
        finally:
            await reader.close()
    asyncio.run(run())


def acquisition_for(client, plan, manifest, journal, *, transport=None, header=None):
    credentials = header or (lambda: client._headers['Authorization'])
    plane = CoordinatorPlane(plan.owner_id, credentials,
        transport=transport or httpx.ASGITransport(app=client._client.app))
    def factory(selected, launches):
        return GroupAuthorityReader(selected, launches, credentials,
            transport=httpx.ASGITransport(app=client._client.app))
    return GroupAcquisition(GroupIntents(journal), plan, manifest, plane, factory)


def enqueue(journal, plan):
    grant = journal.submit('native-caller', 'one', 'claude-services-slice/gpu-0',
        plan.model_id, lifetime='execution')
    return grant, journal.enqueue(grant['id'], 'native-caller', 'one', {'operation': 'chat'})


def deliver_events(client, actor):
    notices = client.get('/api/v1/events/recent?limit=100').json()
    return sum(actor.on_event(event) for event in notices)


def test_actual_acquisition_waits_on_obelisk_then_admits_and_fences_without_dispatch(
        client, placement, tmp_path):
    plan, manifest = placement
    response = client.post('/api/v1/resources/' + plan.members[0].resource_id + '/leases',
        json={'exclusive': True, 'wait': True, 'ttl_seconds': 600, 'max_total_s': 7200})
    assert response.status_code == 200
    blocker = response.json()['lease_id']

    async def run():
        journal = Arbiter(tmp_path / 'inference.sqlite')
        actor = acquisition_for(client, plan, manifest, journal)
        try:
            _, call = enqueue(journal, plan)
            actor.note_demand(call['id'])
            actor.events_reconciled()
            report = await actor.wait()
            assert report.phase == 'waiting_members'
            group = client.get('/api/v1/lease-groups/' + report.group_id).json()
            assert [member['status'] for member in group['members']] == [
                'queued', 'not_requested', 'not_requested']
            assert actor.authority is None
            assert db.query_one('SELECT COUNT(*) n FROM leases')['n'] == 2
            assert client.delete('/api/v1/leases/' + blocker).status_code == 200
            assert deliver_events(client, actor)
            report = await actor.wait()
            assert report.phase == 'admitted' and actor.authority.may_start_ranks()
            assert not actor.authority.may_serve_requests()
            command(client, {'id': report.group_id}, 'serving',
                evidence_ref='test:ready', evidence_sha256='b' * 64)
            assert deliver_events(client, actor)
            assert not actor.authority.may_start_ranks()  # Withdrawn before async I/O.
            assert (await actor.wait()).phase == 'lease_serving'
            command(client, {'id': report.group_id}, 'fence', reason='replace test coordinator')
            assert deliver_events(client, actor)
            assert not actor.authority.may_serve_requests()
            assert (await actor.wait()).phase == 'fenced'
            assert db.query_one('SELECT COUNT(*) n FROM leases')['n'] == 4
            assert db.query_one("SELECT COUNT(*) n FROM leases WHERE status='active'")['n'] == 3
            assert db.query_one('SELECT COUNT(*) n FROM lease_group_members '
                'WHERE release_receipt IS NOT NULL')['n'] == 0
            assert journal.execution(call['id'], 'native-caller')['state'] == 'queued'
            assert journal.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0] == 1
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_actual_owned_create_recovers_after_reply_loss_and_caller_cancellation(
        client, placement, tmp_path):
    plan, manifest = placement
    class DropFirstReply(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.ASGITransport(app=client._client.app)
            self.create_calls = 0
        async def handle_async_request(self, request):
            response = await self.inner.handle_async_request(request)
            if request.method == 'POST' and request.url.path == '/api/v1/lease-groups':
                self.create_calls += 1
                if self.create_calls == 1:
                    await response.aclose()
                    raise httpx.ReadError('simulated lost committed create reply')
            return response
        async def aclose(self):
            await self.inner.aclose()

    async def run():
        path = tmp_path / 'inference.sqlite'
        journal = Arbiter(path)
        actor = acquisition_for(client, plan, manifest, journal, transport=DropFirstReply())
        try:
            grant, call = enqueue(journal, plan)
            actor.note_demand(call['id'])
            actor.events_reconciled()
            assert (await actor.wait()).phase == 'unknown'
            group_id = db.query_one('SELECT id FROM lease_groups')['id']
            assert_still_charged()
            assert actor.intents.snapshot(plan.owner_id, plan.request_key)['group_id'] is None
            journal.request_release(grant['id'], 'native-caller')
            await actor.close()
            journal.close()
            journal = Arbiter(path)
            actor = acquisition_for(client, plan, manifest, journal)
            actor.events_reconciled()  # Recover the owned metadata operation, not inference.
            report = await actor.wait()
            assert report.phase == 'admitted' and report.group_id == group_id
            assert db.query_one('SELECT COUNT(*) n FROM lease_groups')['n'] == 1
            assert_still_charged()
            assert journal.execution(call['id'], 'native-caller')['state'] == 'cancelled'
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_actual_acquisition_wrong_coordinator_credential_makes_no_create(client, placement, tmp_path):
    plan, manifest = placement
    response = client.post('/api/v1/users', json={'username': 'other-coordinator',
        'password': 'test-pw', 'role': 'user', 'scopes': ['device:claude-services-slice']})
    assert response.status_code == 200
    login = client.post('/api/v1/auth/login', json={
        'username': 'other-coordinator', 'password': 'test-pw'})
    assert login.status_code == 200
    transport = ObservedRegistry(client)
    async def run():
        journal = Arbiter(tmp_path / 'inference.sqlite')
        actor = acquisition_for(client, plan, manifest, journal, transport=transport,
            header=lambda: 'Bearer ' + login.json()['token'])
        try:
            _, call = enqueue(journal, plan)
            actor.note_demand(call['id'])
            actor.events_reconciled()
            assert (await actor.wait()).phase == 'unknown'
            assert transport.requests == [('GET', '/api/v1/auth/verify')]
            assert actor.intents.snapshot(plan.owner_id, plan.request_key)['create_started'] == 0
            assert db.query_one('SELECT COUNT(*) n FROM lease_groups')['n'] == 0
            assert db.query_one('SELECT COUNT(*) n FROM leases')['n'] == 0
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_existing_supervisor_wakes_acquisition_before_capacity_then_dispatches_once(
        client, placement, tmp_path):
    plan, manifest = placement
    async def run():
        backend_ready = False
        executions = []
        actor = None
        async def injected_backend(grant, request):
            # CPU backend injection only, not an EXO runtime qualification.
            assert backend_ready and actor.authority.may_serve_requests()
            executions.append(grant['active_execution'])
            return {'kind': 'json', 'body': {'answer': 'fixture'}}
        def permitted(_resource):
            return bool(backend_ready and actor and actor.authority and
                        actor.authority.may_serve_requests())
        supervisor = Supervisor(tmp_path / 'inference.sqlite', injected_backend, permitted)
        actor = acquisition_for(client, plan, manifest, supervisor.journal)
        AcquisitionDemand(supervisor, {plan.model_id: actor})
        def changed(report):
            if report.phase == 'lease_serving':
                supervisor.capacity_available('claude-services-slice/gpu-0')
        actor.notify = changed
        try:
            actor.events_reconciled()
            _, call = enqueue(supervisor.journal, plan)
            # Every existing API/compatibility enqueue already calls pump().
            assert supervisor.pump('claude-services-slice/gpu-0') is None
            assert (await actor.wait()).phase == 'admitted'
            assert executions == []  # Admitted leases alone cannot serve.
            saved = actor.intents.snapshot(plan.owner_id, plan.request_key)
            command(client, {'id': saved['group_id']}, 'serving',
                evidence_ref='test:injected-backend-ready', evidence_sha256='c' * 64)
            backend_ready = True  # Explicit test backend readiness, separate from leases.
            assert deliver_events(client, actor)
            assert (await actor.wait()).phase == 'lease_serving'
            await asyncio.gather(*tuple(supervisor.workers.values()))
            assert executions == [call['id']]
            assert supervisor.journal.execution(call['id'], 'native-caller')['state'] == 'complete'
            assert_still_charged()  # Call completion is not physical session release.
            supervisor.pump('claude-services-slice/gpu-0')
            assert executions == [call['id']]
        finally:
            await actor.close()
            await asyncio.gather(*tuple(supervisor.workers.values()))
            supervisor.close()
    asyncio.run(run())
