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
from ollama.exo_groups.renew import GroupRenewals
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


class RenewalAlarm:
    def __init__(self, loop, notify, failed):
        self.notify, self.deadline, self.closed = notify, None, False
    def arm(self, deadline):
        assert not self.closed
        self.deadline = deadline
    def disarm(self):
        self.deadline = None
    def fire(self):
        assert self.deadline is not None and not self.closed
        self.deadline = None
        self.notify()
    def close(self):
        self.closed = True


def acquisition_for(client, plan, manifest, journal, *, transport=None, header=None,
                    renew_before_s=None):
    credentials = header or (lambda: client._headers['Authorization'])
    plane = CoordinatorPlane(plan.owner_id, credentials,
        transport=transport or httpx.ASGITransport(app=client._client.app))
    def factory(selected, launches):
        return GroupAuthorityReader(selected, launches, credentials,
            transport=httpx.ASGITransport(app=client._client.app))
    renewals = (GroupRenewals(plan, plane, renew_before_s=renew_before_s)
                if renew_before_s is not None else None)
    return GroupAcquisition(GroupIntents(journal), plan, manifest, plane, factory,
        renewals=renewals, renewal_alarm_factory=RenewalAlarm, verify_staging=fixtures.staged_fixture)


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


def test_partial_group_renewal_keeps_ordinary_queue_positions_and_exact_member_ids(
        client, placement, tmp_path):
    from datetime import timedelta
    plan, manifest = placement
    response = client.post('/api/v1/resources/' + plan.members[1].resource_id + '/leases',
        json={'exclusive': True, 'wait': True, 'ttl_seconds': 600, 'max_total_s': 7200})
    assert response.status_code == 200
    async def run():
        journal = Arbiter(tmp_path / 'inference.sqlite')
        actor = acquisition_for(client, plan, manifest, journal,
            renew_before_s={m.rank: 120 for m in plan.members})
        try:
            _, call = enqueue(journal, plan)
            actor.note_demand(call['id'])
            actor.events_reconciled()
            report = await actor.wait()
            assert report.phase == 'waiting_members'
            group_id = report.group_id
            before = client.get('/api/v1/lease-groups/' + group_id).json()
            assert [m['status'] for m in before['members']] == ['active', 'queued', 'active']
            held_ids = [before['members'][i]['lease_id'] for i in (0, 2)]
            queued_id = before['members'][1]['lease_id']
            queue_before = client.get('/api/v1/leases/' + queued_id).json()['queue_position']
            old_expiry = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
            for held_id in held_ids:
                db.execute('UPDATE leases SET expires_at=? WHERE id=?', (old_expiry, held_id))
            actor.renewal_alarm.fire()  # Existing event loop wake, no watcher loop.
            assert (await actor.wait()).phase == 'waiting_members'
            after = client.get('/api/v1/lease-groups/' + group_id).json()
            assert [m['lease_id'] for m in after['members']] == [m['lease_id'] for m in before['members']]
            for held_id in held_ids:
                renewed = client.get('/api/v1/leases/' + held_id).json()
                assert datetime.fromisoformat(renewed['expires_at']) > datetime.now(timezone.utc) + timedelta(seconds=500)
            assert client.get('/api/v1/leases/' + queued_id).json()['queue_position'] == queue_before
            assert actor.authority is None  # Partial renewal never permits model dispatch.
            assert actor.renewal_alarm.deadline is not None
            assert db.query_one('SELECT COUNT(*) n FROM lease_groups')['n'] == 1
            assert db.query_one('SELECT COUNT(*) n FROM leases')['n'] == 4
            assert db.query_one('SELECT COUNT(*) n FROM lease_group_members '
                'WHERE release_receipt IS NOT NULL')['n'] == 0
        finally:
            await actor.close()
            assert actor.renewal_alarm.closed
            journal.close()
    asyncio.run(run())


def test_real_fence_racing_ordinary_renewal_does_not_restore_old_group_permission(
        client, placement, tmp_path):
    from datetime import timedelta
    plan, manifest = placement
    class FenceBeforeRenew(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.ASGITransport(app=client._client.app)
            self.fence = False
            self.renew_calls = 0
        async def handle_async_request(self, request):
            if request.method == 'POST' and request.url.path.endswith('/renew'):
                self.renew_calls += 1
                if self.fence:
                    self.fence = False
                    group_id = db.query_one('SELECT id FROM lease_groups')['id']
                    command(client, {'id': group_id}, 'fence', reason='race the old coordinator renewal')
            return await self.inner.handle_async_request(request)
        async def aclose(self):
            await self.inner.aclose()
    async def run():
        journal = Arbiter(tmp_path / 'inference.sqlite')
        transport = FenceBeforeRenew()
        actor = acquisition_for(client, plan, manifest, journal, transport=transport,
            renew_before_s={m.rank: 120 for m in plan.members})
        try:
            _, call = enqueue(journal, plan)
            actor.note_demand(call['id'])
            actor.events_reconciled()
            assert (await actor.wait()).phase == 'admitted'
            assert actor.authority.may_start_ranks()
            group_id = actor.report.group_id
            member = client.get('/api/v1/lease-groups/' + group_id).json()['members'][0]
            old_expiry = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
            db.execute('UPDATE leases SET expires_at=? WHERE id=?', (old_expiry, member['lease_id']))
            transport.fence = True
            actor.renewal_alarm.fire()
            assert (await actor.wait()).phase == 'unknown'
            assert not actor.authority.may_start_ranks() and not actor.authority.may_serve_requests()
            assert transport.renew_calls == 1
            assert actor.intents.snapshot(plan.owner_id, plan.request_key)['generation'] == 1
            assert client.get('/api/v1/lease-groups/' + group_id).json()['generation'] == 2
            # The ordinary lease POST may extend the row despite the fence; it
            # confers no group permission, and nothing is released/reacquired.
            assert client.get('/api/v1/leases/' + member['lease_id']).json()['expires_at'] > old_expiry
            assert_still_charged()
            assert actor.renewal_alarm.deadline is None
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_expired_member_stays_held_without_renewal_or_replacement(client, placement, tmp_path):
    plan, manifest = placement
    transport = ObservedRegistry(client)
    async def run():
        journal = Arbiter(tmp_path / 'inference.sqlite')
        actor = acquisition_for(client, plan, manifest, journal, transport=transport,
            renew_before_s={m.rank: 120 for m in plan.members})
        try:
            _, call = enqueue(journal, plan)
            actor.note_demand(call['id'])
            actor.events_reconciled()
            assert (await actor.wait()).phase == 'admitted'
            group_id = actor.report.group_id
            before = client.get('/api/v1/lease-groups/' + group_id).json()
            member = before['members'][0]
            db.execute("UPDATE leases SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
                (member['lease_id'],))
            actor.renewal_alarm.fire()
            assert (await actor.wait()).phase == 'draining'
            assert not actor.authority.may_start_ranks()
            assert not any(method == 'POST' and path.endswith('/renew') for method, path in transport.requests)
            after = client.get('/api/v1/lease-groups/' + group_id).json()
            assert [m['lease_id'] for m in after['members']] == [m['lease_id'] for m in before['members']]
            assert_still_charged(expired=1)
            assert actor.renewal_alarm.deadline is None
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_explicit_member_ceiling_arms_end_notification_without_renewal(client, placement, tmp_path):
    from datetime import timedelta
    from ollama.exo_groups.clock import authority_clock
    plan, manifest = placement
    transport = ObservedRegistry(client)
    async def run():
        journal = Arbiter(tmp_path / 'inference.sqlite')
        actor = acquisition_for(client, plan, manifest, journal, transport=transport,
            renew_before_s={m.rank: 120 for m in plan.members})
        try:
            _, call = enqueue(journal, plan)
            actor.note_demand(call['id'])
            actor.events_reconciled()
            assert (await actor.wait()).phase == 'admitted'
            group_id = actor.report.group_id
            member = client.get('/api/v1/lease-groups/' + group_id).json()['members'][0]
            ceiling = datetime.now(timezone.utc) + timedelta(seconds=20)
            granted = ceiling - timedelta(seconds=plan.members[0].max_total_s)
            db.execute('UPDATE leases SET granted_at=?,expires_at=? WHERE id=?',
                (granted.isoformat(), ceiling.isoformat(), member['lease_id']))
            actor.renewal_alarm.fire()
            assert (await actor.wait()).phase == 'admitted'
            assert any(m.at_ceiling and m.lease_id == member['lease_id']
                       for m in actor.renewal_report.members)
            assert authority_clock() < actor.renewal_alarm.deadline < authority_clock() + 21
            assert not any(method == 'POST' and path.endswith('/renew') for method, path in transport.requests)
            assert_still_charged()
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_gateway_group_capacity_uses_three_host_members_without_fourth_slice_lease(
        client, placement, tmp_path):
    """Real registry, GatewayQueue/router and manager source; no native backend."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    from fastapi import FastAPI
    from ollama import server
    from ollama.inference_bootstrap import QueueRuntime, QueueSettings

    plan, manifest = placement
    resource = 'claude-services-slice/gpu-0'
    async def run():
        manager = SimpleNamespace(models={}, residency_renewal=None,
            sc=SimpleNamespace(request_lease=Mock(side_effect=AssertionError('fourth lease'))),
            ollama=SimpleNamespace(memory_estimate_mb=Mock(side_effect=AssertionError('local EXO path'))),
            _collect_stats=lambda: {'memory_available_mb': 10000, 'memory_device_inuse_mb': 0})
        service = SimpleNamespace(app=FastAPI(), mgr=manager, ChatRequest=server.ChatRequest,
            config=SimpleNamespace(WHISPER_MAX_UPLOAD_MB=1, SC_RESOURCE_ID=resource,
                                   SC_SERVICE_ID='fixture-service'))
        runtime = QueueRuntime.install(service, QueueSettings(tmp_path / 'gateway.sqlite',
            {plan.model_id: resource}), registry_owner=plan.owner_id)
        runtime.bind_capacity_source(work_permitted=True)
        actor = acquisition_for(client, plan, manifest, runtime.queue.supervisor.journal)
        installation = runtime.bind_exo({plan.model_id: actor})
        @service.app.middleware('http')
        async def identify(request, call_next):
            request.state.caller = {'username': 'caller'}
            return await call_next(request)
        async def drain():
            async with asyncio.timeout(3):
                while tasks := [task for task in runtime.queue.supervisor.preparations if not task.done()]:
                    await asyncio.gather(*tasks)
        try:
            actor.events_reconciled()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app),
                                        base_url='http://fixture') as http:
                grant = await http.post('/v1/inference/grants', json={
                    'key': 'one', 'model': plan.model_id, 'lifetime': 'execution'})
                assert grant.status_code == 202
                call = await http.post('/v1/inference/grants/' + grant.json()['id'] + '/executions',
                    json={'key': 'one', 'request': {'model': plan.model_id,
                        'messages': [{'role': 'user', 'content': 'hello'}]}})
                assert call.status_code == 202
            await drain()
            assert actor.report.phase == 'admitted'
            saved = actor.intents.snapshot(plan.owner_id, plan.request_key)
            command(client, {'id': saved['group_id']}, 'serving',
                evidence_ref='test:injected-ready', evidence_sha256='c' * 64)
            assert deliver_events(client, actor)
            await drain()
            assert actor.report.phase == 'lease_serving'
            # Serving authority is necessary, never enough without native state.
            assert runtime.queue.supervisor.journal.execution(call.json()['id'], 'caller')['state'] == 'queued'
            assert runtime.capacity_source.permitted_model(resource, plan.model_id) is None
            assert runtime.queue.supervisor.pump(resource) is None
            rows = db.query('SELECT * FROM leases')
            assert len(rows) == 3
            assert {row['resource_id'] for row in rows} == {m.resource_id for m in plan.members}
            assert all(row['exclusive'] == 1 and row['status'] == 'active' for row in rows)
            manager.sc.request_lease.assert_not_called()
            manager.ollama.memory_estimate_mb.assert_not_called()
            assert manager.models == {}
            assert manager.initial_reservations.snapshot(plan.model_id) is None
            assert not (await runtime.quiesce())['drained']
        finally:
            runtime.capacity_source.pause()
            await drain()
            for provider in installation.providers.values():
                provider._closed = True
            await actor.close()  # Fixture cleanup only, not production settlement.
            await drain()
            runtime.residency_actor.close()
            runtime.queue.supervisor.close()
    asyncio.run(run())


def test_unverified_staging_cannot_request_any_real_registry_member(client, placement, tmp_path):
    plan, manifest = placement
    async def run():
        journal = Arbiter(tmp_path / 'staging-gate.sqlite')
        actor = acquisition_for(client, plan, manifest, journal)
        actor.verify_staging = None
        try:
            _, call = enqueue(journal, plan)
            actor.note_demand(call['id'])
            actor.events_reconciled()
            report = await actor.wait()
            assert report.phase == 'unknown' and report.reason == 'staging_unconfirmed'
            assert db.query_one('SELECT COUNT(*) n FROM leases')['n'] == 0
            assert db.query_one('SELECT COUNT(*) n FROM lease_groups')['n'] == 0
            assert not actor.intents.snapshot(plan.owner_id, plan.request_key)['create_started']
            actor.verify_staging = fixtures.staged_fixture  # Explicit injected CPU control.
            actor.wake()
            assert (await actor.wait()).phase == 'admitted'
            assert_still_charged()
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_partial_group_does_not_advance_after_staged_files_disappear(client, placement, tmp_path):
    plan, manifest = placement
    response = client.post('/api/v1/resources/' + plan.members[0].resource_id + '/leases',
        json={'exclusive': True, 'wait': True, 'ttl_seconds': 600, 'max_total_s': 7200})
    assert response.status_code == 200
    blocker = response.json()['lease_id']
    async def run():
        journal = Arbiter(tmp_path / 'partial-staging.sqlite')
        actor = acquisition_for(client, plan, manifest, journal)
        ready = True
        async def staging(selected, raw):
            if not ready:
                raise FileNotFoundError('assigned rank disappeared')
            return await fixtures.staged_fixture(selected, raw)
        actor.verify_staging = staging
        try:
            _, call = enqueue(journal, plan)
            actor.note_demand(call['id'])
            actor.events_reconciled()
            assert (await actor.wait()).phase == 'waiting_members'
            assert db.query_one('SELECT COUNT(*) n FROM leases')['n'] == 2
            ready = False
            assert client.delete('/api/v1/leases/' + blocker).status_code == 200
            assert deliver_events(client, actor)
            report = await actor.wait()
            assert report.phase == 'unknown' and report.reason == 'staging_unconfirmed'
            assert db.query_one('SELECT COUNT(*) n FROM leases')['n'] == 2
            assert db.query_one('SELECT COUNT(*) n FROM leases WHERE resource_id!=?',
                                (plan.members[0].resource_id,))['n'] == 0
            assert not actor.may_start_ranks()
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def host_runtime_for(client, admitted, tmp_path, *, rank='slice', credential_rank='slice',
                     runtime_sha256='b' * 64, boot_id=None):
    """Actual scoped HTTP authority; host process/runtime effects are CPU fakes."""
    import os
    import sys
    from uuid import UUID
    from ollama.exo_groups.host_authority import HostAuthorityAdapter, HostPlaneReader
    from ollama.exo_groups.member import ExpectedMember
    from ollama.exo_groups.rank_lifecycle import RankLifecycle, RankSpec
    from ollama.exo_groups.rank_process import Listener, RunBinding
    from ollama.exo_groups.rank_runtime import HostRankRuntime, RuntimeObservation, launch_sha256, record_host_attempt
    from ollama.test_exo_group_host_authority import FakeAlarm
    from ollama.test_exo_group_rank_lifecycle import FakeProcess
    plan, group, launches = admitted
    response = client.post('/api/v1/users', json={
        'username': 'host-rank-runtime-' + rank, 'password': 'test-pw', 'role': 'user',
        'scopes': ['device:' + next(m.resource_id.split('/')[0] for m in plan.members
                                   if m.rank == credential_rank)],
    })
    assert response.status_code == 200
    login = client.post('/api/v1/auth/login', json={
        'username': 'host-rank-runtime-' + rank, 'password': 'test-pw'})
    assert login.status_code == 200
    header = {'Authorization': 'Bearer ' + login.json()['token']}
    principal = client._client.get('/api/v1/auth/verify', headers=header).json()['owner_id']
    assert principal != plan.owner_id
    expected = ExpectedMember.from_launch(plan, next(value for value in launches if value.rank == rank))
    binding = RunBinding.from_expected(expected, runtime_sha256)
    child = FakeProcess(binding)
    boot = boot_id or str(UUID(int=123))
    actor = RankLifecycle(expected, RankSpec(binding, (sys.executable, '-c', 'pass'),
        tmp_path, (Listener('127.0.0.1', 39999),), 0), tmp_path / 'host-run',
        memory_reader=lambda: 100, port_reader=lambda _: True, boot_reader=lambda: boot,
        monotonic=lambda: 100.1, spawner=lambda *a, **kw: child, emit=lambda _: None)
    reader = HostPlaneReader(lambda: header['Authorization'], owner_id=principal,
                            transport=httpx.ASGITransport(app=client._client.app))
    adapter = HostAuthorityAdapter(actor, reader, alarm_factory=FakeAlarm)
    environment = {}
    async def verify_runtime():
        return RuntimeObservation(runtime_sha256, launch_sha256(actor, environment), boot, os.getuid())
    class Watch:
        def __init__(self, process, notify, failed):
            self.notify = notify
        def close(self):
            pass
    runtime = HostRankRuntime(actor, adapter, verify_runtime=verify_runtime,
        environment=environment, boot_id=boot, watcher_factory=Watch)
    runtime.test_journal = Arbiter(tmp_path / 'host-inference.sqlite')
    record_host_attempt(runtime, runtime.test_journal)
    return runtime, child, header


def test_actual_host_runtime_uses_scoped_authority_then_fence_stops_without_settlement(
        client, admitted, tmp_path):
    async def run():
        runtime, child, header = host_runtime_for(client, admitted, tmp_path)
        try:
            assert client._client.get('/api/v1/lease-groups/' + admitted[1]['id'],
                                      headers=header).status_code == 404
            await runtime.on_connect()
            await runtime.start()
            assert runtime.actor.phase == 'running' and not await runtime.may_admit()
            command(client, admitted[1], 'serving', evidence_ref='test:all-ready',
                    evidence_sha256='b' * 64)
            assert await runtime.may_admit()
            command(client, admitted[1], 'fence', reason='host controller test fence')
            notices = client._client.get('/api/v1/events/recent?limit=100', headers=header)
            assert notices.status_code == 200
            for notice in notices.json():
                await runtime.on_event(notice)
            assert child.stop_count == 1 and runtime.actor.draining
            assert_still_charged()
            child.code = 0
            runtime.exit_watch.notify()
            assert runtime.actor.receipt['lease_released'] is False
            assert_still_charged()
        finally:
            if runtime.actor.process is not None:
                child.code = 0
                runtime.actor.process_exited()
            await runtime.close_after_cleanup()
            runtime.test_journal.close()
    asyncio.run(run())


def test_actual_host_runtime_cannot_use_slice_credential_for_wafer(client, admitted, tmp_path):
    from ollama.exo_groups.rank_process import ObservationUnknown
    async def run():
        runtime, child, _ = host_runtime_for(client, admitted, tmp_path, rank='wafer')
        try:
            with pytest.raises(ObservationUnknown):
                await runtime.on_connect()
                await runtime.start()
            assert runtime.actor.process is None and child.stop_count == 0
            assert_still_charged()
        finally:
            await runtime.attest_never_started()
            await runtime.close_after_cleanup()
            runtime.test_journal.close()
    asyncio.run(run())


def test_actual_host_runtime_stops_on_peer_hold_expiry_and_retains_every_claim(
        client, admitted, tmp_path):
    async def run():
        runtime, child, _ = host_runtime_for(client, admitted, tmp_path)
        try:
            await runtime.on_connect()
            await runtime.start()
            peer = next(value for value in admitted[2] if value.rank == 'obelisk')
            db.execute("UPDATE leases SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
                       (peer.lease_id,))
            await runtime.on_heartbeat()
            assert child.stop_count == 1 and runtime.actor.receipt is None
            resource = client.get('/api/v1/resources/' + peer.resource_id).json()
            assert resource['available_memory_mb'] == 9000
            assert [value['lease_id'] for value in resource['unsettled']] == [peer.lease_id]
            assert_still_charged(expired=1)
        finally:
            child.code = 0
            runtime.actor.process_exited()
            await runtime.close_after_cleanup()
            runtime.test_journal.close()
    asyncio.run(run())


def cleanup_controller(client, plan, manifest, group, journal):
    """Actual registry/SQLite controller with injected physical cleanup artifacts."""
    from ollama.exo_groups.cleanup import CleanupExpectation
    from ollama.exo_groups.completion import GroupCompletion
    from ollama.exo_groups.member import ExpectedMember
    from ollama.exo_groups.rank_process import (
        RunBinding, ProcessIdentity, Listener, CleanupObservation, cleanup_evidence,
    )

    class Artifacts(ObservedRegistry):
        def __init__(self):
            super().__init__(client)
            self.files, self.lose = {}, False
        async def handle_async_request(self, request):
            if request.url.path.startswith('/api/v1/files/'):
                file_id = int(request.url.path.split('/')[4])
                raw = self.files[file_id]
                if request.url.path.endswith('/raw'):
                    return httpx.Response(200, content=raw)
                import hashlib
                return httpx.Response(200, json={'id':file_id,'bytes':len(raw),
                                               'sha256':hashlib.sha256(raw).hexdigest()})
            response = await super().handle_async_request(request)
            if self.lose and '/members/' in request.url.path and request.method == 'POST':
                self.lose = False
                await response.aread()
                raise httpx.ReadError('lost exact member reply')
            return response

    intents = GroupIntents(journal)
    grant = journal.submit('native', 'one', 'claude-services-slice/gpu-0', plan.model_id)
    call = journal.enqueue(grant['id'], 'native', 'one', {'operation':'chat'})
    intents.prepare(call['id'], plan, manifest)
    staged = fixtures.staging_fixture(plan)
    intents.record_staging(plan.owner_id, plan.request_key, staged)
    intents.begin_create(plan.owner_id, plan.request_key)
    intents.record_group(plan.owner_id, plan.request_key, group)
    transport = Artifacts()
    plane = CoordinatorPlane(plan.owner_id, lambda:client._headers['Authorization'], transport=transport)
    actor = GroupAcquisition(intents, plan, manifest, plane,
        lambda *_:None, verify_staging=fixtures.staged_fixture)
    completion = GroupCompletion(actor)
    proofs = {}
    if group['phase'] == 'admitted':
        leases = {m['lease_id']:client.get('/api/v1/leases/'+m['lease_id']).json() for m in group['members']}
        launches = bind_admitted_group(plan, group, leases, expected_generation=1,
                                      now=datetime.now(timezone.utc))
        intents.bind_launches(plan.owner_id, plan.request_key, launches)
        for i, launch in enumerate(launches):
            run = RunBinding.from_expected(ExpectedMember.from_launch(plan, launch), staged.runtime_sha256)
            pin = CleanupExpectation(run, ProcessIdentity(100+i,99,100+i,501,1000+i,1),
                                     (Listener('127.0.0.1',39999+i),),100,0)
            completion.pin_launch(pin)
            raw, digest = cleanup_evidence(CleanupObservation(run,pin.child,0,(),(True,),100,100,0,
                                                             datetime.now(timezone.utc).isoformat()))
            transport.files[i+1] = raw
            proofs[launch.lease_id] = {'file_id':i+1,'sha256':digest}
    journal.request_release(grant['id'],'native')
    return actor, completion, transport, proofs


def test_actual_cleanup_release_and_all_three_exact_settlements(client, admitted, placement, tmp_path):
    async def run():
        plan, group, _ = admitted
        journal = Arbiter(tmp_path/'settlement.sqlite')
        actor, completion, transport, proofs = cleanup_controller(client,plan,placement[1],group,journal)
        try:
            assert (await completion.begin_release(generation=1))['state'] == 'host_cleanup_required'
            assert_still_charged()
            for i, (lease_id, proof) in enumerate(proofs.items()):
                result = await completion.settle_member(lease_id,generation=1,**proof)
                assert result['state'] == 'exact_member_settlement_confirmed'
                assert db.query_one("SELECT COUNT(*) n FROM leases WHERE status='active'")['n'] == 2-i
                assert db.query_one('SELECT COUNT(*) n FROM lease_group_members '
                                    'WHERE release_receipt IS NOT NULL')['n'] == i+1
            assert (await completion.begin_release(generation=1))['state'] == 'group_settlement_confirmed'
            actor.events_reconciled()
            assert (await actor.wait()).group_phase == 'released'
            assert not actor.may_start_ranks()
            writes=[p for m,p in transport.requests if m=='POST']
            assert len(writes)==4  # One group drain plus three exact-member settlements.
            assert all('/release' in p for p in writes)
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_actual_hold_debt_lost_reply_is_settled_once_without_inference_replay(
        client, admitted, placement, tmp_path):
    async def run():
        plan, group, _ = admitted
        journal = Arbiter(tmp_path/'hold-settlement.sqlite')
        actor, completion, transport, proofs = cleanup_controller(client,plan,placement[1],group,journal)
        lease_id=next(iter(proofs))
        try:
            db.execute("UPDATE leases SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?", (lease_id,))
            resource=client.get('/api/v1/resources/'+next(m['resource_id'] for m in group['members']
                                                      if m['lease_id']==lease_id)).json()
            assert resource['available_memory_mb']==9000
            assert resource['unsettled'][0]['lease_id']==lease_id
            transport.lose=True
            with pytest.raises(Exception):
                await completion.settle_member(lease_id,generation=1,**proofs[lease_id])
            receipt=client.get('/api/v1/leases/'+lease_id).json()
            assert receipt['settled_at'] is not None
            assert (await completion.settle_member(lease_id,generation=1,**proofs[lease_id]))['state'] \
                == 'exact_member_settlement_confirmed'
            assert sum(m=='POST' and '/members/' in p for m,p in transport.requests)==1
            assert journal.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0]==1
            assert journal.db.execute("SELECT COUNT(*) FROM executions WHERE state='cancelled'").fetchone()[0]==1
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_actual_pending_group_cancelled_without_member_or_cleanup_fiction(
        client, admitted, placement, tmp_path):
    from dataclasses import replace
    async def run():
        plan=replace(admitted[0],request_key='second-pending-group')
        reply=client.post('/api/v1/lease-groups',json=plan.request())
        assert reply.status_code==200
        group=reply.json()
        assert group['phase']=='pending'
        journal=Arbiter(tmp_path/'pending-close.sqlite')
        actor,completion,transport,_=cleanup_controller(client,plan,placement[1],group,journal)
        try:
            assert (await completion.begin_release(generation=1))['phase']=='cancelled'
            actor.events_reconciled()
            report=await actor.wait()
            assert report.phase=='draining' and report.group_phase=='cancelled'
            assert not actor.ownership_uncertain
            assert all(m['lease_id'] is None for m in actor.group_snapshot['members'])
            assert_still_charged()  # Only the OTHER group's original three claims exist.
            assert journal.db.execute('SELECT COUNT(*) FROM exo_group_cleanup_pins').fetchone()[0]==0
            assert sum(m=='POST' for m,p in transport.requests)==1
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


def test_actual_queued_only_partial_group_cancels_without_cleanup_fiction(client, placement, tmp_path):
    plan, manifest = placement
    response = client.post('/api/v1/resources/' + plan.members[0].resource_id + '/leases',
        json={'exclusive':True,'wait':True,'ttl_seconds':600,'max_total_s':7200})
    assert response.status_code == 200
    blocker = response.json()['lease_id']
    group = client.post('/api/v1/lease-groups', json=plan.request()).json()
    assert group['phase'] == 'acquiring'
    queued = [m['lease_id'] for m in group['members'] if m['lease_id']]
    assert len(queued) == 1
    async def run():
        journal = Arbiter(tmp_path/'queued-only-close.sqlite')
        actor, completion, transport, _ = cleanup_controller(client,plan,manifest,group,journal)
        try:
            result = await completion.begin_release(generation=1)
            assert result['state'] == 'group_settlement_confirmed' and result['phase'] == 'cancelled'
            assert client.get('/api/v1/leases/'+queued[0]).json()['status'] == 'cancelled'
            assert client.get('/api/v1/leases/'+blocker).json()['status'] == 'active'
            assert journal.db.execute('SELECT COUNT(*) FROM exo_group_cleanup_pins').fetchone()[0] == 0
            assert not any('/members/' in p and m=='POST' for m,p in transport.requests)
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


@pytest.mark.parametrize('expire', [False, True])
def test_actual_partly_owned_group_uses_host_no_start_fence_then_exact_settlement(
        client, placement, tmp_path, expire):
    import hashlib
    from pathlib import Path
    from ollama.exo_groups.plan import LaunchBinding
    from ollama.exo_groups.rank_process import ObservationUnknown
    plan,manifest=placement
    slice_member=next(m for m in plan.members if m.rank=='slice')
    blocker=client.post('/api/v1/resources/'+slice_member.resource_id+'/leases',
        json={'exclusive':True,'wait':True,'ttl_seconds':600,'max_total_s':7200}).json()['lease_id']
    group=client.post('/api/v1/lease-groups',json=plan.request()).json()
    assert group['phase']=='acquiring'
    # Once the first host is admitted, core requests both later members through
    # their queues: Slice queues, while Wafer is admitted too. Neither may run.
    active=[m for m in group['members'] if m['status']=='active']
    assert len(active)==2

    async def run():
        journal=Arbiter(tmp_path/'coordinator.sqlite')
        actor,completion,transport,_=cleanup_controller(client,plan,manifest,group,journal)
        runtimes=[]
        try:
            for held in active:
                member=next(m for m in plan.members if m.resource_id==held['resource_id'])
                lease=client.get('/api/v1/leases/'+held['lease_id']).json()
                launch=LaunchBinding(group['id'],1,plan.owner_id,member.rank,member.pipeline_rank,
                    member.resource_id,lease['id'],member.layers,lease['expires_at'],plan.rank_manifest_sha256)
                directory=tmp_path/member.rank
                directory.mkdir()
                runtime,child,_=host_runtime_for(client,(plan,group,(launch,)),directory,
                    rank=member.rank,credential_rank=member.rank,
                    runtime_sha256=fixtures.staging_fixture(plan).runtime_sha256,
                    boot_id=next(boot for name,_resource,boot,_uid in fixtures.staging_fixture(plan).boots
                                 if name==member.rank))
                runtimes.append((runtime,child,lease,member))
                await runtime.on_connect()
                assert runtime.actor.process is None and not runtime.actor.draining
                await completion.pin_prepared(runtime.prepared,generation=1)
            result=await completion.begin_release(generation=1)
            assert result['state']=='host_cleanup_required'
            if expire:
                runtime,child,lease,member=runtimes[0]
                db.execute("UPDATE leases SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",(lease['id'],))
                resource=client.get('/api/v1/resources/'+member.resource_id).json()
                assert resource['available_memory_mb']==9000
                assert resource['unsettled'][0]['lease_id']==lease['id']
            for index,(runtime,child,lease,member) in enumerate(runtimes):
                notices=client.get('/api/v1/events/recent?limit=100').json()
                for notice in notices:
                    await runtime.on_event(notice)
                receipt=await runtime.attest_never_started()
                raw=Path(receipt['evidence_path']).read_bytes()
                transport.files[index+1]=raw
                result=await completion.settle_member(lease['id'],generation=1,
                    file_id=index+1,sha256=hashlib.sha256(raw).hexdigest())
                assert result['state']=='exact_member_settlement_confirmed'
                assert client.get('/api/v1/resources/'+member.resource_id).json()['available_memory_mb']==10000
                assert child.stop_count==0 and runtime.actor.process is None
                assert not runtime.actor.launch_attempted
                with pytest.raises(ObservationUnknown):
                    await runtime.start()
                assert runtime.test_journal.db.execute('SELECT phase FROM exo_host_rank_attempts').fetchone()[0]=='launch_fenced'
            assert (await completion.begin_release(generation=1))['state']=='group_settlement_confirmed'
            assert client.get('/api/v1/leases/'+blocker).json()['status']=='active'
            assert sum(m=='POST' and '/members/' in p for m,p in transport.requests)==2
            assert db.query_one('SELECT COUNT(*) n FROM leases')['n']==4
        finally:
            for runtime,child,lease,member in runtimes:
                if runtime.actor.receipt is None:
                    await runtime.attest_never_started()
                await runtime.close_after_cleanup()
                runtime.test_journal.close()
            await actor.close()
            journal.close()
    asyncio.run(run())


@pytest.mark.parametrize('lose_upload_reply,expire_member', [(False,False),(True,False),(True,True)])
def test_actual_cleanup_upload_then_exact_member_settlement(
        client, admitted, placement, tmp_path, lose_upload_reply, expire_member):
    """Real file upload/list/readback and group receipt, no physical host effects."""
    from ollama.exo_groups.publication import GroupPublication
    from ollama.exo_groups.completion import GroupCompletion
    from ollama.exo_groups.plane import PlaneUnavailable

    class UploadRegistry(ObservedRegistry):
        async def handle_async_request(self, request):
            response = await super().handle_async_request(request)
            if self.lose and request.method == 'POST' and request.url.path == '/api/v1/files':
                self.lose = False
                await response.aread()
                await response.aclose()
                raise httpx.ReadError('lost file reply after actual commit')
            return response

    async def run():
        plan, group, _ = admitted
        path = tmp_path/'publication.sqlite'
        journal = Arbiter(path)
        actor, completion, injected, proofs = cleanup_controller(client,plan,placement[1],group,journal)
        lease = next(iter(proofs))
        raw = injected.files[proofs[lease]['file_id']]
        if expire_member:
            db.execute("UPDATE leases SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",(lease,))
            client.get('/api/v1/resources/'+plan.members[0].resource_id)
        await actor.plane.close()
        transport = UploadRegistry(client)
        transport.lose = lose_upload_reply
        actor.plane = CoordinatorPlane(plan.owner_id,lambda:client._headers['Authorization'],transport=transport)
        completion = GroupCompletion(actor)
        scope = client.get('/api/v1/auth/verify').json()['default_group']
        publication = GroupPublication(completion,group_scope=scope)
        try:
            if lose_upload_reply:
                with pytest.raises(PlaneUnavailable):
                    await publication.complete_member(lease,raw,generation=1)
                assert db.query_one('SELECT COUNT(*) n FROM files')['n'] == 1
                assert_still_charged(expired=int(expire_member))
                # Process restart retains upload_started and rediscovers the
                # same actual file, not a second upload or replacement proof.
                await actor.wait()
                journal.close()
                journal = Arbiter(path)
                actor.intents = GroupIntents(journal)
                completion = GroupCompletion(actor)
                publication = GroupPublication(completion,group_scope=scope)
            result = await publication.complete_member(lease,raw,generation=1)
            assert result['state'] == 'exact_member_settlement_confirmed'
            saved = publication._saved(lease)
            assert saved['confirmed'] == 1
            assert saved['raw'] == raw
            assert db.query_one('SELECT COUNT(*) n FROM files')['n'] == 1
            assert db.query_one('SELECT COUNT(*) n FROM leases')['n'] == 3
            terminal = client.get('/api/v1/leases/'+lease).json()
            assert terminal['settled_at'] is not None if expire_member else terminal['status']=='released'
            assert db.query_one("SELECT COUNT(*) n FROM leases WHERE status='active'")['n'] == 2
            assert client.get('/api/v1/files/'+str(saved['file_id'])+'/raw').content == raw
            assert result['cleanup_evidence']['evidence_ref'].endswith('/'+str(saved['file_id'])+'/raw')
            posts = [p for m,p in transport.requests if m=='POST']
            assert posts.count('/api/v1/files') == 1
            assert len([p for p in posts if '/members/' in p and p.endswith('/release')]) == 1
            assert not any(p.endswith('/advance') or p.endswith('/renew') for p in posts)
        finally:
            await actor.close()
            journal.close()
    asyncio.run(run())


@pytest.mark.parametrize('lost_reply,fence_first', [(False,False),(True,False),(False,True)])
def test_actual_journal_readiness_attestation_transport_is_exact_and_fenced(
        client, admitted, tmp_path, lost_reply, fence_first):
    """Real core attestation contract; host/runtime checks have separate controls."""
    from ollama.exo_groups.prepare import canonical, digest
    from ollama.exo_groups.plane import PlaneUnavailable
    plan, group, _ = admitted
    evidence = {'group_id':group['id'],'generation':1,'kind':'injected-test-readiness'}
    raw = canonical(evidence)
    proof = {'evidence_ref':'model-service-journal:exo_group_readiness:'+digest(evidence),
             'evidence_sha256':digest(evidence)}
    (tmp_path/'injected-readiness.json').write_bytes(raw)
    class ServingRegistry(ObservedRegistry):
        async def handle_async_request(self, request):
            response = await super().handle_async_request(request)
            if self.lose and request.url.path.endswith('/serving') and request.method=='POST':
                self.lose=False
                await response.aread()
                await response.aclose()
                raise httpx.ReadError('lost serving metadata reply after commit')
            return response
    if fence_first:
        command(client,group,'fence',reason='test stale serving owner')
    async def run():
        transport=ServingRegistry(client)
        transport.lose=lost_reply
        plane=CoordinatorPlane(plan.owner_id,lambda:client._headers['Authorization'],transport=transport)
        try:
            if fence_first or lost_reply:
                with pytest.raises(PlaneUnavailable):
                    await plane.mark_serving(group['id'],1,proof)
            else:
                assert (await plane.mark_serving(group['id'],1,proof))['phase']=='serving'
            after=await plane.group(group['id'])
            if fence_first:
                assert after['generation']==2 and after['serving_evidence'] is None
            else:
                assert after['phase']=='serving'
                assert after['serving_evidence']=={'generation':1,'attested_by':plan.owner_id,**proof}
                # Exact same metadata is idempotent; another proof is refused.
                assert (await plane.mark_serving(group['id'],1,proof))['serving_evidence']==after['serving_evidence']
                changed={'evidence_ref':'model-service-journal:exo_group_readiness:'+'f'*64,
                         'evidence_sha256':'f'*64}
                with pytest.raises(PlaneUnavailable):
                    await plane.mark_serving(group['id'],1,changed)
            assert_still_charged()
        finally:
            await plane.close()
    asyncio.run(run())
