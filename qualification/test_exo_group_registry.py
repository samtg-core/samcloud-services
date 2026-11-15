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
        renewals=renewals, renewal_alarm_factory=RenewalAlarm)


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
