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
from ollama.exo_groups.authority import AuthorityReadUnavailable, GroupAuthorityReader
from ollama.exo_groups.plan import GroupPlan, bind_admitted_group

pytest_plugins = ["test_registry"]


@pytest.fixture
def admitted(client):
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
