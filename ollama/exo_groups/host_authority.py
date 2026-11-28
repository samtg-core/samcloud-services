"""Read-only HTTPS/SSE adapter for one exact host member.

Use the host service's existing credential provider and event loop. No token is
copied to disk, no coordinator credential is accepted, and no lease/group write
or automatic reconnect/launch is implemented. Event hints never grant authority.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from typing import Callable, Mapping
from urllib.parse import quote

import httpx

from .alarm import ContinuousAuthorityAlarm
from .member import check_projection, event_requires_read, settlement_confirmed
from .rank_lifecycle import RankLifecycle
from .rank_process import ObservationUnknown
from .http import bounded_json


PLANE = "https://cloud.samtg.xyz"


class ReadUnavailable(RuntimeError):
    pass


class HostPlaneReader:
    """Credential callback stays local; redirects and default timeouts disabled."""

    def __init__(self, header: Callable[[], str], *, owner_id: str, transport=None):
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("pinned existing host principal required")
        self.owner_id = owner_id
        self._verified_digest = None
        self._header = header
        self._client = httpx.AsyncClient(
            base_url=PLANE,
            timeout=None,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    def _headers(self, accept: str) -> dict[str, str]:
        value = self._header()
        if (
            not isinstance(value, str)
            or not value.startswith("Bearer ")
            or len(value) <= 7
            or any(ord(c) < 32 or ord(c) > 126 for c in value)
        ):
            raise ReadUnavailable("host credential unavailable")
        return {"Authorization": value, "Accept": accept, "Cache-Control": "no-cache"}

    async def _json(self, path: str, headers: Mapping) -> Mapping:
        # Same bounded, duplicate/nonfinite-rejecting metadata parser used by
        # the coordinator. Host and coordinator credentials remain separate.
        return await bounded_json(self._client, 'GET', path, headers)

    async def _verify(self, headers: Mapping) -> None:
        digest = hashlib.sha256(headers["Authorization"].encode()).digest()
        if digest != self._verified_digest:
            value = await self._json("/api/v1/auth/verify", headers)
            if value.get("valid") is not True or value.get("owner_id") != self.owner_id:
                raise ReadUnavailable("credential differs from pinned host principal")
            self._verified_digest = digest

    async def projection(self, expected) -> Mapping:
        headers = self._headers("application/json")
        await self._verify(headers)
        binding = expected.binding
        path = (
            "/api/v1/lease-groups/"
            + quote(binding.group_id, safe="")
            + "/members/"
            + quote(binding.lease_id, safe="")
            + "/authority"
        )
        return await self._json(path, headers)

    @asynccontextmanager
    async def events(self, cursor: str | None):
        headers = self._headers("text/event-stream")
        await self._verify(headers)
        if cursor is not None:
            if not cursor.isascii() or not cursor.isdecimal():
                raise ReadUnavailable("invalid event cursor")
            headers["Last-Event-ID"] = cursor
        async with self._client.stream(
            "GET", "/api/v1/events", headers=headers
        ) as response:
            if (
                response.status_code != 200
                or response.headers.get("content-type", "").split(";")[0]
                != "text/event-stream"
            ):
                raise ReadUnavailable("authority event stream unavailable")
            yield response

    async def close(self) -> None:
        await self._client.aclose()


class HostAuthorityAdapter:
    """Serialized callbacks on the owning loop; no extra watchdog/polling loop.

    A newer wakeup invalidates any older in-flight read. An accepted renewal is
    applied only after its native expiry event is armed. Stream loss stops new
    admission while retaining the last explicit alarm and the same owned child.
    """

    def __init__(
        self,
        actor: RankLifecycle,
        reader: HostPlaneReader,
        *,
        alarm_factory=ContinuousAuthorityAlarm,
    ):
        if reader.owner_id == actor.expected.binding.owner_id:
            raise ValueError("coordinator credential must not be installed on a host")
        self.actor, self.reader = actor, reader
        self._epoch = 0
        self._lock = asyncio.Lock()
        self.cursor = None
        self._stream_running = False
        self._closed = False
        self.settlement_observed = False
        self.before_settlement = lambda: None
        self.alarm = alarm_factory(
            asyncio.get_running_loop(), self._expired, self._alarm_failed
        )

    def invalidate(self, reason: str) -> int:
        self._epoch += 1
        self.actor.require_observation(reason)
        return self._epoch

    def _expired(self) -> None:
        self.actor.authority_alarm()
        deadline = self.actor.authority_deadline
        if not self.actor.draining and deadline is not None:
            # A native event may be rounded early; never extend the deadline.
            try:
                self.alarm.arm(deadline)
            except (OSError, ValueError, RuntimeError):
                self._alarm_failed()

    def _alarm_failed(self) -> None:
        try:
            self.invalidate("native authority alarm unavailable")
        finally:
            self.actor.request_stop("explicit authority expiry cannot be enforced")

    async def refresh(self, reason: str = "fresh authority read"):
        if self._closed:
            raise ObservationUnknown("authority adapter is closed")
        epoch = self.invalidate(reason)  # Before awaiting the lock or network.
        async with self._lock:
            if epoch != self._epoch:
                return None
            started = self.actor.monotonic()
            projection = None
            try:
                projection = await self.reader.projection(self.actor.expected)
            except asyncio.CancelledError:
                self.invalidate("authority read cancelled; same attempt retained")
                raise
            except (httpx.HTTPError, ReadUnavailable, OSError, ValueError):
                pass  # No response detail, credential or caller body is logged.
            received = self.actor.monotonic()
            if epoch != self._epoch:
                return None  # A later event/read must be reconciled first.
            if (not self.settlement_observed and self.actor.receipt is not None
                    and self.actor.phase == 'awaiting_settlement'
                    and settlement_confirmed(self.actor.expected, projection)):
                self.before_settlement()
                self.settlement_observed = True
            candidate = check_projection(
                self.actor.expected,
                projection,
                request_started_monotonic=started,
                received_monotonic=received,
                now_monotonic=self.actor.monotonic(),
            )
            if candidate.state == "authorized":
                try:
                    self.alarm.arm(candidate.expires_at_monotonic)
                except (OSError, ValueError, RuntimeError):
                    projection = None  # Old alarm/last_good remain intact.
            return self.actor.observe_authority(
                projection,
                request_started_monotonic=started,
                received_monotonic=received,
            )

    async def on_connect(self):
        return await self.refresh("event stream connected; fresh snapshot required")

    async def on_heartbeat(self):
        return await self.refresh("event stream heartbeat")

    async def on_event(self, event: Mapping):
        if not isinstance(event, Mapping):
            self.invalidate("event observation malformed")
            return None
        if event_requires_read(self.actor.expected, event):
            return await self.refresh("scoped member event")
        return None

    def on_disconnect(self) -> None:
        self.invalidate("event stream disconnected; same attempt retained")

    async def before_start(self, *, environment: Mapping[str, str]) -> None:
        await self.refresh("before rank launch")
        self.actor.start(environment=environment)

    async def before_admit(self) -> bool:
        await self.refresh("before inference admission")
        return self.actor.may_admit()

    async def consume_once(self) -> None:
        """Optional use of the existing event subscription; never auto-reconnect.

        A service already consuming SSE routes the public callbacks instead.
        Every stream frame is a reconciliation opportunity, including unrelated
        visible activity which can suppress the server's keepalive comments.
        """
        if self._stream_running or self._closed or getattr(self.reader, '_event_feed_owner', None) is not None:
            raise ObservationUnknown("one existing event subscription only")
        self._stream_running = True
        self.invalidate("opening event stream")
        try:
            async with self.reader.events(self.cursor) as response:
                await self.on_connect()
                data, event_id, event_type, size = [], None, None, 0
                async for line in response.aiter_lines():
                    size += len(line)
                    if size > 128 * 1024:
                        raise ReadUnavailable("authority event frame too large")
                    if line.startswith(":"):
                        await self.on_heartbeat()
                    elif line:
                        name, _, value = line.partition(":")
                        value = value[1:] if value.startswith(" ") else value
                        if name == "data":
                            data.append(value)
                        elif name == "id":
                            event_id = value
                        elif name == "event":
                            event_type = value
                    else:
                        if data:
                            event = json.loads("\n".join(data))
                            if not isinstance(event, dict) or (
                                event_type and event_type != event.get("type")
                            ):
                                raise ReadUnavailable("authority event framing invalid")
                            if event_id is not None:
                                if not event_id.isascii() or not event_id.isdecimal():
                                    raise ReadUnavailable("authority event id invalid")
                                self.cursor = event_id
                            if event_requires_read(self.actor.expected, event):
                                await self.on_event(event)
                            else:
                                await self.refresh("visible stream activity")
                        data, event_id, event_type, size = [], None, None, 0
                raise ReadUnavailable(
                    "authority stream ended; reconnect snapshot required"
                )
        finally:
            self.on_disconnect()
            self._stream_running = False

    async def close_after_cleanup(self) -> None:
        if self.actor.launch_attempted and self.actor.phase != "awaiting_settlement":
            raise ObservationUnknown("owned attempt still requires alarm and cleanup")
        if self._stream_running:
            raise ObservationUnknown("event stream must finish before closing adapter")
        self._closed = True
        self.invalidate("authority adapter closed after cleanup")
        self.alarm.close()
        await self.reader.close()
