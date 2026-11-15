"""One explicit registry subscription per principal, never a polling watchdog.

The service owns this coroutine and any explicit reconnect. No timer, automatic
retry, compute submission, lease mutation or permission from an event is added.
Every callback still performs its own fresh authority read before acting.
"""

import asyncio

from .http import decode_json


class EventStreamUnavailable(RuntimeError):
    pass


async def _lines(response):
    """Bound both unfinished lines and whole frames before buffering bytes."""
    pending = bytearray()
    size = 0
    async for chunk in response.aiter_bytes():
        offset = 0
        while offset < len(chunk):
            end = chunk.find(b'\n', offset)
            stop = len(chunk) if end < 0 else end + 1
            fragment = memoryview(chunk)[offset:stop]
            if size + len(fragment) > 128 * 1024:
                raise EventStreamUnavailable('registry event frame exceeds bound')
            size += len(fragment)
            pending.extend(fragment)
            offset = stop
            if end >= 0:
                line = bytes(pending[:-1]).removesuffix(b'\r').decode('utf-8')
                pending.clear()
                yield line
                if not line:
                    size = 0
    if pending:
        raise EventStreamUnavailable('registry event stream ended in a line')


class RegistryEventFeed:
    """Exact credential/participants, stable cursor, one owned stream at a time."""

    @classmethod
    def coordinator(cls, plane, actors):
        from .acquire import GroupAcquisition
        from .plane import CoordinatorPlane
        actors = tuple(actors)
        if (not isinstance(plane, CoordinatorPlane) or not actors
                or len({id(actor) for actor in actors}) != len(actors)
                or any(not isinstance(actor, GroupAcquisition) or actor.plane is not plane
                       or actor.plan.owner_id != plane.owner_id or actor._closed for actor in actors)):
            raise ValueError('exact shared coordinator principal and live actors required')
        return cls(plane, actors, None)

    @classmethod
    def host(cls, owner):
        from .rank_runtime import HostRankRuntime
        if not isinstance(owner, HostRankRuntime) or owner.closed or owner.adapter._closed:
            raise ValueError('live registered host owner required')
        if owner.adapter._stream_running:
            raise ValueError('host principal already has an event subscription')
        return cls(owner.adapter.reader, (), owner)

    def __init__(self, reader, actors, host):
        if getattr(reader, '_event_feed_owner', None) is not None:
            raise ValueError('this principal reader already owns a registry feed')
        self.reader, self.actors, self.host = reader, actors, host
        self.cursor = None
        self.running = False
        self.closed = False
        reader._event_feed_owner = self

    def _disconnect(self):
        failed = False
        if self.host is not None:
            try:
                self.host.on_disconnect()
            except Exception:
                failed = True
        else:
            for actor in self.actors:
                try:
                    actor.events_lost()
                except Exception:
                    failed = True
        if failed:
            raise EventStreamUnavailable('registry participant invalidation unavailable')

    async def _connect(self):
        if self.host is not None:
            # The runtime wrapper updates its own event epoch as well as the
            # adapter's. Calling adapter.consume_once alone misses this gate.
            await self.host.on_connect()
        else:
            for actor in self.actors:
                actor.events_reconciled()

    async def _heartbeat(self):
        if self.host is not None:
            await self.host.on_heartbeat()
        else:
            for actor in self.actors:
                actor.wake()

    async def _event(self, event):
        if self.host is not None:
            from .member import event_requires_read
            if event_requires_read(self.host.actor.expected, event):
                await self.host.on_event(event)
            else:
                await self.host.on_heartbeat()
        else:
            for actor in self.actors:
                if not actor.on_event(event):
                    actor.wake()  # Visible activity can suppress keepalives.

    async def consume_once(self):
        if self.closed or self.running or self.host is not None and (
                self.host.closed or self.host.adapter._closed or self.host.adapter._stream_running):
            raise EventStreamUnavailable('one live owned registry subscription required')
        self.running = True
        if self.host is not None:
            self.host.adapter._stream_running = True
        try:
            self._disconnect()
            async with self.reader.events(self.cursor) as response:
                await self._connect()
                data, event_id, event_type = [], None, None
                stream_cursor = None
                async for line in _lines(response):
                    if line.startswith(':'):
                        await self._heartbeat()
                    elif line:
                        name, _, value = line.partition(':')
                        value = value[1:] if value.startswith(' ') else value
                        if name == 'data':
                            data.append(value)
                        elif name in ('id', 'event'):
                            if name == 'id':
                                if event_id is not None or not value.isascii() or not value.isdecimal():
                                    raise EventStreamUnavailable('registry event id invalid')
                                event_id = value
                            else:
                                if event_type is not None:
                                    raise EventStreamUnavailable('registry event type repeated')
                                event_type = value
                    else:
                        if data:
                            event = decode_json('\n'.join(data))
                            if (not isinstance(event.get('type'), str) or not event['type']
                                    or event_type and event_type != event['type']):
                                raise EventStreamUnavailable('registry event type differs')
                            if event_id is not None and stream_cursor is not None and int(event_id) < int(stream_cursor):
                                raise EventStreamUnavailable('registry event cursor moved backwards')
                            await self._event(event)
                            # A failed callback must not acknowledge this event.
                            if event_id is not None:
                                self.cursor = event_id
                                # Core IDs are monotonic per process, not
                                # globally. A new connection reconciles exact
                                # state before accepting a new lower epoch.
                                stream_cursor = event_id
                        data, event_id, event_type = [], None, None
                raise EventStreamUnavailable('registry stream ended; reconcile same owner')
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never emit remote body, credentials, prompts or arbitrary errors.
            raise EventStreamUnavailable('registry event observation unavailable') from None
        finally:
            try:
                self._disconnect()
            finally:
                self.running = False
                if self.host is not None:
                    self.host.adapter._stream_running = False

    def close(self):
        if self.running:
            raise EventStreamUnavailable('finish the same event coroutine before close')
        self._disconnect()
        self.closed = True
