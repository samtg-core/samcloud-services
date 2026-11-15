"""Service-loop owner for one exact host rank; no independent watchdog loop.

The existing service routes plane/inventory events to this object. It owns one
foreground child-exit subscription and the already explicit member-expiry alarm.
Neither child exit nor closing this owner releases or settles a physical lease.
"""

import asyncio
from dataclasses import asdict, dataclass
import errno
import os
import select
import sqlite3
import sys
from uuid import UUID

from .host_authority import HostAuthorityAdapter
from .prepare import digest
from .rank_lifecycle import RankLifecycle
from .rank_process import ObservationUnknown, OwnedRankProcess


@dataclass(frozen=True)
class RuntimeObservation:
    runtime_sha256: str
    launch_sha256: str
    boot_id: str
    uid: int


def launch_sha256(actor, environment):
    # Do not publish raw environment values, which can contain the private EXO
    # API configuration. The coordinator's plane credential never belongs here.
    return digest({'binding': asdict(actor.spec.binding), 'argv': list(actor.spec.argv),
                   'cwd': str(actor.spec.cwd), 'environment_sha256': digest(environment)})


def record_host_attempt(owner, journal):
    """Same host inference journal: never rebind an old member as a new run.

    This precedes binding/launch. A lost owner after the commit requires exact
    attempt recovery, including when no child/output was observed.
    """
    binding = owner.actor.spec.binding
    try:
        with journal.transaction():
            journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_host_rank_attempts (
                resource_id TEXT NOT NULL, lease_id TEXT NOT NULL,
                group_id TEXT NOT NULL, generation INTEGER NOT NULL,
                runtime_sha256 TEXT NOT NULL, launch_sha256 TEXT NOT NULL,
                boot_id TEXT NOT NULL, state_path TEXT NOT NULL,
                PRIMARY KEY(resource_id,lease_id), UNIQUE(group_id,generation,resource_id)
            )''')
            journal.db.execute('INSERT INTO exo_host_rank_attempts VALUES(?,?,?,?,?,?,?,?)',
                (binding.resource_id, binding.lease_id, binding.group_id, binding.generation,
                 binding.runtime_sha256, owner.launch_sha256, owner.boot_id, str(owner.actor.state)))
    except sqlite3.IntegrityError:
        raise ObservationUnknown('host member attempt already owned; inspect it, do not replay') from None


class ChildExitWatch:
    """One actual Popen child, using EVFILT_PROC/NOTE_EXIT on the owning loop."""
    def __init__(self, child, notify, failed):
        if sys.platform != 'darwin' or not isinstance(child, OwnedRankProcess):
            raise ObservationUnknown('qualified owned Darwin child required')
        self.child, self.notify, self.failed = child, notify, failed
        self.loop = asyncio.get_running_loop()
        self._closed, self._notified = False, False
        self._queue = select.kqueue()
        self._pid = child.identity.pid
        try:
            self.loop.add_reader(self._queue.fileno(), self._ready)
            if child.exit_code() is not None:
                self.loop.call_soon(self._finished)
                return
            try:
                self._queue.control([select.kevent(self._pid, filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                    fflags=select.KQ_NOTE_EXIT)], 0, 0)
            except OSError as error:
                # Exit between the retained-child check and registration is a
                # real race. Inspect the same handle; never adopt a replacement.
                if error.errno != errno.ESRCH or child.exit_code() is None:
                    raise
                self.loop.call_soon(self._finished)
            if child.exit_code() is not None:
                self.loop.call_soon(self._finished)
        except BaseException:
            self.close()
            raise

    def _finished(self):
        if self._closed or self._notified:
            return
        if self.child.exit_code() is None:
            self.failed()
            return
        self._notified = True
        self.close()
        self.notify()

    def _ready(self):
        if self._closed:
            return
        try:
            for event in self._queue.control(None, 8, 0):
                if event.ident != self._pid:
                    continue
                if event.flags & select.KQ_EV_ERROR:
                    self.failed()
                elif event.fflags & select.KQ_NOTE_EXIT:
                    self._finished()
        except OSError:
            self.failed()

    def close(self):
        if not self._closed:
            self._closed = True
            self.loop.remove_reader(self._queue.fileno())
            self._queue.close()


class HostRankRuntime:
    """Explicit installation on an existing host service, not a daemon entrypoint.

    verify_runtime must freshly attest the complete configured runtime, own-host
    files, boot and launch spec. No default verifier or permission from a saved
    observation. The actor/adapter keep the physical attempt alive across caller
    disconnect; an uncertain launch is never replayed.
    """
    def __init__(self, actor, adapter, *, verify_runtime, environment, boot_id,
                 watcher_factory=ChildExitWatch):
        if (not isinstance(actor, RankLifecycle) or not isinstance(adapter, HostAuthorityAdapter)
                or adapter.actor is not actor or not callable(verify_runtime)
                or not callable(watcher_factory) or not isinstance(environment, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items())
                or any(any(part in k.upper() for part in ('TOKEN', 'BEARER', 'SAMCLOUD_AUTH'))
                       for k in environment)
                or actor.launch_attempted or actor.phase != 'prepared'
                or not isinstance(boot_id, str) or str(UUID(boot_id)) != boot_id
                or UUID(boot_id).int == 0):
            raise ValueError('fresh exact host actor, runtime verifier and boot required')
        self.actor, self.adapter = actor, adapter
        self.verify_runtime, self.boot_id = verify_runtime, boot_id
        self._environment = dict(environment)
        self.launch_sha256 = launch_sha256(actor, self._environment)
        self._watcher_factory, self.exit_watch = watcher_factory, None
        self._lock, self._closed = asyncio.Lock(), False
        self._event_epoch, self._events_ready = 0, False
        self.observation_error = None

    @property
    def closed(self):
        return self._closed

    async def start(self):
        async with self._lock:
            if self._closed or self.actor.launch_attempted:
                raise ObservationUnknown('host launch already owned; do not replay')
            if not self._events_ready:
                raise ObservationUnknown('reconciled existing host event feed required')
            epoch = self._event_epoch
            observed = await self.verify_runtime()
            if (not isinstance(observed, RuntimeObservation)
                    or observed.runtime_sha256 != self.actor.spec.binding.runtime_sha256
                    or observed.launch_sha256 != self.launch_sha256
                    or type(observed.uid) is not int or observed.uid != os.getuid()
                    or observed.boot_id != self.boot_id
                    or self.actor.boot_reader() != self.boot_id):
                raise ObservationUnknown('exact fresh host runtime/boot/spec unconfirmed')
            await self.adapter.refresh('before verified rank launch')
            if (launch_sha256(self.actor, self._environment) != self.launch_sha256
                    or self.actor.boot_reader() != self.boot_id
                    or not self._events_ready or epoch != self._event_epoch):
                raise ObservationUnknown('host runtime spec or boot changed during authority read')
            self.actor.start(environment=dict(self._environment))
            if self.actor.host_boot != self.boot_id:
                self.actor.request_stop('host boot changed during launch')
                raise ObservationUnknown('host boot changed during launch; inspect same attempt')
            try:
                self.exit_watch = self._watcher_factory(
                    self.actor.process, self._child_exited, self._observation_failed)
            except BaseException:
                self._observation_failed()
                raise

    def _observation_failed(self):
        self.observation_error = 'child_exit_unconfirmed'
        try:
            self.adapter.invalidate('owned child-exit observation unavailable')
        finally:
            self.actor.request_stop('owned child-exit observation unavailable')

    def _child_exited(self):
        try:
            self.actor.process_exited()
        except (OSError, ObservationUnknown):
            self._observation_failed()

    def inventory_changed(self):
        """Existing inventory/device events or explicit owner reconciliation."""
        if self.actor.draining and self.actor.receipt is None:
            return self.actor.refresh_cleanup()
        return self.actor.receipt

    async def on_connect(self):
        self._events_ready = False
        self._event_epoch += 1
        epoch = self._event_epoch
        result = await self.adapter.on_connect()
        if epoch == self._event_epoch:
            self._events_ready = True
        self.inventory_changed()
        return result

    async def on_event(self, event):
        result = await self.adapter.on_event(event)
        self.inventory_changed()
        return result

    async def on_heartbeat(self):
        result = await self.adapter.on_heartbeat()
        self.inventory_changed()
        return result

    def on_disconnect(self):
        self._events_ready = False
        self._event_epoch += 1
        self.adapter.on_disconnect()

    async def may_admit(self):
        if self._closed or not self._events_ready:
            return False
        epoch = self._event_epoch
        admitted = await self.adapter.before_admit()
        return admitted and self._events_ready and epoch == self._event_epoch

    def stop(self, reason):
        self.actor.request_stop(reason)
        return self.inventory_changed()

    async def close_after_cleanup(self):
        # Adapter refuses while this attempt is running/unknown. An observation
        # failure must not silently drop the explicit stop alarm or child owner.
        await self.adapter.close_after_cleanup()
        if self.exit_watch is not None:
            self.exit_watch.close()
        self._closed = True
