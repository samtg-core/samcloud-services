"""Service-loop owner for one exact host rank; no independent watchdog loop.

The existing service routes plane/inventory events to this object. It owns one
foreground child-exit subscription and the already explicit member-expiry alarm.
Neither child exit nor closing this owner releases or settles a physical lease.
"""

import asyncio
from dataclasses import asdict, dataclass
import errno
import hashlib
import os
import select
import sqlite3
import sys
from uuid import UUID

from .host_authority import HostAuthorityAdapter
from .cleanup import NoStartExpectation, validate_cleanup
from .evidence import write_new
from .prepare import canonical, digest
from .rank_lifecycle import RankLifecycle
from .rank_process import ObservationUnknown, OwnedRankProcess


class LaunchPreflightPending(ObservationUnknown):
    """A superseded authority read before the durable launch decision.

    Only the existing owner may resume these reads. This is never evidence that
    a committed launch, Popen, or an inference request can be repeated.
    """


@dataclass(frozen=True)
class RuntimeObservation:
    runtime_sha256: str
    launch_sha256: str
    boot_id: str
    uid: int


def launch_sha256(actor, environment):
    return spec_launch_sha256(actor.spec, environment)


def spec_launch_sha256(spec, environment):
    # Do not publish raw environment values, which can contain the private EXO
    # API configuration. The coordinator's plane credential never belongs here.
    return digest({'binding': asdict(spec.binding), 'argv': list(spec.argv),
                   'cwd': str(spec.cwd), 'environment_sha256': digest(environment)})


def record_host_attempt(owner, journal):
    """Same host inference journal: never rebind an old member as a new run.

    This precedes binding/launch. A lost owner after the commit requires exact
    attempt recovery, including when no child/output was observed.
    """
    binding = owner.actor.spec.binding
    if owner._journal is not None:
        raise ObservationUnknown('host member already registered; no new attempt')
    actor = owner.actor
    baseline = actor.memory_reader()
    if (actor.launch_attempted or actor.draining or actor.phase != 'prepared'
            or actor.process is not None or actor.boot_reader() != owner.boot_id
            or type(baseline) is not int or baseline < 0
            or not all(actor.port_reader(v) is True for v in actor.spec.listeners)
            or launch_sha256(actor, owner._environment) != owner.launch_sha256):
        raise ObservationUnknown('fresh cold prepared host attempt required')
    attempt = digest({'binding': asdict(binding), 'launch_sha256': owner.launch_sha256,
                      'boot_id': owner.boot_id, 'state_path': str(actor.state)})
    prepared = NoStartExpectation(binding, owner.boot_id, owner.launch_sha256, attempt,
        actor.spec.listeners, baseline, actor.spec.cleanup_tolerance_bytes, actor.clock().isoformat())
    prepared_json = canonical(asdict(prepared)).decode()
    with journal.transaction():
        journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_host_rank_attempts (
            resource_id TEXT NOT NULL, lease_id TEXT NOT NULL,
            group_id TEXT NOT NULL, generation INTEGER NOT NULL,
            runtime_sha256 TEXT NOT NULL, launch_sha256 TEXT NOT NULL,
            boot_id TEXT NOT NULL, state_path TEXT NOT NULL,
            phase TEXT NOT NULL DEFAULT 'unknown', prepared_json TEXT,
            PRIMARY KEY(resource_id,lease_id), UNIQUE(group_id,generation,resource_id)
        )''')
        columns = {row['name'] for row in journal.db.execute('PRAGMA table_info(exo_host_rank_attempts)')}
        # Old rows never recorded a launch decision. They remain unknown,
        # not retrospectively declared unlaunched after a source upgrade.
        for name, declaration in (('phase', "TEXT NOT NULL DEFAULT 'unknown'"),
                                  ('prepared_json', 'TEXT')):
            if name not in columns:
                journal.db.execute(f'ALTER TABLE exo_host_rank_attempts ADD COLUMN {name} {declaration}')
    try:
        with journal.transaction():
            journal.db.execute('''INSERT INTO exo_host_rank_attempts
                (resource_id,lease_id,group_id,generation,runtime_sha256,launch_sha256,
                 boot_id,state_path,phase,prepared_json) VALUES(?,?,?,?,?,?,?,?,'prepared',?)''',
                (binding.resource_id, binding.lease_id, binding.group_id, binding.generation,
                 binding.runtime_sha256, owner.launch_sha256, owner.boot_id, str(actor.state), prepared_json))
    except sqlite3.IntegrityError:
        raise ObservationUnknown('host member attempt already owned; inspect it, do not replay') from None
    # A failure here leaves the DB claim retained, with no enabled owner/start.
    write_new(actor.state / 'host-prepared.json', asdict(prepared))
    owner._journal, owner.prepared = journal, prepared
    actor.launch_guard = lambda: owner._transition('launch_committed', 'spawning')


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
                 watcher_factory=ChildExitWatch, verify_unstarted=None):
        if (not isinstance(actor, RankLifecycle) or not isinstance(adapter, HostAuthorityAdapter)
                or adapter.actor is not actor or not callable(verify_runtime)
                or not callable(watcher_factory)
                or verify_unstarted is not None and not callable(verify_unstarted)
                or not isinstance(environment, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items())
                or any(any(part in k.upper() for part in ('TOKEN', 'BEARER', 'SAMCLOUD_AUTH'))
                       and not (k == 'EXO_LEASED_TOKEN_TRACE' and environment[k] == '16')
                       for k in environment)
                or actor.launch_attempted or actor.phase != 'prepared'
                or not isinstance(boot_id, str) or str(UUID(boot_id)) != boot_id
                or UUID(boot_id).int == 0):
            raise ValueError('fresh exact host actor, runtime verifier and boot required')
        self.actor, self.adapter = actor, adapter
        self.verify_runtime, self.boot_id = verify_runtime, boot_id
        self.verify_unstarted = verify_runtime if verify_unstarted is None else verify_unstarted
        self._environment = dict(environment)
        self.launch_sha256 = launch_sha256(actor, self._environment)
        self._watcher_factory, self.exit_watch = watcher_factory, None
        self._lock, self._closed = asyncio.Lock(), False
        self._event_epoch, self._events_ready = 0, False
        self._connected = asyncio.Event()
        self.observation_error = None
        self._journal, self.prepared = None, None

    def _attempt(self):
        if self._journal is None or self.prepared is None:
            raise ObservationUnknown('host attempt must be journaled before launch or cleanup')
        binding = self.actor.spec.binding
        row = self._journal.db.execute('SELECT * FROM exo_host_rank_attempts '
            'WHERE resource_id=? AND lease_id=?', (binding.resource_id, binding.lease_id)).fetchone()
        if (row is None or row['prepared_json'] != canonical(asdict(self.prepared)).decode()
                or binding != self.prepared.binding or row['boot_id'] != self.boot_id
                or row['group_id'] != binding.group_id or row['generation'] != binding.generation
                or row['runtime_sha256'] != binding.runtime_sha256
                or row['state_path'] != str(self.actor.state)
                or row['launch_sha256'] != self.launch_sha256
                or launch_sha256(self.actor, self._environment) != self.launch_sha256):
            raise ObservationUnknown('exact journaled host attempt changed')
        return row

    def _transition(self, before, after):
        with self._journal.transaction():
            row = self._attempt()
            if row['phase'] != before:
                raise ObservationUnknown('host launch decision already committed; inspect same attempt')
            self._journal.db.execute('UPDATE exo_host_rank_attempts SET phase=? '
                'WHERE resource_id=? AND lease_id=? AND phase=?',
                (after, row['resource_id'], row['lease_id'], before))

    @property
    def closed(self):
        return self._closed

    async def start(self):
        async with self._lock:
            if self._closed or self.actor.launch_attempted:
                raise ObservationUnknown('host launch already owned; do not replay')
            if self._attempt()['phase'] != 'prepared':
                raise ObservationUnknown('host launch already committed or fenced')
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
                    or self.actor.boot_reader() != self.boot_id):
                raise ObservationUnknown('host runtime spec or boot changed during authority read')
            if not self._events_ready or epoch != self._event_epoch:
                raise LaunchPreflightPending('host event feed changed before launch decision')
            if self.actor.authority is None or not self.actor.authority.allows_start(self.actor.monotonic()):
                if not self.actor.draining:
                    raise LaunchPreflightPending('fresh host authority required before committing launch')
                raise ObservationUnknown('host authority withdrawn before committing launch')
            # Commit before any Popen or filesystem launch intent. A crash or
            # preflight failure here stays uncertain, never a no-start proof.
            self._transition('prepared', 'launch_committed')
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
            if self._journal is not None and self._attempt()['phase'] == 'launch_fenced':
                return self._never_started_cleanup()
            return self.actor.refresh_cleanup()
        return self.actor.receipt

    async def attest_never_started(self):
        """Fence this registered attempt permanently, then inspect it fresh.

        Uses the same lock as start. No contact/elapsed-time or absent PID is
        accepted as a substitute for the durable negative launch decision.
        """
        async with self._lock:
            if (self._closed or self.actor.launch_attempted or self.actor.process is not None
                    or (self.actor.state / 'launch-intent.json').exists()):
                raise ObservationUnknown('launch may have occurred; inspect the exact attempt')
            row = self._attempt()
            if row['phase'] not in ('prepared', 'launch_fenced'):
                raise ObservationUnknown('launch outcome is not provably unstarted')
            # A configured preparation verifier can omit launch-only transport
            # permission. It still proves the exact files/spec/boot; the same
            # lock, durable negative CAS, ports and memory prove no compute.
            # start() never uses this verifier.
            observed = await self.verify_unstarted()
            if (not isinstance(observed, RuntimeObservation)
                    or observed.runtime_sha256 != self.actor.spec.binding.runtime_sha256
                    or observed.launch_sha256 != self.launch_sha256
                    or type(observed.uid) is not int or observed.uid != os.getuid()
                    or observed.boot_id != self.boot_id or self.actor.boot_reader() != self.boot_id):
                raise ObservationUnknown('fresh host runtime/boot unconfirmed for no-start proof')
            if row['phase'] == 'prepared':
                self._transition('prepared', 'launch_fenced')
            self.actor.request_stop('registered host attempt permanently fenced before launch')
            return self._never_started_cleanup()

    def _never_started_cleanup(self):
        actor = self.actor
        if (self._attempt()['phase'] != 'launch_fenced' or actor.launch_attempted
                or actor.process is not None or (actor.state / 'launch-intent.json').exists()):
            raise ObservationUnknown('never-started proof requires the same fenced host attempt')
        if actor.receipt is not None:
            return actor.receipt
        if actor.audit_write_failures:
            if actor._record_stop('audit_write_failures', writes=list(actor.audit_write_failures)) is None:
                return None
            actor.audit_write_failures.clear()
        pinned = self.prepared
        try:
            if actor.boot_reader() != self.boot_id:
                raise ObservationUnknown('host boot changed')
            ports = [actor.port_reader(v) for v in pinned.listeners]
            memory = actor.memory_reader()
        except (ObservationUnknown, OSError):
            self.observation_error = 'never_started_cleanup_unconfirmed'
            return None
        observation = {k: v for k, v in asdict(pinned).items() if k != 'listeners'}
        observation.update(launch_fenced=True, launch_intent_absent=True, launch_attempted=False,
            processes_started=0, attempt_phase='launch_fenced',
            listeners_closed=ports, observed_in_use_bytes=memory, observed_at=actor.clock().isoformat())
        value = {'kind': 'observed_rank_never_started', **asdict(pinned.binding),
                 'observation': observation, 'lease_released': False, 'settlement_recorded': False}
        # Incomplete memory/port observations retain the fence and claim until
        # an existing inventory event or explicit reconciliation observes them.
        try:
            validate_cleanup(canonical(value), pinned)
        except ValueError:
            return None
        path = actor.state / 'cleanup-evidence.json'
        if path.exists():
            # A previous evidence write may have succeeded before an audit
            # notification failed. Preserve that exact artifact, never replace.
            validate_cleanup(path.read_bytes(), pinned)
        else:
            write_new(path, value)
        actor.phase, actor.draining = 'awaiting_settlement', True
        actor.receipt = actor._record_stop('cleanup_required',
            reason='registered attempt never launched; exact owner settlement required',
            evidence_path=str(path), evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            lease_released=False, settlement_recorded=False)
        if actor.receipt is not None:
            self.observation_error = None
        return actor.receipt

    async def on_connect(self):
        self._events_ready = False
        self._connected.clear()
        self._event_epoch += 1
        epoch = self._event_epoch
        result = await self.adapter.on_connect()
        if epoch == self._event_epoch:
            self._events_ready = True
            self._connected.set()
        self.inventory_changed()
        return result

    async def consume_events_once(self):
        """Explicit service-owned subscription, including this runtime's gates."""
        from .events import RegistryEventFeed
        feed = getattr(self, '_event_feed', None)
        if feed is None:
            feed = self._event_feed = RegistryEventFeed.host(self)
        return await feed.consume_once()

    async def wait_connected(self, subscription):
        """Wait on the existing event transition, never poll or own its task."""
        waiter = asyncio.create_task(self._connected.wait())
        try:
            await asyncio.wait((waiter, subscription), return_when=asyncio.FIRST_COMPLETED)
            if subscription.done() or not self._events_ready:
                raise ObservationUnknown('host event observation unavailable; same attempt retained')
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

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
        self._connected.clear()
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
        if self._journal is not None and (self.actor.receipt is None
                                         or self.actor.phase != 'awaiting_settlement'):
            raise ObservationUnknown('registered host owner still needs observed cleanup')
        await self.adapter.close_after_cleanup()
        if self.exit_watch is not None:
            self.exit_watch.close()
        self._closed = True
