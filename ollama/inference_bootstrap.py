"""Explicit queue installation and trusted capacity-event boundary.

Not invoked by the deployed service launcher. A deployment must supply an exact
model/resource map and authoritative capacity events; neither an HTTP caller nor
a backend health response can enable capacity. Startup/restart begins UNKNOWN.
This is scheduling permission, not a replacement for physical memory leases.
"""
import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .inference_arbiter import StintPolicy
from .inference_gateway import GatewayQueue
from . import capacity


def vision_backend_absent(host,port):
    """A refused local connection is absence; silence or other errors are not."""
    import errno
    import socket
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        return probe.connect_ex((host,port))==errno.ECONNREFUSED


@dataclass(frozen=True)
class QueueSettings:
    journal: Path
    models: Mapping[str, str]
    stint_policy: StintPolicy = field(default_factory=StintPolicy)
    reservation_mb: Mapping[str, int] = field(default_factory=dict)
    source_path: Path | None = None
    source_sha256: str | None = None
    shared_queue_wait_s: int | None = None
    shared_queue_enabled: bool = False

    @property
    def shared_queue(self):
        # Preserve explicit finite-wait configurations while allowing an
        # independent queue opt-in without a default deadline.
        return self.shared_queue_enabled or self.shared_queue_wait_s is not None

    @classmethod
    def from_file(cls, path):
        """Read explicit staged configuration without enabling the gateway.

        No environment switch is exposed until production startup can bind the
        qualified capacity/ownership source. Parsing performs no registry or
        provider operation and does not create the inference journal.
        """
        requested = Path(path)
        if not requested.is_absolute():
            raise ValueError('queue configuration requires an explicit absolute path')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate queue configuration key')
                result[key] = value
            return result
        try:
            resolved = requested.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError('queue configuration must be a regular file')
            raw = resolved.read_bytes()
            document = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
            fields = {'journal', 'models', 'reservation_mb', 'stint_policy', 'shared_queue_wait_s', 'shared_queue_enabled'}
            if (not isinstance(document, dict) or not {'journal','models'} <= set(document)
                    or set(document) - fields):
                raise ValueError('queue configuration requires journal and models, with optional sizing and scheduling policy')
            reservations = document.get('reservation_mb', {})
            policy = document.get('stint_policy', {})
            if (not isinstance(policy,dict) or
                    set(policy)-{'idle_grace_seconds','fairness_seconds'}):
                raise ValueError('explicit service-owned stint policy required')
            if (not isinstance(document['journal'], str) or
                    not isinstance(document['models'], dict) or
                    not isinstance(reservations, dict) or
                    (reservations and set(document['models']) != set(reservations))):
                raise ValueError('exact model/resource map and optional complete reservation map required')
            return cls(Path(document['journal']), document['models'],
                       stint_policy=StintPolicy(**policy),
                       reservation_mb=reservations, source_path=resolved,
                       source_sha256=hashlib.sha256(raw).hexdigest(),
                       shared_queue_wait_s=document.get('shared_queue_wait_s'),
                       shared_queue_enabled=document.get('shared_queue_enabled', False))
        except (OSError, UnicodeError, TypeError, ValueError):
            # Do not quote a malformed file's contents into operational logs.
            raise ValueError('Invalid queue startup configuration') from None

    def __post_init__(self):
        journal = Path(self.journal)
        if not journal.is_absolute():
            raise ValueError('queue journal must have an explicit absolute path')
        models = dict(self.models)
        if not models or any(not isinstance(value, str) or not value.strip()
                             for pair in models.items() for value in pair):
            raise ValueError('exact model/resource configuration is required')
        if not isinstance(self.stint_policy, StintPolicy):
            raise ValueError('explicit scheduling policy is required')
        if type(self.shared_queue_enabled) is not bool:
            raise ValueError('shared_queue_enabled must be a boolean')
        reservations = dict(self.reservation_mb)
        if (self.shared_queue_wait_s is not None and
                (type(self.shared_queue_wait_s) is not int or self.shared_queue_wait_s <= 0)):
            raise ValueError('shared queue requires explicit positive metadata wait')
        if (set(reservations) - set(models) or
                any(type(value) is not int or value <= 0 for value in reservations.values())):
            raise ValueError('reservation budgets must name configured models with positive MiB')
        object.__setattr__(self, 'journal', journal)
        object.__setattr__(self, 'models', MappingProxyType(models))
        object.__setattr__(self, 'reservation_mb', MappingProxyType(reservations))
        if self.source_path is not None:
            source = Path(self.source_path)
            if (not source.is_absolute() or not isinstance(self.source_sha256, str) or
                    len(self.source_sha256) != 64 or
                    any(char not in '0123456789abcdef' for char in self.source_sha256)):
                raise ValueError('configuration provenance must name absolute source and SHA-256')
            object.__setattr__(self, 'source_path', source)
        elif self.source_sha256 is not None:
            raise ValueError('configuration hash requires its source path')


def cold_capacity(resource, required_mb, *, physical, registry, work_permitted):
    """Evaluate fresh trusted observations BEFORE any resident allocation.

    This does not authorize an already-resident model (which needs exact lease
    coverage and working-set evidence), nor infer permission from availability.
    The source must read current hardware and the exact registry resource; no
    timestamps, HTTP health checks or old cached offer can replace either read.
    A positive result is eligibility to TRY the registry allocation, not
    a promise that the subsequent allocation cannot conflict.
    """
    if not isinstance(resource, str) or not resource or type(required_mb) is not int or required_mb <= 0:
        raise ValueError('exact resource and positive reservation size required')
    result = {'required_mb':required_mb, 'physical_usable_mb':None, 'registry_available_mb':None}
    def verdict(state, reason):
        return result | {'state':state, 'reason':reason}
    if work_permitted is False:
        return verdict('unavailable', 'work_mode')
    if work_permitted is not True:
        return verdict('unknown', 'work_permission_unknown')
    available = physical.get('memory_available_mb') if isinstance(physical, dict) else None
    if type(available) is not int or available < 0:
        return verdict('unknown', 'physical_capacity_unknown')
    result['physical_usable_mb'] = capacity.usable_mb(available)
    if not isinstance(registry, dict) or registry.get('id') != resource:
        return verdict('unknown', 'registry_resource_unknown')
    if registry.get('allocation_mode') not in ('shared', 'exclusive'):
        return verdict('unknown', 'registry_allocation_unknown')
    if registry['allocation_mode'] != 'shared':
        return verdict('unavailable', 'resource_not_shared')
    leases = registry.get('leases')
    if not isinstance(leases, list) or any(not isinstance(row, dict) or
            row.get('status') not in ('active', 'queued') or
            row.get('exclusive') not in (0, 1) for row in leases):
        return verdict('unknown', 'registry_leases_unknown')
    if any(row['status'] == 'active' and row['exclusive'] == 1 for row in leases):
        return verdict('unavailable', 'exclusive_holder')
    registered = registry.get('available_memory_mb')
    if type(registered) is not int or registered < 0:
        return verdict('unknown', 'registry_capacity_unknown')
    result['registry_available_mb'] = registered
    if required_mb > registered:
        return verdict('unavailable', 'registry_capacity')
    if not capacity.fits(required_mb, available):
        return verdict('unavailable', 'physical_capacity')
    return verdict('available', 'cold_fit')


class CapacityEvents:
    """Loop-local permission from one trusted, ordered event source.

    Revision numbers belong to this runtime's source, not to arbitrary observers.
    A duplicate/out-of-order event cannot resurrect capacity. Contact loss must
    explicitly publish unknown; silence is not an availability measurement.
    The source must send current truth after each restart: no cached permission
    is recovered from the inference journal. No polling or clock expiry here.
    """
    def __init__(self, resources):
        self._state = {resource: (0, 'unknown') for resource in resources}
        self._pump = None
        self._accepting = True
        self.guard = None

    def permitted(self, resource):
        if not self._accepting or self._state.get(resource,(0,'unknown'))[1] != 'available':
            return False
        try:
            return self.guard is None or self.guard(resource) is True
        except Exception:
            return False

    def pause(self):
        """Stop dispatch, not observation or already-owned backend work."""
        self._accepting = False

    def resume(self):
        """Require fresh authoritative capacity evidence after a handover check."""
        self._state = {resource:(revision, 'unknown')
                       for resource, (revision, _) in self._state.items()}
        self._accepting = True

    def observe(self, resource, revision, state):
        if resource not in self._state:
            raise ValueError('resource is not configured for this queue')
        if type(revision) is not int or revision <= 0:
            raise ValueError('positive source revision required')
        if state not in ('available', 'unavailable', 'unknown'):
            raise ValueError('explicit capacity state required')
        previous_revision, _ = self._state[resource]
        if revision <= previous_revision:
            return False
        self._state[resource] = (revision, state)
        if self.permitted(resource) and self._pump is not None:
            self._pump(resource)
        return True


@dataclass(frozen=True)
class QueueStartup:
    """Trusted launcher binding, not an HTTP option or live activation flag.

    Cold Ollama startup only. Existing residency needs explicit reconciliation;
    never adopt it by issuing a load/keep-alive request. Installing the runtime
    does not authorize dispatch: its capacity source still begins UNKNOWN.
    """
    settings: QueueSettings
    work_permitted: bool | None = None

    async def start(self, service):
        if not isinstance(self.settings,QueueSettings):
            raise TypeError('QueueSettings required')
        if self.work_permitted is not None and type(self.work_permitted) is not bool:
            raise ValueError('explicit work permission required')
        manager = service.mgr
        try:
            identity = await asyncio.to_thread(manager.sc.verify_token)
            owner = identity.get('owner_id') if isinstance(identity,dict) else None
            # Lease.agent_id stores AuthContext.identity(), not the display
            # username. Binding the latter passes synthetic name-only tests
            # but cannot confirm an actual registry receipt (user:<id>).
            if (not isinstance(identity,dict) or identity.get('valid') is not True or
                    not isinstance(owner,str) or not owner.strip()):
                raise ValueError('unverified identity')
        except Exception:
            raise RuntimeError('Durable startup registry identity is unverified') from None
        try:
            index = await asyncio.to_thread(manager.sc.api_index)
            route = index['endpoints']['leases']['rollover']
            if (not isinstance(route,dict) or route.get('method') != 'POST' or
                    route.get('path') != '/api/v1/leases/{id}/rollover'):
                raise ValueError('registry rollover contract absent')
            if self.settings.shared_queue:
                wait = index['endpoints']['leases']['wait']
                if wait.get('method') != 'GET' or not wait.get('path','').startswith('/api/v1/leases/{id}/wait'):
                    raise ValueError('registry queue contract absent')
                resources = set(self.settings.models.values())
                if len(resources) != 1:
                    raise ValueError('one shared capacity domain required')
                resource_id = resources.pop()
                resource = await asyncio.to_thread(manager.sc.get_resource, resource_id)
                if (not isinstance(resource,dict) or resource.get('id') != resource_id or
                        resource.get('expiry_policy') != 'hold'):
                    raise ValueError('shared queue requires fail-closed expiry accounting')
        except Exception:
            raise RuntimeError('Durable startup requires the deployed registry rollover contract'
                + (' and queue/hold policy' if self.settings.shared_queue else '')) from None
        try:
            running = await asyncio.to_thread(manager.ollama.list_running)
            if not isinstance(running,list) or running:
                raise ValueError('not cold')
        except Exception:
            raise RuntimeError('Durable startup requires confirmed empty Ollama inventory') from None
        from .manager import VLM_MODELS
        if any(info['default'] in self.settings.models for info in VLM_MODELS.values()):
            try:
                absent=await asyncio.to_thread(vision_backend_absent,
                    service.config.VLM_HOST,service.config.VLM_PORT)
                if absent is not True:raise ValueError('vision backend not cold')
            except Exception:
                raise RuntimeError('Durable startup requires confirmed absent vision backend') from None
        runtime = QueueRuntime.install(service,self.settings,registry_owner=owner)
        runtime.bind_capacity_source(work_permitted=self.work_permitted)
        await runtime.release_stranded()
        return runtime


class QueueRuntime:
    """Mount once, before serving, with all routes sharing the same dispatcher.

    Installation failures propagate: a launcher must fail startup, not catch an
    exception and fall back to legacy direct dispatch. Even after installation,
    resources cannot dispatch until their trusted capacity source says available.
    The trusted startup path binds that source to exact residency receipts;
    installation alone still cannot authorize work.
    """
    @classmethod
    def install(cls, service, settings, *, registry_owner=None):
        if not isinstance(settings, QueueSettings):
            raise TypeError('QueueSettings required')
        if getattr(service, 'inference_dispatcher', None) is not None:
            raise RuntimeError('an inference dispatcher is already installed')
        if getattr(service, 'inference_runtime', None) is not None:
            raise RuntimeError('an inference runtime is already installed')
        if any(getattr(route, 'path', '').startswith('/v1/inference')
               for route in service.app.routes):
            raise RuntimeError('inference routes are already installed')
        manager = getattr(service, 'mgr', None)
        initial = None
        if manager is not None:
            if (getattr(manager,'_background_started',False) or
                    getattr(manager,'models',{}) or getattr(manager,'_pool_leases',set()) or
                    getattr(manager,'_unreleased',{}) or getattr(manager,'residency_renewal',None) or
                    getattr(manager,'initial_reservations',None) or
                    getattr(manager,'durable_inference',False)):
                raise RuntimeError('Queue installation requires a fresh manager before startup adoption')
            if settings.reservation_mb and set(settings.reservation_mb) != set(settings.models):
                raise ValueError('explicit reservation overrides must cover every model')
            if set(settings.models.values()) != {service.config.SC_RESOURCE_ID}:
                raise ValueError('model resources must match this manager local allocation target')
            # The launcher must verify its own registry bearer before binding;
            # never infer this identity from an inference caller or a lease's
            # size. No manager mutation or mounted route on a failed binding.
            from .inference_reservation import InitialReservations
            initial = InitialReservations(settings.journal.parent/'initial-reservations.sqlite',
                manager.sc, registry_owner=registry_owner,
                service_id=service.config.SC_SERVICE_ID, models=settings.models,
                budgets=settings.reservation_mb, wait_s=settings.shared_queue_wait_s,
                wait=settings.shared_queue)
        capacity = CapacityEvents(set(settings.models.values()))
        queue = GatewayQueue(settings.journal, service, settings.models,
                             capacity.permitted, stint_policy=settings.stint_policy)
        try:
            service.app.include_router(queue.router())
        except BaseException:
            queue.supervisor.close()
            raise
        capacity._pump = queue.supervisor.capacity_available
        # No await: the durable API and compatibility routes become one unit
        # before the caller permits the ASGI app to receive requests.
        service.inference_dispatcher = queue
        if manager is not None:
            manager.initial_reservations = initial
            manager.durable_inference = True
        runtime = cls()
        runtime.queue, runtime.capacity = queue, capacity
        runtime.manager = manager
        runtime.settings = settings
        runtime.capacity_source = None
        runtime.residency_actor = None
        runtime.shared_resource = None
        service.inference_runtime = runtime
        if settings.source_path is not None:
            logging.getLogger(__name__).info('Inference queue config source=%s sha256=%s',
                                            settings.source_path, settings.source_sha256)
        return runtime

    async def release_stranded(self):
        """Startup only: end this service's own residency a dead predecessor left.

        A gateway that dies without releasing leaves its allocation covered and
        its exact lease active, while nothing it loaded survives: startup has
        confirmed Ollama empty. The live lease cannot be retired, and the registry
        keeps counting its bytes, so no fresh load fits (#1072: lease_34ff on
        12 Nov, lease_2f24 on 13 Nov). Release only the exact journaled pointer
        after a fresh absence read, re-read its terminal receipt, then retire.
        A pending rollover, a changed or foreign lease, or any unknown read is
        left as it was, and ordinary reconciliation still fails closed on it.
        """
        from .inference_arbiter import Conflict
        from .inference_reservation import needs_settlement
        manager,store = self.manager,self.residency_actor.store
        log = logging.getLogger(__name__)
        async def exact(resource,row):
            rows = await asyncio.to_thread(lambda:manager.sc.list_leases(
                resource_id=resource,service_id=store.service))
            matches = [item for item in rows if isinstance(item,dict) and
                       item.get('id') == row['lease_id']] if isinstance(rows,list) else []
            if len(matches) != 1:
                raise Conflict('exact allocation unknown')
            receipt = matches[0]
            if (receipt.get('agent_id') != store.owner or receipt.get('service_id') != store.service
                    or receipt.get('resource_id') != resource or receipt.get('exclusive') != 0
                    or receipt.get('memory_mb') != row['memory_mb']):
                raise Conflict('allocation ownership differs')
            return receipt
        released = []
        for model,resource in self.settings.models.items():
            try:
                row = store.snapshot(model)
            except Conflict:
                continue
            if row['state'] == 'retired':
                continue
            try:
                if row['pending_id'] is not None:
                    raise Conflict('handover pending')
                receipt = await exact(resource,row)
                if receipt.get('status') == 'active' or needs_settlement(receipt):
                    running = await asyncio.to_thread(manager.ollama.list_running)
                    if manager.models or not isinstance(running,list) or running:
                        raise Conflict('backend absence unconfirmed')
                    await asyncio.to_thread(manager.sc.release_lease,row['lease_id'])
                    receipt = await exact(resource,row)
                store.retire(model,receipt)
            except Exception as error:
                log.warning('Startup left %s residency %s for reconciliation: %s', model,
                            row['lease_id'], error if isinstance(error,Conflict) else type(error).__name__)
                continue
            released.append(row['lease_id'])
            log.info('Startup released %s residency %s (%sMB) left by a previous gateway',
                     model, row['lease_id'], row['memory_mb'])
        return released

    def bind_capacity_source(self,*,work_permitted=None):
        """Bind manager-owned observations/renewal before maintenance starts."""
        if self.capacity_source is not None or self.manager.residency_renewal is not None:
            raise RuntimeError('Capacity/residency owner already bound')
        from .inference_capacity_source import ManagerCapacitySource
        from .inference_residency import ResidencyStore
        from .inference_residency_worker import ResidencyRollover
        from .inference_residency_manager import ManagerResidency
        from .manager import LEASE_TTL
        initial = self.manager.initial_reservations
        actor = ResidencyRollover(ResidencyStore(self.settings.journal.parent/'residency.sqlite',
            registry_owner=initial.owner,service_id=initial.service),self.manager.sc)
        try:
            binding = ManagerResidency(self.manager,actor,self.settings.models,ttl_seconds=LEASE_TTL,
                                      require_size=bool(self.settings.reservation_mb))
            source = ManagerCapacitySource(self,binding,work_permitted=work_permitted)
        except BaseException:
            actor.close()
            raise
        self.residency_actor,self.capacity_source = actor,source
        self.manager.residency_renewal = binding
        self.manager.inference_capacity_source = source
        if self.settings.shared_queue:
            from .inference_shared_resource import ResourceHandoff
            self.shared_resource = ResourceHandoff(self)
        binding.on_change = lambda model:source.publish(self.settings.models[model],'unknown','residency_changed',model=model)
        previous = self.queue.supervisor.on_finished
        async def completed(grant,execution):
            from .inference_events import source_call
            if previous is not None:
                await previous(grant,execution)
            if not source.paused:
                source.invalidate()
                await source.refresh(await source_call(self.manager._collect_stats))
        self.queue.supervisor.on_finished = completed

    async def quiesce(self, *, _resource_handoff=False):
        """Drain known workers with no age deadline; report unresolved ownership.

        HTTP observers and durable submissions can continue. Canceling this
        wait does not cancel workers. Uncertain work is not called drained just
        because its Python task has ended; the caller must reconcile it before
        any manager shutdown/unload or legacy-mode rollback.
        """
        if not _resource_handoff:
            self._external_quiesce = True
        self.capacity.pause()
        handoff = getattr(self, 'shared_resource', None)
        if (not _resource_handoff and handoff is not None and handoff.task is not None
                and not handoff.task.done() and handoff.task is not asyncio.current_task()):
            await asyncio.shield(handoff.task)
        if self.capacity_source is not None:
            self.capacity_source.pause()
            await self.capacity_source.drain()
        binding = getattr(self.manager, 'residency_renewal', None)
        supervisor = self.queue.supervisor
        while tasks := tuple(task for task in
                (*supervisor.workers.values(), *supervisor.preparations) if not task.done()):
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
        counts = {row[0]:row[1] for row in supervisor.journal.db.execute(
            'SELECT state,COUNT(*) FROM executions GROUP BY state')}
        claims = len(self.queue.held_claims)
        manager_claims = sum(max(0,mm.in_flight) for mm in
                             getattr(self.manager,'models',{}).values())
        metadata_claims = len(getattr(binding,'claims',{}))
        if (binding is not None and not (counts.get('running',0) or counts.get('uncertain',0)
                or claims or manager_claims > metadata_claims)):
            # Only stop renewal AFTER known compute drains. Unknown compute or
            # untracked manager work must retain its physical reservation while
            # the operator reconciles it, however long that takes.
            binding.pause()
            await binding.drain()
            manager_claims = sum(max(0,mm.in_flight) for mm in self.manager.models.values())
        handoff_pending = (not _resource_handoff and handoff is not None and
                           handoff.snapshot()['phase'] != 'idle')
        return {'drained':not (counts.get('running', 0) or counts.get('uncertain', 0)
                              or claims or manager_claims or handoff_pending),
                'running':counts.get('running', 0), 'uncertain':counts.get('uncertain', 0),
                'queued':counts.get('queued', 0), 'held_claims':claims,
                'manager_claims':manager_claims}

    def resume(self):
        handoff = getattr(self, 'shared_resource', None)
        if handoff is not None and handoff.snapshot()['phase'] != 'idle':
            raise RuntimeError('Physical handoff requires reconciliation before resume')
        self._external_quiesce = False
        binding = getattr(self.manager,'residency_renewal',None)
        if binding is not None:
            binding.resume()
        self.capacity.resume()  # fresh capacity evidence still required
        if self.capacity_source is not None:
            self.capacity_source.resume()

    def close(self):
        """Only after quiescence and manager cleanup, never on uncertain work."""
        handoff = getattr(self, 'shared_resource', None)
        if handoff is not None and handoff.task is not None and not handoff.task.done():
            raise RuntimeError('Owned physical handoff must finish before closing its journal')
        if self.residency_actor is not None:
            self.residency_actor.close()
        self.queue.supervisor.close()
