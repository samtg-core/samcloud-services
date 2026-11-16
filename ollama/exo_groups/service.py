"""Trusted gateway startup and ownership of the configured registry streams.

No profile discovery, HTTP activation, implicit reconnect, rank launch or lease
operation. The launcher's configure function binds exact prepared provider/host
objects to the existing QueueRuntime, before maintenance or public serving.
"""

import asyncio
from dataclasses import dataclass
import logging

from ..inference_bootstrap import QueueStartup
from .install import EXOInstallation
from .rank_runtime import HostRankRuntime


class ServiceFeedUnavailable(RuntimeError):
    pass


class EXOService:
    """One lifecycle owner, with distinct host and coordinator credentials.

    Subscription tasks are observers, not Supervisor preparations: putting a
    permanent stream in preparations would deadlock ordinary quiescence. A feed
    ending withdraws its participants' authority and reports once. Only an
    explicit call on this same owner may reconnect; it never recreates actors,
    changes an attempt, starts ranks or resubmits inference.
    """

    def __init__(self, runtime, *, deployment=None):
        from .deployment import HostDeployment
        installation = getattr(runtime, 'exo', None)
        host = getattr(runtime, 'host_rank', None)
        if (getattr(runtime, 'exo_service', None) is not None
                or installation is None and host is None and deployment is None
                or deployment is not None and not isinstance(deployment, HostDeployment)
                or installation is not None and (
                    not isinstance(installation, EXOInstallation)
                    or installation.runtime is not runtime or installation.closed)
                or host is not None and (
                    not isinstance(host, HostRankRuntime) or host.closed
                    or getattr(host, '_queue_runtime', None) is not runtime)):
            raise ValueError('exact live provider/host bindings on this runtime required')
        self.runtime = runtime
        self.roles = {}
        if installation is not None:
            self.roles['coordinator'] = installation
        if host is not None:
            self.roles['host'] = host
        self.tasks = {}
        self.commands = []
        self.failures = {}
        self.started = self.closed = self.closing = False
        self.draining = False
        self._close_lock = asyncio.Lock()
        self.control = None
        if deployment is not None:
            from .host_control import HostControl
            self.control = HostControl(runtime, deployment, self)

    def start(self):
        if (self.started or self.closed or self.closing
                or getattr(self.runtime, 'exo_service', None) is not None):
            raise RuntimeError('fresh configured EXO service owner required')
        # Claim the owner before scheduling any coroutine. No feed is included
        # in Supervisor.preparations and no physical effect occurs here.
        self.runtime.exo_service = self
        self.started = True
        for role in self.roles:
            self._subscribe(role)
        return self

    def attach_host(self, owner):
        """Trusted launcher/controller binding, never an HTTP prepare operation."""
        if (not self.started or self.closed or self.closing or self.draining
                or not isinstance(owner, HostRankRuntime) or owner.closed
                or getattr(self.runtime, 'host_rank', None) is not owner
                or getattr(owner, '_queue_runtime', None) is not self.runtime
                or 'host' in self.roles and self.roles['host'] is not owner):
            raise RuntimeError('exact newly registered host owner required')
        if 'host' not in self.roles:
            self.roles['host'] = owner
            self._subscribe('host')
        return self.tasks['host']

    def own_host_commands(self, commands):
        from .host_http import HostCommands
        installation = self.roles.get('coordinator')
        if (not isinstance(commands, HostCommands) or self.closed or self.closing or self.draining
                or installation is None or installation.closed
                or commands.actor not in [p.actor for p in installation.providers.values()]
                or commands.journal is not self.runtime.queue.supervisor.journal
                or commands.actor.task_factory != self.runtime.queue.supervisor.prepare):
            raise ValueError('host commands must belong to this installed queue coordinator')
        self.commands.append(commands)

    def _subscribe(self, role):
        task = asyncio.create_task(self._consume(role))
        self.tasks[role] = task

        def observed(done):
            if not done.cancelled():
                done.exception()  # _consume has already journaled/logged failure.

        task.add_done_callback(observed)
        return task

    def reconnect(self, role):
        """Trusted owner action, never called by failure or elapsed time."""
        if (not self.started or self.closed or self.closing or role not in self.roles
                or self.tasks[role].done() is False):
            raise RuntimeError('same stopped EXO subscription required')
        return self._subscribe(role)

    def _report_failure(self, role, error):
        # No exception text/args, remote response, credentials or inference body.
        self.failures[role] = type(error).__name__
        log = logging.getLogger(__name__)
        log.warning('EXO %s event subscription unavailable; explicit reconciliation required', role)
        try:
            journal = self.runtime.queue.supervisor.journal
            for resource in sorted(set(self.runtime.settings.models.values())):
                journal.capacity_event(resource, None, 'unknown', 'exo_' + role + '_feed_unavailable')
        except Exception:
            log.error('EXO event subscription failure journal unavailable')

    async def _consume(self, role):
        try:
            await self.roles[role].consume_events_once()
            # A stream ending normally is also loss of observation.
            raise ServiceFeedUnavailable('EXO event subscription ended')
        except asyncio.CancelledError as error:
            if not self.closing:
                self._report_failure(role, error)
            raise
        except Exception as error:
            self._report_failure(role, error)
            raise ServiceFeedUnavailable('EXO event subscription requires reconciliation') from None

    async def wait(self, role):
        """Observe this subscription without giving its HTTP waiter ownership."""
        if role not in self.tasks:
            raise ValueError('configured EXO subscription required')
        return await asyncio.shield(self.tasks[role])

    def snapshot(self):
        # Diagnostic state only. A live task is NOT an authority observation.
        return {'closed': self.closed, 'closing': self.closing, 'subscriptions': {
            role: {'observing': not task.done(), 'last_failure': self.failures.get(role)}
            for role, task in self.tasks.items()}}

    async def close(self):
        async with self._close_lock:
            if self.closed:
                return
            if not self.started:
                raise RuntimeError('EXO service has not been started')
            self.draining = True
            if self.control is not None:
                self.control.pause()
                await self.control.drain()
            for commands in self.commands:
                commands.pause()
            for commands in self.commands:
                await commands.drain()
            # Do not cancel observers, explicit authority alarms or child owners
            # as a shortcut through uncertain work. Component close methods
            # verify cleanup/exact settlement and retain ownership on refusal.
            host = self.roles.get('host')
            installation = self.roles.get('coordinator')
            if host is not None and not host.closed and (
                    host.actor.receipt is None or host.actor.phase != 'awaiting_settlement'):
                raise RuntimeError('host cleanup must be observed before stopping its subscription')
            if installation is not None and not installation.closed:
                await self.runtime.close_exo()
            self.closing = True
            for task in self.tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
            if host is not None and not host.closed:
                await self.runtime.close_exo_rank()
            for commands in self.commands:
                await commands.client.close()
            self.closed = True


@dataclass(frozen=True)
class EXOQueueStartup(QueueStartup):
    """Trusted Python composition, deliberately absent from QueueSettings JSON.

    configure(runtime) must bind exact provider/host objects and return their
    EXOService. It supplies real own-credential callbacks and pinned runtime
    verifiers. There is no default factory or import-by-name/profile fallback.
    """
    configure: object = None

    async def start(self, service):
        if not callable(self.configure):
            raise ValueError('explicit trusted EXO composition required')
        runtime = await super().start(service)
        composition = await self.configure(runtime)
        if not isinstance(composition, EXOService) or composition.runtime is not runtime:
            raise ValueError('configured EXO service must own this exact runtime')
        composition.start()
        return runtime
