"""Service-owned dispatch lifetime, independent of HTTP observers.

Not installed in the gateway yet. The transport adapter must return only after
backend termination, or raise BackendEnded for a confirmed terminal failure.
Generic transport exceptions/cancellation are UNKNOWN, never free capacity.
"""
import asyncio
import fcntl
import json
import logging
import os
from pathlib import Path
import time

from .inference_arbiter import Arbiter, encoded, CapacityDeferred


class BackendEnded(Exception):
    """Adapter has affirmative evidence of terminal backend failure.

    A socket timeout/disconnect is NOT such evidence. Store only a sanitized
    client-facing result; exception text may contain credentials or prompts.
    """
    def __init__(self, result):
        self.result = result
        super().__init__('backend ended')


class Supervisor:
    def __init__(self, path, execute, permitted, on_finished=None, *, stint_policy=None, journal_factory=None):
        """One scheduler per journal; execute/permitted are trusted service code.

        execute(grant, request) is async and must include the model's in_flight
        claim for the actual worker lifetime. permitted(resource) is a current
        cached capacity/work-mode check (no blocking I/O); UNKNOWN returns False. Resource mapping
        comes from service configuration, not the caller's request body.

        journal_factory is an explicit trusted internal constructor with the
        same (path, stint_policy=...) contract as Arbiter. It is never selected
        by request data. Default construction and legacy behavior are unchanged.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = os.open(str(path) + '.supervisor', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if os.fstat(self._lock).st_mode & 0o077:
                raise PermissionError('supervisor lock must be private')
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.journal = (Arbiter if journal_factory is None else journal_factory)(path, stint_policy=stint_policy)
            # Owning the local scheduler lock establishes the former scheduler
            # is gone, NOT that its remote/backend compute is gone.
            self.journal.quarantine_running()
        except BaseException:
            if hasattr(self, 'journal'):
                self.journal.close()
            os.close(self._lock)
            raise
        self.execute = execute
        self.permitted = permitted
        self.permitted_model = None
        self.on_finished = on_finished
        self.on_job_boundary = None
        # Synchronous, loop-local hint before the dispatch gate. A qualified
        # controller may withdraw permission and register coalesced work via
        # prepare(); the hint itself is never physical admission.
        self.on_resource_demand = None
        self.workers = {}
        self.preparations = set()
        self.wakeups = {}

    def prepare(self, operation):
        """Own a captured-input handoff even when its HTTP waiter disconnects.

        No backend dispatch is allowed before the durable execution commit.
        Shutdown must drain these tasks as well as generation workers.
        """
        task = asyncio.create_task(operation)
        self.preparations.add(task)
        def finished(done):
            self.preparations.discard(done)
            if not done.cancelled():
                done.exception()  # consume unobserved errors; do not log input detail
        task.add_done_callback(finished)
        return task

    def pump(self, resource):
        """Run after enqueue/release, and when capacity becomes available.

        No await between the committed dispatch intent and task registration.
        An HTTP handler only observes this task; it never owns/cancels it.
        """
        previous = self.wakeups.pop(resource, None)
        if previous is not None:
            previous.cancel()
        if self.on_resource_demand is not None:
            try:
                self.on_resource_demand(resource)
            except Exception:
                logging.getLogger(__name__).warning('Resource demand observation incomplete')
                return None
        if not self.permitted(resource):
            return None
        item = self.journal.dispatch_next(resource,permitted_model=self.permitted_model)
        if item is None:
            deadline = self.journal.next_handoff_at(resource)
            if deadline is not None and deadline > time.time():
                self.wakeups[resource] = asyncio.get_running_loop().call_later(
                    max(0, deadline - time.time()), self._wake, resource)
            return None
        execution_id = item['execution']['id']
        task = asyncio.create_task(self._drive(item))
        self.workers[execution_id] = task
        def finished(done):
            # A fresh capacity event can dispatch the same saved execution
            # before this older task's callback runs. Never untrack its heir.
            if self.workers.get(execution_id) is done:
                self.workers.pop(execution_id, None)
        task.add_done_callback(finished)
        return execution_id

    def _wake(self, resource):
        self.wakeups.pop(resource, None)
        self.pump(resource)

    def capacity_available(self, resource):
        """After fresh trusted capacity evidence, wake saved work without a caller."""
        if self.permitted(resource):
            self.journal.capacity_available(resource)
        # Even unavailable capacity can require admission preparation or
        # cleanup of an ENDING physical root. The pump still gates dispatch.
        return self.pump(resource)

    async def _drive(self, item):
        execution_id = item['execution']['id']
        try:
            failed = False
            try:
                result = await self.execute(item['grant'], json.loads(item['execution']['request']))
            except BackendEnded as exc:
                result, failed = exc.result, True
            # Invalid adapter output must not leave observers waiting forever
            # on a 'running' record after the supervisor task has ended.
            encoded(result)
        except CapacityDeferred as wait:
            try:
                self.journal.defer_capacity(execution_id,wait.reason)
            except Exception:
                self.journal.uncertain(execution_id)
            return
        except (Exception, asyncio.CancelledError):
            # Includes uncertain delivery, broken transport, and shutdown.
            # Do not cache raw exceptions, retry inference, or release a grant.
            self.journal.uncertain(execution_id)
            return
        else:
            self.journal.finish(execution_id, result, failed=failed)
        resource = item['grant']['resource']
        if item['execution'].get('job_id') and not self.journal.owns_job(resource):
            await self._job_boundaries([resource])
        else:
            self.pump(resource)
        if self.on_finished is not None:
            try:
                await self.on_finished(item['grant'],execution_id)
            except (Exception,asyncio.CancelledError):
                # The saved backend result is already authoritative. Failed
                # housekeeping must not relabel it uncertain or strand the queue.
                logging.getLogger(__name__).warning('Saved execution cleanup incomplete')

    async def _job_boundaries(self, resources):
        for resource in resources:
            try:
                if self.on_job_boundary is not None:
                    await self.on_job_boundary(resource)
            except (Exception, asyncio.CancelledError):
                # Admission remains withdrawn by the owning capacity source.
                # A fresh authoritative observation can retry reconciliation.
                logging.getLogger(__name__).warning('Requester job boundary reconciliation incomplete')
                continue
            self.pump(resource)

    def end_job(self, owner, key, *, finish=True):
        rows = self.journal.end_job(owner, key, finish=finish)
        resources = [row['resource'] for row in rows if row['state'] != 'owned']
        if resources:
            # Registered before returning the logical receipt. HTTP observer
            # cancellation cannot cancel this service-owned reconciliation.
            self.prepare(self._job_boundaries(resources))
        if self.on_resource_demand is not None:
            for row in rows:
                if row['state'] == 'owned':
                    # Root mode retains this logical slot until physical end
                    # is acknowledged. Waiting for !owns_job would deadlock.
                    self.pump(row['resource'])
        return rows

    def release(self, grant_id, owner):
        resource = self.journal.get(grant_id, owner)['resource']
        owned = self.journal.owns_job(resource)
        grant = self.journal.request_release(grant_id, owner)
        if owned and not self.journal.owns_job(resource):
            self.prepare(self._job_boundaries([resource]))
        else:
            self.pump(resource)
        return grant

    def yield_grant(self, grant_id, owner):
        grant = self.journal.request_yield(grant_id, owner)
        self.pump(grant['resource'])
        return self.journal.get(grant_id, owner)

    def yield_by_key(self, owner, grant_key, expected_call_key):
        grant = self.journal.yield_by_key(owner, grant_key, expected_call_key)
        self.pump(grant['resource'])
        return self.journal.get(grant['id'], owner)

    async def observe(self, execution_id, owner, interval=0.05):
        """Wait for a saved state. Cancelling an observer does not cancel work.

        HTTP may use short polls instead; no inference-age timeout is implied.
        Uncertain is returned explicitly for reconciliation, not called failure.
        """
        while True:
            row = self.journal.execution(execution_id, owner)
            if row['state'] not in ('queued', 'running'):
                return row
            await asyncio.sleep(interval)

    def close(self):
        if any(not task.done() for task in (*self.workers.values(),*self.preparations)):
            raise RuntimeError('cannot close supervisor with live backend work')
        for handle in self.wakeups.values():
            handle.cancel()
        self.wakeups.clear()
        self.journal.close()
        os.close(self._lock)
