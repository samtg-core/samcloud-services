"""Service-owned metadata request lifetime for residency handovers (offline).

No polling/retry/compute deadline, model mutation, unload or legacy fallback.
The caller supplies current registry evidence and requests a handover explicitly.
Capacity binding must consult has_current_receipt plus its physical working-set
policy; on_state is a synchronous diagnostic/event hint, never sole permission.
"""
import asyncio
from dataclasses import dataclass
import inspect
import logging

from .inference_arbiter import Conflict
from .inference_residency import RolloverIntent


@dataclass(frozen=True)
class RunningRollover:
    operation: RolloverIntent
    task: asyncio.Task


class ResidencyRollover:
    def __init__(self, store, client, *, on_state=None):
        self.store, self.client, self.on_state = store, client, on_state
        self.workers = {}
        self.closed = False

    def has_current_receipt(self, allocation):
        if self.closed:
            return False
        snapshot = self.store.snapshot(allocation)
        return (snapshot['state'] == 'covered' and snapshot['fresh_receipt'] and
                snapshot['receipt_unexpired'] and snapshot['pending_id'] is None)

    def _notify(self, allocation, outcome):
        if self.on_state is None:
            return
        try:
            result = self.on_state({'outcome':outcome, **self.store.snapshot(allocation)})
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError('residency event callback must be synchronous')
        except Exception:
            # A diagnostics failure cannot undo a committed registry receipt.
            # No response body or exception text is written into incidental logs.
            logging.getLogger(__name__).warning('Residency event callback incomplete')

    def begin(self, allocation, *, ttl_seconds):
        if self.closed:
            raise RuntimeError('residency worker is closed')
        existing = self.workers.get(allocation)
        if existing is not None and not existing.task.done():
            if (type(ttl_seconds) is not int or ttl_seconds != existing.operation.ttl_seconds or
                    self.store.snapshot(allocation)['pending_id'] != existing.operation.id):
                raise Conflict('different handover is still running')
            return existing
        operation = self.store.prepare(allocation,ttl_seconds=ttl_seconds)
        # Intent commit withdraws the exact local receipt BEFORE this task can
        # send. Existing compute is untouched; fresh dispatch must recheck it.
        self._notify(allocation,'handover_pending')
        run = RunningRollover(operation,asyncio.create_task(self._drive(operation)))
        self.workers[allocation] = run
        return run

    async def _drive(self, operation):
        try:
            # Only HTTP runs on the worker thread. SQLite receipt application
            # stays on its owning event-loop thread.
            response = await asyncio.to_thread(self.client.rollover_lease,
                operation.predecessor,key=operation.id,ttl_seconds=operation.ttl_seconds)
            result = self.store.accept(operation,response)
        except asyncio.CancelledError:
            self.store.forget_confirmation(operation.allocation)
            self._notify(operation.allocation,'unknown')
            # Process/service cancellation may leave the network call running.
            # Persisted intent remains; never infer its termination or release.
            raise
        except Exception:
            self.store.forget_confirmation(operation.allocation)
            self._notify(operation.allocation,'unknown')
            return {'outcome':'unknown', **self.store.snapshot(operation.allocation)}
        outcome = ('unchanged' if not result['changed'] else
                   'covered' if self.has_current_receipt(operation.allocation) else 'uncovered')
        self._notify(operation.allocation,outcome)
        return {'outcome':outcome, **result}

    async def observe(self, run):
        """Cancel this wait without canceling the service-owned request."""
        return await asyncio.shield(run.task)

    async def drain(self):
        while tasks := tuple(run.task for run in self.workers.values() if not run.task.done()):
            await asyncio.shield(asyncio.gather(*tasks,return_exceptions=True))

    def close(self):
        if any(not run.task.done() for run in self.workers.values()):
            raise RuntimeError('cannot close residency store during a live handover')
        self.closed = True
        self.store.close()
