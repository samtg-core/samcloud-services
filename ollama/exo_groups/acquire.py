"""Service-owned, event-driven assembly of one pinned EXO lease group.

The existing inference journal owns demand and the registry owns each physical
queue. This actor owns metadata requests across observer disconnects. It never
loads a model, starts a rank, submits inference or releases physical ownership.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from ..inference_arbiter import NotFound
from .intents import GroupIntents
from .plan import GroupPlan, bind_admitted_group, check_rank_manifest
from .staging import StagingUnavailable


@dataclass(frozen=True)
class AcquisitionReport:
    phase: str
    reason: str
    group_id: str | None = None
    group_phase: str | None = None


class GroupAcquisition:
    """Explicit provider; no assembly/compute timer or automatic retry loop.

    The existing event consumer calls events_reconciled after its cursor is
    current, on_event for hints, and events_lost on contact/cursor loss. A new
    demand or an event coalesces a metadata reconciliation. Event fields never
    grant authority. HTTP observers call wait(), which shields the owned task.
    """

    def __init__(self, intents: GroupIntents, plan: GroupPlan, manifest: bytes,
                 plane, authority_factory, *, notify=lambda _report: None,
                 wall_clock=lambda: datetime.now(timezone.utc), renewals=None,
                 renewal_alarm_factory=None, verify_staging=None):
        if not isinstance(intents, GroupIntents) or not isinstance(plan, GroupPlan):
            raise TypeError('existing journal and provider-owned plan required')
        if not isinstance(manifest, bytes) or plane.owner_id != plan.owner_id:
            raise ValueError('exact manifest and coordinator identity required')
        check_rank_manifest(plan, manifest)
        self.intents, self.plan, self.manifest = intents, plan, manifest
        self.plane, self.authority_factory = plane, authority_factory
        self.notify, self.wall_clock = notify, wall_clock
        if verify_staging is not None and not callable(verify_staging):
            raise TypeError('explicit staging verifier required')
        self.verify_staging = verify_staging
        self.authority = None
        self.group_snapshot = None
        self.ownership_uncertain = False
        self.report = AcquisitionReport('unknown', 'event_source_unconfirmed')
        self._task = None
        self._epoch = 0
        self._events_ready = False
        self._closed = False
        self._drained = False
        self.task_factory = asyncio.create_task
        self.renewals, self.renewal_report, self.renewal_alarm = renewals, None, None
        if renewals is not None:
            if renewals.plan != plan or renewals.plane is not plane:
                raise ValueError('renewals must name this exact coordinator plan and client')
            from .alarm import ContinuousAuthorityAlarm
            factory = renewal_alarm_factory or ContinuousAuthorityAlarm
            self.renewal_alarm = factory(asyncio.get_running_loop(), self.wake, self._renewal_failed)

    def _renewal_failed(self):
        self.events_lost()
        self._publish(AcquisitionReport('unknown', 'renewal_alarm_unavailable', self._known_group()))

    def _disarm_renewal(self):
        if self.renewal_alarm is not None:
            self.renewal_alarm.disarm()

    def _invalidate(self):
        if self.authority is not None:
            self.authority.invalidate()

    def _publish(self, report):
        self.report = report
        try:
            self.notify(report)
        except Exception:
            self._invalidate()
            self.report = AcquisitionReport('unknown', 'notification_failed', report.group_id)

    def _known_group(self):
        try:
            return self.intents.snapshot(self.plan.owner_id, self.plan.request_key)['group_id']
        except Exception:
            return self.report.group_id

    def note_demand(self, execution_id):
        if self._closed:
            raise RuntimeError('acquisition actor closed')
        self.intents.prepare(execution_id, self.plan, self.manifest)
        self.wake()

    def events_reconciled(self):
        if self._closed:
            raise RuntimeError('acquisition actor closed')
        self._events_ready = True
        self.wake()

    def events_lost(self):
        self.ownership_uncertain = True
        self._events_ready = False
        self._epoch += 1
        self._invalidate()
        self._disarm_renewal()
        self._publish(AcquisitionReport('unknown', 'event_source_unconfirmed', self.report.group_id))

    def on_event(self, event):
        if self._closed or not isinstance(event, Mapping):
            return False
        data, kind = event.get('data'), event.get('type')
        resources = {member.resource_id for member in self.plan.members}
        if (not isinstance(data, Mapping) or not isinstance(kind, str)
                or not kind.startswith(('resource.', 'lease.'))
                or data.get('resource_id') not in resources):
            return False
        if kind == 'resource.stats':
            # Utilisation telemetry is not an ownership/capacity mutation.
            # The feed coalesces it as liveness, like a keepalive.
            return False
        # Other holders on a required host can unblock our ordinary queued
        # member, so these hints are relevant before our lease id is known too.
        self.wake()
        return True

    def wake(self):
        if self._closed:
            return
        self._epoch += 1
        self._invalidate()  # Withdraw before scheduling, not after awaiting I/O.
        if not self._events_ready:
            return
        if self._task is None or self._task.done():
            task = self.task_factory(self._drive())
            self._task = task
            def finished(done):
                if self._task is done:
                    self._task = None
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished)

    def refresh(self):
        """Coalesce liveness hints without invalidating an in-flight read.

        A keepalive or unrelated event says the feed is alive, not that this
        group's authority changed. Starting another epoch for each such hint
        can starve a full host-file audit that takes longer than a keepalive.
        Exact resource/lease events and contact loss still invalidate at once.
        No hint extends the continuous-clock authority of an existing receipt.
        """
        if self._closed:
            return
        if self._task is None or self._task.done():
            self.wake()

    async def wait(self):
        task = self._task
        if task is not None:
            await asyncio.shield(task)
        return self.report

    def may_serve_requests(self):
        """Current event cursor and lease prerequisite; backend checks additional."""
        return bool(not self._closed and self._events_ready
            and self.report.phase == 'lease_serving' and self.authority is not None
            and self.authority.may_serve_requests())

    def may_start_ranks(self):
        return bool(not self._closed and self._events_ready
            and self.report.phase in ('admitted', 'lease_serving') and self.authority is not None
            and self.authority.may_start_ranks())

    def _current(self, epoch):
        if self._closed or not self._events_ready or epoch != self._epoch:
            raise RuntimeError('acquisition observation changed')

    async def _drive(self):
        while not self._closed and self._events_ready:
            epoch = self._epoch
            try:
                report = await self._reconcile(epoch)
                self._current(epoch)
                self._publish(report)
            except (Exception, asyncio.CancelledError) as error:
                self.ownership_uncertain = True
                self._invalidate()
                self._disarm_renewal()
                reason = 'staging_unconfirmed' if isinstance(error, StagingUnavailable) else 'reconciliation_required'
                self._publish(AcquisitionReport('unknown', reason, self._known_group()))
                if isinstance(error, asyncio.CancelledError):
                    raise
            # Only a new demand/event requests another read. An error or elapsed
            # time alone never retries even this idempotent metadata operation.
            if epoch == self._epoch:
                break

    async def _reconcile(self, epoch):
        try:
            saved = self.intents.snapshot(self.plan.owner_id, self.plan.request_key)
        except NotFound:
            return AcquisitionReport('idle', 'no_saved_demand')
        self.intents.bind_renewal_policy(self.plan.owner_id, self.plan.request_key,
            self.renewals.lead if self.renewals is not None else None)
        await self.plane.verify_owner()
        self._current(epoch)
        if saved['group_id'] is None:
            if not saved['create_started']:
                if self.intents.create_request(self.plan.owner_id, self.plan.request_key) is None:
                    return AcquisitionReport('idle', 'no_queued_demand')
                # All bytes must exist before the first physical request.
                await self._stage(epoch)
            request = self.intents.begin_create(self.plan.owner_id, self.plan.request_key)
            if request is None:
                return AcquisitionReport('idle', 'no_queued_demand')
            self.ownership_uncertain = True
            group = await self.plane.create(request)
            # Always persist a returned ownership identity, even if the event
            # feed was lost or the actor closed while this HTTP operation ran.
            saved = self.intents.record_group(self.plan.owner_id, self.plan.request_key, group)
        else:
            group = await self.plane.group(saved['group_id'])
            self.intents.record_group(self.plan.owner_id, self.plan.request_key, group)
        self._current(epoch)
        self.group_snapshot = copy.deepcopy(group)
        self.ownership_uncertain = False
        group_id = saved['group_id']
        if (group.get('phase') not in ('pending', 'acquiring', 'admitted', 'serving',
                'releasing', 'released', 'cancelled', 'uncertain')
                or type(group.get('release_requested')) is not bool):
            raise RuntimeError('group state requires reconciliation')
        if group['generation'] != saved['generation']:
            self._disarm_renewal()
            return AcquisitionReport('fenced', 'coordinator_generation_changed', group_id, group['phase'])
        if (self.intents.closing(self.plan.owner_id, self.plan.request_key)
                or group.get('release_requested')
                or group['phase'] in ('releasing', 'released', 'cancelled', 'uncertain')):
            self._disarm_renewal()
            return AcquisitionReport('draining', 'group_not_available', group_id, group['phase'])
        if group['phase'] in ('pending', 'acquiring'):
            await self._stage(epoch)  # Fresh files before requesting another member.
            self.ownership_uncertain = True
            group = await self.plane.advance(group_id, saved['generation'])
            self.intents.record_group(self.plan.owner_id, self.plan.request_key, group)
            self._current(epoch)
            self.group_snapshot = copy.deepcopy(group)
            self.ownership_uncertain = False
        if self.renewals is not None:
            self.renewal_report = await self.renewals.maintain(group,
                generation=saved['generation'], current=lambda: self._current(epoch))
            self._current(epoch)
            if self.renewal_report.next_event is None:
                self._disarm_renewal()
            else:
                # Arm before publishing any refreshed permission. This is a
                # declared lease-renewal wake, not an assembly or compute cap.
                self.renewal_alarm.arm(self.renewal_report.next_event)
        if group['phase'] not in ('admitted', 'serving'):
            return AcquisitionReport('waiting_members', 'ordinary_host_queue', group_id, group['phase'])
        await self._stage(epoch)
        leases = {}
        for member in group['members']:
            lease_id = member.get('lease_id')
            leases[lease_id] = await self.plane.lease(lease_id)
            self._current(epoch)
        fresh = await self.plane.group(group_id)
        self._current(epoch)
        self.group_snapshot = copy.deepcopy(fresh)
        self.ownership_uncertain = False
        launches = bind_admitted_group(self.plan, fresh, leases,
            expected_generation=saved['generation'], now=self.wall_clock())
        self.intents.bind_launches(self.plan.owner_id, self.plan.request_key, launches)
        if self.authority is None:
            self.authority = self.authority_factory(self.plan, launches)
        await self.authority.refresh()
        self._current(epoch)
        if not self.authority.may_start_ranks():
            return AcquisitionReport('unknown', 'fresh_member_authority_unavailable', group_id, fresh['phase'])
        phase = 'lease_serving' if self.authority.may_serve_requests() else 'admitted'
        return AcquisitionReport(phase, 'all_member_authority', group_id, fresh['phase'])

    async def _stage(self, epoch):
        if self.verify_staging is None:
            raise StagingUnavailable('host staging verifier unbound')
        try:
            proof = await self.verify_staging(self.plan, self.manifest)
        except Exception:
            raise StagingUnavailable('host staging observation unavailable') from None
        self._current(epoch)
        self.intents.record_staging(self.plan.owner_id, self.plan.request_key, proof)

    async def close(self):
        if self._drained:
            return
        self._closed = True
        self.events_lost()
        await self.wait()  # Drain metadata ownership; never cancel backend work.
        if self.authority is not None:
            await self.authority.close()
        if self.renewal_alarm is not None:
            self.renewal_alarm.close()
        await self.plane.close()
        self._drained = True
        # Physical ownership remains recorded until the separate observed
        # cleanup/settlement path acknowledges it. Closing is not release.
