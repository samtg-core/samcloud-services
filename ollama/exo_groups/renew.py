"""Renew exact group members in place; never infer execution permission.

Renewal lead times are explicit provider controls, not assembly/compute caps.
The caller schedules the returned continuous-clock wake on its existing loop.
No loop, model action, release, replacement or inference retry exists here.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from .clock import authority_clock


class RenewalUnavailable(RuntimeError):
    """No renewed authority established; retain ownership for reconciliation."""


@dataclass(frozen=True)
class MemberRenewal:
    rank: str
    lease_id: str
    deadline: float
    next_due: float | None
    at_ceiling: bool


@dataclass(frozen=True)
class RenewalReport:
    members: tuple[MemberRenewal, ...]

    @property
    def next_due(self):
        return min((item.next_due for item in self.members if item.next_due is not None), default=None)

    @property
    def next_event(self):
        events = [item.next_due for item in self.members if item.next_due is not None]
        events.extend(item.deadline for item in self.members if item.at_ceiling)
        return min(events, default=None)


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _instant(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError('aware lease timestamp required')
    return parsed


class GroupRenewals:
    def __init__(self, plan, plane, *, renew_before_s, clock=authority_clock):
        ranks = {member.rank for member in plan.members}
        if (plane.owner_id != plan.owner_id or set(renew_before_s) != ranks
                or any(not _finite(renew_before_s[m.rank])
                       or not 0 < renew_before_s[m.rank] < m.ttl_seconds for m in plan.members)):
            raise ValueError('explicit positive per-member renewal lead below its TTL required')
        self.plan, self.plane, self.clock = plan, plane, clock
        self.lead = dict(renew_before_s)

    def _group(self, group, group_id, generation):
        if (type(generation) is not int or generation < 1
                or group.get('id') != group_id or group.get('owner_id') != self.plan.owner_id
                or group.get('request_key') != self.plan.request_key
                or group.get('intent_sha256') != self.plan.intent_sha256
                or type(group.get('generation')) is not int or group['generation'] != generation
                or group.get('release_requested') is not False
                or group.get('phase') not in ('pending', 'acquiring', 'admitted', 'serving')
                or not isinstance(group.get('members'), list)
                or [row.get('resource_id') for row in group['members']]
                    != [member.resource_id for member in self.plan.members]):
            raise RenewalUnavailable('group renewal requires reconciliation')

    async def _observe(self, group_id, generation, member, lease_id):
        started = self.clock()
        projection = await self.plane.member(group_id, lease_id)
        received = self.clock()
        own = projection.get('member')
        if (not _finite(started) or not _finite(received) or started > received
                or projection.get('group_id') != group_id
                or projection.get('owner_id') != self.plan.owner_id
                or type(projection.get('generation')) is not int
                or projection['generation'] != generation
                or projection.get('release_requested') is not False
                or projection.get('phase') not in ('acquiring', 'admitted', 'serving')
                or not isinstance(own, dict) or own.get('resource_id') != member.resource_id
                or own.get('lease_id') != lease_id
                or own.get('cleanup_attested') is not False or own.get('settlement_recorded') is not False):
            raise RenewalUnavailable('member renewal requires reconciliation')
        lease = own.get('lease')
        if (not isinstance(lease, dict) or lease.get('id') != lease_id
                or lease.get('agent_id') != self.plan.owner_id
                or lease.get('resource_id') != member.resource_id
                or lease.get('exclusive') is not True or lease.get('priority') != 'normal'
                or not {'service_id', 'released_at', 'settled_at'} <= set(lease)
                or any(lease[key] is not None for key in ('service_id', 'released_at', 'settled_at'))
                or any(type(lease.get(key)) is not int or lease[key] != value for key, value in (
                    ('memory_mb', member.memory_mb), ('ttl_s', member.ttl_seconds),
                    ('max_total_s', member.max_total_s)))):
            raise RenewalUnavailable('exact member lease changed')
        if lease.get('status') == 'queued' and lease.get('expires_at') is None:
            return None  # Indefinite queue waits have no expiry to renew.
        if lease.get('status') != 'active':
            raise RenewalUnavailable('member lease is not active')
        observed = _instant(projection.get('observed_at'))
        granted, expiry = _instant(lease.get('granted_at')), _instant(lease.get('expires_at'))
        ceiling = granted + timedelta(seconds=member.max_total_s)
        if (not granted <= observed < expiry <= ceiling
                or (expiry - observed).total_seconds() > member.ttl_seconds):
            raise RenewalUnavailable('member lease is outside its declared authority')
        deadline = started + (expiry - observed).total_seconds()
        now = self.clock()
        if not _finite(now) or not received <= now < deadline:
            raise RenewalUnavailable('observed member authority elapsed')
        at_ceiling = expiry == ceiling
        return MemberRenewal(member.rank, lease_id, deadline,
            None if at_ceiling else deadline - self.lead[member.rank], at_ceiling)

    async def maintain(self, group, *, generation, current=lambda: None):
        """Fresh reads around each due renewal; a concurrent fence wins authority.

        The ordinary lease endpoint is not a group-generation CAS. If a fence
        races its POST, the lease can extend but the fresh group/member read
        refuses the old generation. No such response grants execution authority.
        """
        try:
            group_id = group['id']
            self._group(group, group_id, generation)
            result = []
            for member, linked in zip(self.plan.members, group['members']):
                lease_id = linked.get('lease_id')
                if lease_id is None:
                    if linked.get('status') != 'not_requested':
                        raise RenewalUnavailable('member link unavailable')
                    continue
                observation = await self._observe(group_id, generation, member, lease_id)
                current()
                if observation is None:
                    continue
                if observation.next_due is not None and self.clock() >= observation.next_due:
                    fresh = await self.plane.group(group_id)
                    current()
                    self._group(fresh, group_id, generation)
                    same = next(row for row in fresh['members'] if row['resource_id'] == member.resource_id)
                    if same.get('lease_id') != lease_id:
                        raise RenewalUnavailable('member lease identity changed')
                    await self.plane.renew(lease_id, member.ttl_seconds)
                    current()
                    # Never accept a renewal reply as proof of permission.
                    fresh = await self.plane.group(group_id)
                    current()
                    self._group(fresh, group_id, generation)
                    observation = await self._observe(group_id, generation, member, lease_id)
                    current()
                    if (observation is None or observation.next_due is not None
                            and self.clock() >= observation.next_due):
                        raise RenewalUnavailable('renewal did not establish a future wake')
                result.append(observation)
            # Slow later reads may consume the first member's remaining time.
            now = self.clock()
            if not _finite(now) or any(now >= item.deadline for item in result):
                raise RenewalUnavailable('member authority elapsed during renewal observation')
            current()
            return RenewalReport(tuple(result))
        except Exception:
            raise RenewalUnavailable('group member renewal requires reconciliation') from None
