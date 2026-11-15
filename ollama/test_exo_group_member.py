"""Pure host observation policy; no credential, service or process effects."""

import copy
import unittest
from dataclasses import replace

from . import test_exo_group_plan as fixtures
from .exo_groups.plan import BindingRefused
from .exo_groups.member import ExpectedMember, check_projection, event_requires_read


class MemberAuthorityControls(unittest.TestCase):
    def setUp(self):
        source = fixtures.GroupBindingControls()
        source.setUp()
        binding = source.bind()[0]
        self.expected = ExpectedMember.from_launch(source.plan, binding)
        self.plan = source.plan
        self.projection = {
            "group_id": binding.group_id,
            "owner_id": binding.owner_id,
            "generation": binding.generation,
            "phase": "admitted",
            "release_requested": False,
            "observed_at": "2026-11-14T10:05:00+00:00",
            "authoritative": True,
            "authority_expires_at": "2026-11-14T10:10:00+00:00",
            "member": {
                "resource_id": binding.resource_id,
                "lease_id": binding.lease_id,
                "cleanup_attested": False,
                "settlement_recorded": False,
                "lease": copy.deepcopy(source.leases[binding.lease_id]),
            },
        }

    def check(self, projection=None, **clock):
        return check_projection(
            self.expected,
            self.projection if projection is None else projection,
            **{
                "request_started_monotonic": 100.0,
                "received_monotonic": 101.0,
                "now_monotonic": 101.0,
                **clock,
            },
        )

    def test_explicit_authority_maps_to_monotonic_without_a_default_age_cap(self):
        result = self.check()
        self.assertEqual(result.state, "authorized")
        self.assertEqual(result.expires_at_monotonic, 400)
        self.assertFalse(result.allows_start(99))
        self.assertTrue(result.allows_start(399.9))
        self.assertFalse(result.allows_start(400))
        # A different host wall-clock zone does not change the lease interval.
        self.projection["observed_at"] = "2026-11-14T17:05:00+07:00"
        self.assertEqual(self.check().expires_at_monotonic, 400)
        # Renewing the exact same lease may extend the explicit authority;
        # the old binding's snapshot expiry is not a default work deadline.
        self.projection["member"]["lease"]["expires_at"] = "2026-11-14T10:15:00+00:00"
        self.projection["authority_expires_at"] = "2026-11-14T10:15:00+00:00"
        self.assertEqual(self.check().expires_at_monotonic, 700)

    def test_observation_delay_cannot_extend_the_lease(self):
        result = self.check(received_monotonic=200, now_monotonic=201)
        self.assertEqual(result.expires_at_monotonic, 400)
        self.assertFalse(
            self.check(received_monotonic=401, now_monotonic=401).allows_start(401)
        )
        self.assertEqual(
            self.check(received_monotonic=401, now_monotonic=401).state, "stop_required"
        )

    def test_other_member_expiry_caps_cached_authority(self):
        self.projection["authority_expires_at"] = "2026-11-14T10:05:02+00:00"
        result = self.check()
        self.assertEqual(result.expires_at_monotonic, 102)
        self.assertTrue(result.allows_start(101))
        self.assertFalse(result.allows_start(102))
        del self.projection["authority_expires_at"]
        self.assertEqual(self.check().state, "unknown")

    def test_unavailable_or_malformed_observations_grant_nothing(self):
        for value in (
            None,
            [],
            {},
            {**self.projection, "authoritative": 1},
            {**self.projection, "generation": True},
        ):
            with self.subTest(value=value):
                result = check_projection(
                    self.expected,
                    value,
                    request_started_monotonic=100,
                    received_monotonic=101,
                    now_monotonic=101,
                )
                self.assertFalse(result.allows_start(101))
        for clock in (
            {"received_monotonic": float("nan")},
            {"now_monotonic": float("inf")},
            {"received_monotonic": 99},
            {"now_monotonic": 100},
        ):
            with self.subTest(clock=clock):
                self.assertEqual(self.check(**clock).state, "unknown")

    def test_a_fence_never_adopts_the_new_generation(self):
        original = self.expected.binding
        self.projection["generation"] = 2
        self.assertEqual(self.check().state, "stop_required")
        self.assertEqual(self.expected.binding, original)
        self.assertEqual(self.expected.binding.generation, 1)

    def test_partial_release_or_uncertain_group_grants_nothing(self):
        for phase in (
            "pending",
            "acquiring",
            "releasing",
            "uncertain",
            "released",
            "cancelled",
        ):
            with self.subTest(phase=phase):
                self.assertEqual(
                    self.check({**self.projection, "phase": phase}).state,
                    "stop_required",
                )
        for change in ({"release_requested": True}, {"authoritative": False}):
            self.assertEqual(
                self.check({**self.projection, **change}).state, "stop_required"
            )

    def test_changed_exact_binding_or_envelope_never_starts(self):
        changes = {
            "id": "other-lease",
            "resource_id": "other/gpu",
            "agent_id": "user:99",
            "exclusive": 1,
            "priority": "high",
            "memory_mb": 999,
            "ttl_s": 601,
            "max_total_s": 600,
            "service_id": "other/service",
            "status": "expired",
            "released_at": "now",
            "settled_at": "now",
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                projection = copy.deepcopy(self.projection)
                projection["member"]["lease"][name] = value
                self.assertFalse(self.check(projection).allows_start(101))
        for key in ("group_id", "owner_id"):
            self.assertFalse(
                self.check({**self.projection, key: "other"}).allows_start(101)
            )
        for key in (
            "lease_id",
            "resource_id",
            "cleanup_attested",
            "settlement_recorded",
        ):
            projection = copy.deepcopy(self.projection)
            projection["member"][key] = (
                True if key.endswith(("attested", "recorded")) else "other"
            )
            self.assertFalse(self.check(projection).allows_start(101))

    def test_missing_nullable_fields_are_unknown_not_authority(self):
        for name in ("service_id", "released_at", "settled_at"):
            with self.subTest(name=name):
                projection = copy.deepcopy(self.projection)
                del projection["member"]["lease"][name]
                self.assertEqual(self.check(projection).state, "unknown")

    def test_bad_or_outside_envelope_clock_cannot_authorize(self):
        for stamp in (None, "bad", "2026-11-14T10:05:00", "2026-11-14T10:11:00+00:00"):
            self.assertFalse(
                self.check({**self.projection, "observed_at": stamp}).allows_start(101)
            )
        projection = copy.deepcopy(self.projection)
        projection["member"]["lease"]["expires_at"] = "2026-11-14T14:00:00+00:00"
        self.assertEqual(self.check(projection).state, "unknown")

    def test_events_wake_only_the_pinned_member_and_never_adopt_generation(self):
        binding = self.expected.binding
        event = {
            "type": "lease.group_changed",
            "target": binding.lease_id,
            "data": {
                "group_id": binding.group_id,
                "generation": 2,
                "resource_id": binding.resource_id,
                "lease_id": binding.lease_id,
            },
        }
        self.assertTrue(event_requires_read(self.expected, event))
        self.assertEqual(binding.generation, 1)
        for name, value in (
            ("group_id", "other"),
            ("resource_id", "other/gpu"),
            ("lease_id", "other"),
        ):
            changed = copy.deepcopy(event)
            changed["data"][name] = value
            self.assertFalse(event_requires_read(self.expected, changed))
        self.assertTrue(
            event_requires_read(self.expected, {**event, "type": "lease.renewed"})
        )
        self.assertFalse(
            event_requires_read(self.expected, {**event, "type": "lease_group.fenced"})
        )

    def test_inconsistent_launch_plan_is_refused_before_any_observation(self):
        for changed in (
            replace(self.expected.binding, resource_id="other/gpu"),
            replace(self.expected.binding, rank_manifest_sha256="b" * 64),
            replace(self.expected.binding, owner_id="user:99"),
        ):
            with self.assertRaises(BindingRefused):
                ExpectedMember.from_launch(self.plan, changed)


if __name__ == "__main__":
    unittest.main()
