"""Admission controls; no provider, production API, process or inference."""

import copy
from datetime import datetime, timezone
import hashlib
import json
import unittest

from .exo_groups.plan import (
    BindingRefused,
    GroupPlan,
    bind_admitted_group,
    check_rank_manifest,
)


class GroupBindingControls(unittest.TestCase):
    def setUp(self):
        self.manifest = json.dumps(
            {
                "model_id": "small/fixture",
                "revision": "a" * 40,
                "ranks": [
                    {"name": name, "layers": layers}
                    for name, layers in [
                        ("slice", [0, 7]),
                        ("wafer", [7, 12]),
                        ("obelisk", [12, 28]),
                    ]
                ],
            }
        ).encode()
        members = {
            "slice": {
                "rank": "slice",
                "resource_id": "claude-services-slice/gpu-0",
                "pipeline_rank": 0,
                "layers": [0, 7],
            },
            "wafer": {
                "rank": "wafer",
                "resource_id": "wafer-services/gpu-metal",
                "pipeline_rank": 1,
                "layers": [7, 12],
            },
            "obelisk": {
                "rank": "obelisk",
                "resource_id": "obelisk-services/gpu-0",
                "pipeline_rank": 2,
                "layers": [12, 28],
            },
        }
        self.raw = {
            "request_key": "qualification-fixture-v1",
            "owner_id": "user:58",
            "model_id": "small/fixture",
            "revision": "a" * 40,
            "rank_manifest_sha256": hashlib.sha256(self.manifest).hexdigest(),
            "members": [
                {
                    **members[name],
                    "memory_mb": 1000,
                    "ttl_seconds": 600,
                    "max_total_s": 7200,
                }
                for name in ("obelisk", "slice", "wafer")
            ],
        }
        self.plan = GroupPlan.parse(self.raw)
        self.group = {
            "id": "group-one",
            "generation": 1,
            "phase": "admitted",
            "release_requested": False,
            "owner_id": "user:58",
            "request_key": self.plan.request_key,
            "intent_sha256": self.plan.intent_sha256,
            "members": [],
        }
        self.leases = {}
        for ordinal, m in enumerate(self.plan.members):
            lid = "lease-" + m.rank
            self.group["members"].append(
                {
                    "ordinal": ordinal,
                    "resource_id": m.resource_id,
                    "lease_id": lid,
                    "status": "active",
                    "expires_at": "2026-11-14T10:10:00+00:00",
                    "cleanup_evidence": None,
                    "release_receipt": None,
                }
            )
            self.leases[lid] = {
                "id": lid,
                "resource_id": m.resource_id,
                "agent_id": "user:58",
                "status": "active",
                "exclusive": True,
                "priority": "normal",
                "memory_mb": 1000,
                "max_total_s": 7200,
                "ttl_s": 600,
                "service_id": None,
                "released_at": None,
                "settled_at": None,
                "granted_at": "2026-11-14T10:00:00+00:00",
                "expires_at": "2026-11-14T10:10:00+00:00",
            }
        self.now = datetime(2026, 11, 14, 10, 5, tzinfo=timezone.utc)

    def bind(self):
        return bind_admitted_group(
            self.plan, self.group, self.leases, expected_generation=1, now=self.now
        )

    def test_acquisition_order_does_not_change_pipeline_rank(self):
        bound = self.bind()
        self.assertEqual([b.rank for b in bound], ["slice", "wafer", "obelisk"])
        self.assertEqual(bound[0].lease_id, "lease-slice")
        self.assertEqual(bound[2].layers, (12, 28))
        check_rank_manifest(self.plan, self.manifest)

    def test_waiting_partial_unknown_and_releasing_never_authorize_launch(self):
        for phase in (
            "pending",
            "acquiring",
            "releasing",
            "released",
            "uncertain",
            "cancelled",
        ):
            with self.subTest(phase=phase):
                self.group["phase"] = phase
                with self.assertRaises(BindingRefused):
                    self.bind()

    def test_stale_generation_never_silently_rebinds(self):
        self.group["generation"] = 2
        with self.assertRaisesRegex(BindingRefused, "generation changed"):
            self.bind()

    def test_release_request_blocks_even_an_active_lease(self):
        self.group["release_requested"] = True
        with self.assertRaises(BindingRefused):
            self.bind()

    def test_each_nonactive_member_blocks_the_whole_ring(self):
        for state in (
            "queued",
            "not_requested",
            "missing",
            "expired",
            "released",
            "cancelled",
        ):
            with self.subTest(state=state):
                self.group["members"][1]["status"] = state
                with self.assertRaises(BindingRefused):
                    self.bind()

    def test_wrong_owner_resource_envelope_priority_and_shared_lease_refuse(self):
        original = copy.deepcopy(self.leases["lease-slice"])
        for field, value in [
            ("agent_id", "user:79"),
            ("resource_id", "other/gpu-0"),
            ("exclusive", False),
            ("priority", "high"),
            ("memory_mb", 999),
            ("ttl_s", 599),
            ("max_total_s", 3600),
            ("service_id", "other"),
            ("settled_at", "2026-11-14T10:01:00+00:00"),
        ]:
            with self.subTest(field=field):
                self.leases["lease-slice"] = {**original, field: value}
                with self.assertRaises(BindingRefused):
                    self.bind()

    def test_missing_extra_or_substituted_receipt_refuses(self):
        original = copy.deepcopy(self.leases)
        del self.leases["lease-slice"]
        with self.assertRaises(BindingRefused):
            self.bind()
        self.leases = copy.deepcopy(original)
        self.leases["extra"] = original["lease-slice"]
        with self.assertRaises(BindingRefused):
            self.bind()
        self.leases = copy.deepcopy(original)
        self.leases["lease-slice"]["id"] = "lease-wafer"
        with self.assertRaises(BindingRefused):
            self.bind()

    def test_expired_or_naive_timestamp_never_means_cleaned_up(self):
        for expiry in (
            "2026-11-14T10:05:00+00:00",
            "2026-11-14T10:04:59+00:00",
            "2026-11-14T10:10:00",
            "not-a-date",
        ):
            with self.subTest(expiry=expiry):
                self.leases["lease-slice"]["expires_at"] = expiry
                self.group["members"][1]["expires_at"] = expiry
                with self.assertRaises(BindingRefused):
                    self.bind()

    def test_rollover_between_reads_requires_fresh_observations(self):
        self.leases["lease-slice"]["expires_at"] = "2026-11-14T10:15:00+00:00"
        with self.assertRaisesRegex(BindingRefused, "reads raced"):
            self.bind()

    def test_changed_intent_and_member_order_refuse(self):
        self.group["intent_sha256"] = "f" * 64
        with self.assertRaises(BindingRefused):
            self.bind()
        self.group["intent_sha256"] = self.plan.intent_sha256
        self.group["members"].reverse()
        with self.assertRaises(BindingRefused):
            self.bind()

    def test_manifest_and_placement_are_both_bound(self):
        with self.assertRaises(BindingRefused):
            check_rank_manifest(self.plan, self.manifest + b" ")
        raw = copy.deepcopy(self.raw)
        raw["members"][1]["layers"] = [0, 6]
        raw["members"][2]["layers"] = [6, 12]
        other = GroupPlan.parse(raw)
        with self.assertRaisesRegex(BindingRefused, "declared pipeline"):
            check_rank_manifest(other, self.manifest)

    def test_all_controls_are_explicit_no_deadlines_or_priority_fields(self):
        for key in ("ttl_seconds", "max_total_s"):
            raw = copy.deepcopy(self.raw)
            del raw["members"][0][key]
            with self.assertRaises(BindingRefused):
                GroupPlan.parse(raw)
        for key in ("assembly_deadline", "idle_grace", "priority"):
            raw = copy.deepcopy(self.raw)
            raw[key] = 1
            with self.assertRaises(BindingRefused):
                GroupPlan.parse(raw)
        self.assertEqual(set(self.plan.request()), {"request_key", "members"})

    def test_duplicate_rank_gap_and_legacy_resource_refuse(self):
        for mutate in (
            lambda r: r["members"][0].update(
                resource_id="claude-services-slice/exo-pool"
            ),
            lambda r: r["members"][0].update(pipeline_rank=0),
            lambda r: r["members"][2].update(layers=[8, 12]),
            lambda r: r["members"][2].update(ttl_seconds=True),
        ):
            raw = copy.deepcopy(self.raw)
            mutate(raw)
            with self.assertRaises(BindingRefused):
                GroupPlan.parse(raw)


if __name__ == "__main__":
    unittest.main()
