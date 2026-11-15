"""Real EXO wire/loader controls, no runtime, sockets or lease acquisition."""

import copy
from dataclasses import replace
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from exo.api.types.api import CreateInstanceParams
from exo.shared.types.worker.instances import BoundInstance, MlxRingInstance
from exo.utils.leased_runtime import LeasedRuntimeProfile

from ollama.exo_groups.prepare import RankEndpoint, build_explicit_instance, instance_digest
from ollama.exo_groups.plan import BindingRefused, GroupPlan
from ollama.exo_groups.prepare import canonical, digest
from ollama.exo_groups.instance import InstanceContract


class ExplicitInstanceControls(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        )
        self.network.start()
        self.addCleanup(self.network.stop)
        self.parent = {
            "model_id": "qualification/qwen3",
            "revision": "a" * 40,
            "config": {
                "model_type": "qwen3",
                "num_hidden_layers": 28,
                "hidden_size": 1024,
                "num_key_value_heads": 8,
                "max_position_embeddings": 32768,
            },
            "weight_files": [
                {"name": "model.safetensors", "bytes": 1024, "sha256": "b" * 64}
            ],
        }
        self.ranks = [("slice", [0, 7]), ("wafer", [7, 12]), ("obelisk", [12, 28])]
        self.manifest = {
            "model_id": self.parent["model_id"],
            "revision": self.parent["revision"],
            "parent_manifest_sha256": digest(self.parent),
            "config_sha256": digest(self.parent["config"]),
            "ranks": [{"name": rank, "layers": layers} for rank, layers in self.ranks],
        }
        resources = {
            "slice": "claude-services-slice/gpu-0",
            "wafer": "wafer-services/gpu-metal",
            "obelisk": "obelisk-services/gpu-0",
        }
        # Acquisition order deliberately differs from pipeline order.
        self.raw = {
            "request_key": "explicit-placement-control",
            "owner_id": "user:58",
            "model_id": self.parent["model_id"],
            "revision": self.parent["revision"],
            "rank_manifest_sha256": digest(self.manifest),
            "members": [
                {
                    "rank": self.ranks[i][0],
                    "resource_id": resources[self.ranks[i][0]],
                    "pipeline_rank": i,
                    "layers": self.ranks[i][1],
                    "memory_mb": 4096,
                    "ttl_seconds": 600,
                    "max_total_s": 7200,
                }
                for i in (2, 0, 1)
            ],
        }
        self.plan = GroupPlan.parse(self.raw)
        self.endpoints = tuple(
            RankEndpoint(rank, str(i + 1) * 32, f"192.168.50.{i + 11}")
            for i, (rank, _) in enumerate(self.ranks)
        )
        self.options = {
            "group_id": "group_" + "c" * 24,
            "generation": 1,
            "ring_port": 52518,
        }

    def build(self, **changes):
        return build_explicit_instance(
            self.plan,
            canonical(self.manifest),
            canonical(self.parent),
            self.endpoints,
            **(self.options | changes),
        )

    def parsed(self):
        result = CreateInstanceParams.model_validate_json(canonical(self.build()))
        self.assertIsInstance(result.instance, MlxRingInstance)
        return result.instance

    def test_provider_contract_uses_actual_aliases_and_same_placement(self):
        from exo.shared.types.state import State
        from exo.shared.types.worker.runners import RunnerReady
        contract = InstanceContract.prepare(
            self.plan, canonical(self.manifest), canonical(self.parent),
            self.endpoints, runtime_sha256="e" * 64, **self.options,
        )
        instance = CreateInstanceParams.model_validate(contract.create_request()).instance
        state = State(
            instances={instance.instance_id: instance},
            runners={runner: RunnerReady() for runner in instance.shard_assignments.runner_to_shard},
        ).model_dump(mode="json", by_alias=True)
        self.assertEqual(state["instances"][instance.instance_id],
                         __import__("json").loads(contract.state_instance_json))
        self.assertEqual(contract.create_request(), self.build())
        self.assertEqual(contract.instance_id, instance.instance_id)

    def test_committed_gateway_fixture_matches_actual_state_types(self):
        import json
        from exo.shared.types.state import State
        from exo.shared.types.worker.runners import RunnerReady
        fixture = json.loads((Path(__file__).parent / "fixtures/exo-instance-v1.json").read_text())
        instance = CreateInstanceParams.model_validate(fixture["request"]).instance
        from exo.shared.types.worker.downloads import DownloadCompleted
        state = State(
            instances={instance.instance_id: instance},
            runners={runner: RunnerReady() for runner in instance.shard_assignments.runner_to_shard},
            downloads={node: [DownloadCompleted(
                node_id=node, shard_metadata=instance.shard_assignments.runner_to_shard[runner],
                model_directory="/private/ranks/" + node,
                total=instance.shard_assignments.runner_to_shard[runner].model_card.storage_size,
                read_only=True,
            )] for node, runner in instance.shard_assignments.node_to_runner.items()},
        ).model_dump(mode="json", by_alias=True)
        self.assertEqual(fixture["state"], state)

    def test_real_tagged_request_preserves_unequal_assignments(self):
        wire = self.build()
        self.assertIn("MlxRingInstance", wire["instance"])
        actual = self.parsed()
        self.assertEqual(actual.ephemeral_port, 52518)
        for endpoint, (rank, layers) in zip(self.endpoints, self.ranks):
            runner = actual.shard_assignments.node_to_runner[endpoint.node_id]
            bound = BoundInstance(
                instance=actual, bound_runner_id=runner, bound_node_id=endpoint.node_id
            )
            self.assertEqual(
                [bound.bound_shard.start_layer, bound.bound_shard.end_layer], layers
            )
            self.assertEqual(
                bound.bound_shard.device_rank, [r[0] for r in self.ranks].index(rank)
            )
            self.assertEqual(bound.bound_shard.world_size, 3)
            self.assertEqual(bound.bound_shard.n_layers, 28)
            self.assertIsNone(bound.bound_shard.should_timeout)

    def test_every_actual_ring_guard_accepts_the_same_rank_order(self):
        instance = self.parsed()
        peers = tuple(endpoint.ip for endpoint in self.endpoints)
        for i, endpoint in enumerate(self.endpoints):
            profile = LeasedRuntimeProfile(
                self.options["group_id"],
                1,
                endpoint.rank,
                self.plan.rank_manifest_sha256,
                Path("/private/ranks"),
                Path("/private/state"),
                peers,
                Path("/private/key"),
                52515,
                52514,
                52516,
                52518,
            )
            hosts = [
                (host.ip, host.port)
                for host in instance.hosts_by_node[endpoint.node_id]
            ]
            self.assertEqual(
                profile.ring_hosts(
                    hosts, rank=i, instance_port=instance.ephemeral_port
                ),
                [f"{peer}:52518" for peer in peers],
            )
            with self.assertRaises(ValueError):
                profile.ring_hosts(
                    list(reversed(hosts)), rank=i, instance_port=instance.ephemeral_port
                )

    def test_card_matches_pinned_parent_and_does_not_enable_remote_code(self):
        card = next(
            iter(self.parsed().shard_assignments.runner_to_shard.values())
        ).model_card
        self.assertEqual(card.storage_size.in_bytes, 1024)
        self.assertEqual(card.hidden_size, 1024)
        self.assertEqual(card.num_key_value_heads, 8)
        self.assertEqual(card.context_length, 32768)
        self.assertFalse(card.trust_remote_code)
        self.assertFalse(card.supports_tensor)

    def test_card_saves_and_loads_locally_with_hf_forbidden(self):
        import asyncio
        from anyio import Path as AsyncPath
        from exo.shared.models.model_cards import ModelCard

        card = next(
            iter(self.parsed().shard_assignments.runner_to_shard.values())
        ).model_card
        with tempfile.TemporaryDirectory() as directory:
            path = AsyncPath(directory) / "card.toml"

            async def round_trip():
                await card.save(path)
                return await ModelCard.load_from_path(path)

            with patch.object(
                ModelCard, "fetch_from_hf", side_effect=AssertionError("HF forbidden")
            ):
                loaded = asyncio.run(round_trip())
            self.assertEqual(loaded.model_dump(), card.model_dump())

    def test_group_generation_changes_instance_and_runners_but_not_layers(self):
        first = self.build()
        self.assertEqual(first, self.build())
        self.assertEqual(instance_digest(first), instance_digest(self.build()))
        second = self.build(generation=2)
        self.assertNotEqual(instance_digest(first), instance_digest(second))
        a = CreateInstanceParams.model_validate_json(canonical(first)).instance
        b = CreateInstanceParams.model_validate_json(canonical(second)).instance
        self.assertNotEqual(a.instance_id, b.instance_id)
        self.assertTrue(
            set(a.shard_assignments.runner_to_shard).isdisjoint(
                b.shard_assignments.runner_to_shard
            )
        )
        self.assertEqual(
            sorted(
                (s.start_layer, s.end_layer)
                for s in a.shard_assignments.runner_to_shard.values()
            ),
            sorted(
                (s.start_layer, s.end_layer)
                for s in b.shard_assignments.runner_to_shard.values()
            ),
        )

    def test_parent_or_config_substitution_refused(self):
        for key, value in (
            ("revision", "d" * 40),
            ("config", self.parent["config"] | {"hidden_size": 2048}),
        ):
            parent = copy.deepcopy(self.parent)
            parent[key] = value
            with self.assertRaises(BindingRefused):
                build_explicit_instance(
                    self.plan,
                    canonical(self.manifest),
                    canonical(parent),
                    self.endpoints,
                    **self.options,
                )

    def test_manifest_substitution_refused(self):
        self.manifest["ranks"][0]["layers"] = [0, 8]
        with self.assertRaises(BindingRefused):
            self.build()

    def test_host_reorder_missing_duplicate_or_public_endpoint_refused(self):
        variations = [
            self.endpoints[:2],
            tuple(reversed(self.endpoints)),
            (self.endpoints[0], self.endpoints[0], self.endpoints[2]),
        ]
        for field, value in (
            ("node_id", self.endpoints[0].node_id),
            ("ip", "8.8.8.8"),
            ("ip", "127.0.0.1"),
            ("ip", self.endpoints[0].ip),
            ("node_id", "0" * 32),
        ):
            variations.append(
                (
                    self.endpoints[0],
                    replace(self.endpoints[1], **{field: value}),
                    self.endpoints[2],
                )
            )
        for endpoints in variations:
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                build_explicit_instance(
                    self.plan,
                    canonical(self.manifest),
                    canonical(self.parent),
                    endpoints,
                    **self.options,
                )

    def test_unsupported_group_generation_and_port_refused(self):
        for changes in (
            {"group_id": "other"},
            {"generation": True},
            {"generation": 0},
            {"ring_port": True},
            {"ring_port": 0},
            {"ring_port": 70000},
        ):
            with self.subTest(changes=changes), self.assertRaises(BindingRefused):
                self.build(**changes)

    def test_alternate_pipeline_order_is_not_silently_reinterpreted(self):
        raw = copy.deepcopy(self.raw)
        for entry in raw["members"]:
            if entry["rank"] == "slice":
                entry.update(pipeline_rank=1, layers=[7, 12])
            if entry["rank"] == "wafer":
                entry.update(pipeline_rank=0, layers=[0, 7])
        manifest = copy.deepcopy(self.manifest)
        manifest["ranks"] = [
            {"name": rank, "layers": layers}
            for rank, layers in [
                ("wafer", [0, 7]),
                ("slice", [7, 12]),
                ("obelisk", [12, 28]),
            ]
        ]
        raw["rank_manifest_sha256"] = digest(manifest)
        with self.assertRaises(BindingRefused):
            build_explicit_instance(
                GroupPlan.parse(raw),
                canonical(manifest),
                canonical(self.parent),
                self.endpoints,
                **self.options,
            )


if __name__ == "__main__":
    unittest.main()
