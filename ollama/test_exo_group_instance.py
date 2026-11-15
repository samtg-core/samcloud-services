"""Pinned actual wire fixture; gateway checks require no EXO/MLX imports."""

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import unittest

from .exo_groups.instance import InstanceContract
from .exo_groups.plan import BindingRefused, GroupPlan, LaunchBinding
from .exo_groups.prepare import canonical, digest


class PrivateInstanceControls(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "qualification/fixtures/exo-instance-v1.json"
        fixture = json.loads(path.read_text())
        self.plan = GroupPlan.parse(fixture["plan"])
        self.state = fixture["state"]
        self.runtime = fixture["runtime_sha256"]
        instance = next(iter(self.state["instances"].values()))
        self.contract = InstanceContract(
            digest(asdict(self.plan)), "group_" + "c" * 24, 1, self.runtime,
            canonical(fixture["request"]).decode(), canonical(instance).decode(),
        )
        self.launches = tuple(
            LaunchBinding(
                self.contract.group_id, 1, self.plan.owner_id, member.rank,
                member.pipeline_rank, member.resource_id, "lease-" + member.rank,
                member.layers, "2026-11-15T04:00:00Z", self.plan.rank_manifest_sha256,
            ) for member in sorted(self.plan.members, key=lambda m: m.pipeline_rank)
        )

    def check(self, **changes):
        args = dict(plan=self.plan, launches=self.launches, state=self.state,
                    runtime_sha256=self.runtime)
        self.contract.check_idle(**(args | changes))

    def test_actual_snake_request_and_camel_state_are_distinct(self):
        self.check()
        request = self.contract.create_request()
        self.assertIn("instance_id", request["instance"]["MlxRingInstance"])
        self.assertIn("instanceId", self.contract._instance)
        request["instance"].clear()
        self.check()
        self.assertEqual(len(self.contract.sha256), 64)

    def test_importing_checker_does_not_import_exo_or_mlx(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; from ollama.exo_groups.instance import InstanceContract; "
            "assert not any(n.split('.')[0] in ('exo','mlx','mlx_lm','mflux') for n in sys.modules)"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_runtime_plan_group_or_generation_refuses(self):
        for changes in (
            {"runtime_sha256": "a" * 64},
            {"plan": replace(self.plan, revision="b" * 40)},
            {"launches": (replace(self.launches[0], group_id="group_" + "d" * 24),) + self.launches[1:]},
            {"launches": (replace(self.launches[0], generation=2),) + self.launches[1:]},
            {"launches": (replace(self.launches[0], generation=True),) + self.launches[1:]},
            {"launches": self.launches[:2]},
            {"launches": tuple(reversed(self.launches))},
            {"launches": (replace(self.launches[0], lease_id=self.launches[1].lease_id),) + self.launches[1:]},
            {"launches": (replace(self.launches[0], layers=(0, 8)),) + self.launches[1:]},
        ):
            with self.subTest(changes=changes), self.assertRaises(BindingRefused):
                self.check(**changes)

    def test_same_model_second_instance_refuses(self):
        self.state["instances"]["another"] = copy.deepcopy(next(iter(self.state["instances"].values())))
        with self.assertRaises(BindingRefused):
            self.check()

    def test_placement_card_and_boolean_substitution_refuse(self):
        original = copy.deepcopy(self.state)
        for field, value in (("startLayer", 1), ("worldSize", True), ("deviceRank", 2)):
            self.state = copy.deepcopy(original)
            instance = next(iter(self.state["instances"].values()))["MlxRingInstance"]
            shard = next(iter(instance["shardAssignments"]["runnerToShard"].values()))["PipelineShardMetadata"]
            shard[field] = value
            with self.subTest(field=field), self.assertRaises(BindingRefused):
                self.check()
        self.state = copy.deepcopy(original)
        instance = next(iter(self.state["instances"].values()))["MlxRingInstance"]
        shard = next(iter(instance["shardAssignments"]["runnerToShard"].values()))["PipelineShardMetadata"]
        shard["modelCard"]["contextLength"] = 40959
        with self.assertRaises(BindingRefused):
            self.check()

    def test_runner_not_ready_missing_extra_or_prefill_listener_refuses(self):
        original = copy.deepcopy(self.state["runners"])
        runner = next(iter(original))
        for runners in (
            {}, original | {"other": {"RunnerReady": {"prefillServerPort": None}}},
            original | {runner: {"RunnerRunning": {}}},
            original | {runner: {"RunnerReady": {"prefillServerPort": 55000}}},
        ):
            with self.subTest(runners=runners), self.assertRaises(BindingRefused):
                self.check(state=self.state | {"runners": runners})

    def test_disaggregated_links_downloads_or_missing_observations_refuse(self):
        for key in ("downloads", "instanceLinks", "prefillServerPorts", "tasks"):
            state = copy.deepcopy(self.state)
            state.pop(key)
            with self.subTest(key=key), self.assertRaises(BindingRefused):
                self.check(state=state)
            state[key] = {"unowned": {}}
            with self.subTest(key=key), self.assertRaises(BindingRefused):
                self.check(state=state)

    def test_unfinished_foreign_or_unknown_tasks_refuse_terminal_history_allowed(self):
        task = {"taskId": "t", "instanceId": self.contract.instance_id}
        for status in ("Pending", "Running", "unknown", None):
            with self.subTest(status=status), self.assertRaises(BindingRefused):
                self.check(state=self.state | {"tasks": {"t": {"LoadModel": task | {"taskStatus": status}}}})
        for status in ("Complete", "Failed", "TimedOut", "Cancelled"):
            self.check(state=self.state | {"tasks": {"t": {"LoadModel": task | {"taskStatus": status}}}})
        for tagged in ({"Invented": task}, {"LoadModel": task | {"taskStatus": "Complete", "instanceId": "foreign"}},
                       {"LoadModel": task | {"taskStatus": "Complete", "taskId": "wrong"}}):
            with self.assertRaises(BindingRefused):
                self.check(state=self.state | {"tasks": {"t": tagged}})

    def test_staged_discovery_must_be_read_only_and_match_the_exact_node_shard(self):
        original = copy.deepcopy(self.state)
        node = next(iter(original["downloads"]))
        for field, value in (("nodeId", "foreign"), ("readOnly", False),
                             ("modelDirectory", "relative"), ("total", {"inBytes": 1}),
                             ("shardMetadata", {})):
            state = copy.deepcopy(original)
            state["downloads"][node][0]["DownloadCompleted"][field] = value
            with self.subTest(field=field), self.assertRaises(BindingRefused):
                self.check(state=state)
        for tag in ("DownloadPending", "DownloadOngoing", "DownloadFailed"):
            state = copy.deepcopy(original)
            state["downloads"][node] = [{tag: {}}]
            with self.subTest(tag=tag), self.assertRaises(BindingRefused):
                self.check(state=state)


if __name__ == "__main__":
    unittest.main()
