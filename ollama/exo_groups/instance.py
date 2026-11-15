"""Exact private-instance observations, independent of lease authority.

Preparation uses the pinned EXO types in its own environment. The provider can
check the resulting immutable JSON without importing MLX or EXO into its gateway.
An idle observation is not admission, host attestation or backend-death proof.
"""

from dataclasses import asdict, dataclass
import json

from .member import ExpectedMember
from .plan import BindingRefused, GroupPlan, LaunchBinding, _digest
from .prepare import RankEndpoint, build_explicit_instance, canonical, digest


@dataclass(frozen=True)
class InstanceContract:
    plan_sha256: str
    group_id: str
    generation: int
    runtime_sha256: str
    request_json: str
    state_instance_json: str

    @classmethod
    def prepare(
        cls, plan: GroupPlan, manifest: bytes, parent: bytes,
        endpoints: tuple[RankEndpoint, ...], *, group_id: str,
        generation: int, ring_port: int, runtime_sha256: str,
    ) -> "InstanceContract":
        """CPU-only preparation; the caller independently pins the EXO source.

        Preserve both real EXO wire forms: /instance uses snake_case while
        /state uses aliases. Do not guess a recursive name conversion.
        """
        _digest(runtime_sha256, "runtime source")
        wire = build_explicit_instance(
            plan, manifest, parent, endpoints, group_id=group_id,
            generation=generation, ring_port=ring_port,
        )
        from exo.api.types.api import CreateInstanceParams

        instance = CreateInstanceParams.model_validate_json(canonical(wire)).instance
        state_wire = instance.model_dump(mode="json", by_alias=True)
        return cls(
            digest(asdict(plan)), group_id, generation, runtime_sha256,
            canonical(wire).decode(), canonical(state_wire).decode(),
        )

    @property
    def instance_id(self) -> str:
        return self._instance["instanceId"]

    @property
    def _instance(self) -> dict:
        return json.loads(self.state_instance_json)["MlxRingInstance"]

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def create_request(self) -> dict:
        """Return detached data; obtaining it never authorizes its submission."""
        return json.loads(self.request_json)

    def check_idle(
        self, plan: GroupPlan, launches: tuple[LaunchBinding, ...],
        state: dict, *, runtime_sha256: str,
    ) -> None:
        """Refuse another instance, placement, runner or unfinished task.

        Requires a fresh authenticated private /state read and independently
        verified host/runtime receipts. It performs no I/O and grants no lease
        permission. Authority must still be current at actual dispatch.
        """
        if digest(asdict(plan)) != self.plan_sha256 or runtime_sha256 != self.runtime_sha256:
            raise BindingRefused("private instance plan or runtime changed")
        if (len(launches) != 3
                or tuple(item.pipeline_rank for item in launches) != (0, 1, 2)
                or any(type(item.pipeline_rank) is not int or type(item.generation) is not int
                       for item in launches)
                or len({item.lease_id for item in launches}) != 3):
            raise BindingRefused("exact three pipeline launch bindings required")
        for item in launches:
            ExpectedMember.from_launch(plan, item)
            if item.group_id != self.group_id or item.generation != self.generation:
                raise BindingRefused("private instance belongs to another group generation")
        if not isinstance(state, dict):
            raise BindingRefused("private EXO state is not an object")
        expected = {self.instance_id: json.loads(self.state_instance_json)}
        # Canonical comparison also distinguishes JSON true from integer 1.
        if canonical(state.get("instances")) != canonical(expected):
            raise BindingRefused("private namespace has a different or extra instance")
        assigned = self._instance["shardAssignments"]["runnerToShard"]
        runners = state.get("runners")
        if not isinstance(runners, dict) or set(runners) != set(assigned):
            raise BindingRefused("private namespace has missing or extra runners")
        # The qualified profile disables disaggregation. A prefill server opens
        # another listener and must not silently widen that network contract.
        ready = {"RunnerReady": {"prefillServerPort": None}}
        if any(canonical(status) != canonical(ready) for status in runners.values()):
            raise BindingRefused("assigned runner is not ready on the qualified profile")
        if state.get("prefillServerPorts") != {} or state.get("instanceLinks") != {}:
            raise BindingRefused("unexpected disaggregated instance or prefill listener")
        # EXO retains read-only discovery as DownloadCompleted history. It is
        # not an active network download and must match this node's exact shard.
        downloads = state.get("downloads")
        nodes = self._instance["shardAssignments"]["nodeToRunner"]
        if not isinstance(downloads, dict) or set(downloads) != set(nodes):
            raise BindingRefused("private namespace has unrelated download observations")
        for node, entries in downloads.items():
            if not isinstance(entries, list) or len(entries) != 1:
                raise BindingRefused("unexpected private download history")
            progress = entries[0]
            if not isinstance(progress, dict) or set(progress) != {"DownloadCompleted"}:
                raise BindingRefused("private namespace still has unfinished downloads")
            done = progress["DownloadCompleted"]
            if (not isinstance(done, dict) or done.get("nodeId") != node
                    or done.get("readOnly") is not True
                    or not isinstance(done.get("modelDirectory"), str)
                    or not done["modelDirectory"].startswith("/")
                    or canonical(done.get("total")) != canonical(
                        assigned[nodes[node]]["PipelineShardMetadata"]["modelCard"]["storageSize"])
                    or canonical(done.get("shardMetadata")) != canonical(assigned[nodes[node]])):
                raise BindingRefused("completed discovery differs from the read-only rank")
        tasks = state.get("tasks")
        if not isinstance(tasks, dict):
            raise BindingRefused("private state lacks task observations")
        known = {"CreateRunner", "DownloadModel", "LoadModel", "ConnectToGroup",
                 "StartWarmup", "TextGeneration", "CancelTask", "ImageGeneration",
                 "ImageEdits", "Shutdown"}
        ended = {"Complete", "TimedOut", "Failed", "Cancelled"}
        for ident, tagged in tasks.items():
            if not isinstance(tagged, dict) or len(tagged) != 1 or not set(tagged) <= known:
                raise BindingRefused("unknown private task shape")
            task = next(iter(tagged.values()))
            if (not isinstance(task, dict) or task.get("taskId") != ident
                    or task.get("instanceId") != self.instance_id
                    or not isinstance(task.get("taskStatus"), str)
                    or task["taskStatus"] not in ended):
                raise BindingRefused("private task is unfinished or belongs to another instance")
