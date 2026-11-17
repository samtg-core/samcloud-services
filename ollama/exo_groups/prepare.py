"""Pinned configuration for EXO's existing explicit-instance API.

No lease, HTTP, model load or automatic placement. Build before acquisition;
the caller must independently establish fresh member authority before sending.
Uses the real pinned EXO types, including their tagged wire representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from ipaddress import IPv4Address, IPv4Network
import json
import re

from .plan import BindingRefused, GroupPlan, check_rank_manifest


def canonical(value: object) -> bytes:
    # The staged parent/config manifests and instance identities use these
    # exact bytes. A trailing newline changes every externally pinned digest.
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class RankEndpoint:
    rank: str
    node_id: str
    ip: str


def _identity(group_id: str, generation: int, manifest: str, name: str) -> str:
    return digest([group_id, generation, manifest, name])


def build_explicit_instance(
    plan: GroupPlan,
    manifest_bytes: bytes,
    parent_bytes: bytes,
    endpoints: tuple[RankEndpoint, ...],
    *,
    group_id: str,
    generation: int,
    ring_port: int,
) -> dict:
    """Exact three-host pipeline request; not evidence of admission/readiness."""
    # Import only for the explicit adapter. Ordinary staging stays CPU-only and
    # usable without EXO. Its EXO_HOME/resources/dashboard are caller-owned.
    from exo.api.types.api import CreateInstanceParams
    from exo.shared.models.model_cards import ModelCard, ModelTask
    from exo.shared.types.backends import Backend
    from exo.shared.types.common import Host, ModelId, NodeId
    from exo.shared.types.memory import Memory
    from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
    from exo.shared.types.worker.runners import RunnerId, ShardAssignments
    from exo.shared.types.worker.shards import PipelineShardMetadata

    check_rank_manifest(plan, manifest_bytes)
    manifest = json.loads(manifest_bytes)
    parent = json.loads(parent_bytes)
    config = parent.get("config", {})
    if (
        digest(parent) != manifest.get("parent_manifest_sha256")
        or digest(config) != manifest.get("config_sha256")
        or parent.get("model_id") != plan.model_id
        or parent.get("revision") != plan.revision
        or config.get("model_type") not in ("qwen3", "qwen3_moe")
    ):
        raise BindingRefused("model card parent differs from the pinned bundle")
    ordered = sorted(plan.members, key=lambda member: member.pipeline_rank)
    # The current runtime's ring guard uses this fixed rank order. Refuse a
    # different order rather than generate a request it will interpret wrongly.
    if tuple(member.rank for member in ordered) != plan.pipeline_ranks:
        raise BindingRefused("pipeline order differs from the qualified ring profile")
    if config.get("num_hidden_layers") != ordered[-1].layers[1]:
        raise BindingRefused("parent layers differ from the complete placement")
    if (
        not re.fullmatch(r"group_[a-f0-9]{24}", group_id)
        or type(generation) is not int
        or generation < 1
        or type(ring_port) is not int
        or not 1024 <= ring_port <= 65535
    ):
        raise BindingRefused("explicit group generation and ring port required")
    if (
        len(endpoints) != len(ordered)
        or [endpoint.rank for endpoint in endpoints]
        != [member.rank for member in ordered]
        or len({endpoint.node_id for endpoint in endpoints}) != len(ordered)
        or len({endpoint.ip for endpoint in endpoints}) != len(ordered)
    ):
        raise BindingRefused("exact unique pipeline-ordered endpoints required")
    private = tuple(
        IPv4Network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    )
    for endpoint in endpoints:
        if not re.fullmatch(r"[1-9a-f][0-9a-f]{0,31}", endpoint.node_id):
            raise BindingRefused("node id is not the pinned runtime's Zenoh id")
        address = IPv4Address(endpoint.ip)
        if str(address) != endpoint.ip or not any(address in net for net in private):
            raise BindingRefused("exact private IPv4 endpoints required")
    files = parent.get("weight_files")
    if not isinstance(files, list) or not files:
        raise BindingRefused("pinned parent weight identities required")
    for entry in files:
        if (
            type(entry.get("bytes")) is not int
            or entry["bytes"] <= 0
            or not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", ""))
        ):
            raise BindingRefused("invalid parent weight identity")
    card = ModelCard(
        model_id=ModelId(plan.model_id),
        storage_size=Memory.from_bytes(sum(entry["bytes"] for entry in files)),
        n_layers=config["num_hidden_layers"],
        hidden_size=config["hidden_size"],
        supports_tensor=False,
        num_key_value_heads=config["num_key_value_heads"],
        tasks=[ModelTask.TextGeneration],
        family=config["model_type"],
        context_length=config["max_position_embeddings"],
        trust_remote_code=False,
        backends=[Backend.MlxMetal],
    )

    def token(name: str) -> str:
        return _identity(group_id, generation, plan.rank_manifest_sha256, name)

    runners = {
        member.rank: RunnerId(token("runner:" + member.rank)) for member in ordered
    }
    hosts = [Host(ip=endpoint.ip, port=ring_port) for endpoint in endpoints]
    instance = MlxRingInstance(
        instance_id=InstanceId(token("instance")),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={
                NodeId(endpoint.node_id): runners[endpoint.rank]
                for endpoint in endpoints
            },
            runner_to_shard={
                runners[member.rank]: PipelineShardMetadata(
                    model_card=card,
                    device_rank=member.pipeline_rank,
                    world_size=len(ordered),
                    start_layer=member.layers[0],
                    end_layer=member.layers[1],
                    n_layers=card.n_layers,
                )
                for member in ordered
            },
        ),
        hosts_by_node={NodeId(endpoint.node_id): list(hosts) for endpoint in endpoints},
        ephemeral_port=ring_port,
    )
    request = CreateInstanceParams(instance=instance)
    wire = request.model_dump(mode="json")
    # Validate the actual tagged JSON, rather than only in-process objects.
    parsed = CreateInstanceParams.model_validate_json(canonical(wire))
    if parsed.model_dump(mode="json") != wire:
        raise BindingRefused("EXO instance wire round trip changed the assignment")
    return wire


def instance_digest(request: dict) -> str:
    return hashlib.sha256(canonical(request)).hexdigest()
