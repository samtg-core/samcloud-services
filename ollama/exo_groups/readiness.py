"""Durable provider readiness attestation, separate from inference submission.

The exact evidence is retained in the existing model-service journal. Core
records its attributed journal reference/hash; no model is launched here.
"""

from dataclasses import asdict
import json

from ..inference_arbiter import Conflict
from .plan import BindingRefused
from .prepare import canonical, digest
from .startup import InstanceStartup


class GroupReadiness:
    def __init__(self, provider, startup):
        if (
            not isinstance(startup, InstanceStartup)
            or provider.receipts is None
            or startup.actor is not provider.actor
            or provider.backend is None
            or startup.starts.bindings is not provider.backend.bindings
            or startup.contract != provider.backend.contract
            or startup.launches != provider.backend.launches
        ):
            raise ValueError(
                "one exact provider, receipt set and private startup required"
            )
        self.provider, self.startup = provider, startup
        self.actor, self.plan = provider.actor, provider.actor.plan
        self.completion, self.journal = provider.completion, provider.completion.journal
        with self.journal.transaction():
            self.journal.db.execute("""CREATE TABLE IF NOT EXISTS exo_group_readiness (
                owner TEXT NOT NULL,request_key TEXT NOT NULL,generation INTEGER NOT NULL,
                evidence_json TEXT NOT NULL,evidence_sha256 TEXT NOT NULL,
                evidence_ref TEXT NOT NULL,confirmed INTEGER NOT NULL DEFAULT 0 CHECK(confirmed IN(0,1)),
                PRIMARY KEY(owner,request_key,generation),
                FOREIGN KEY(owner,request_key) REFERENCES exo_group_intents(owner,request_key)
            )""")

    def saved(self):
        row = self.journal.db.execute(
            "SELECT * FROM exo_group_readiness "
            "WHERE owner=? AND request_key=? AND generation=?",
            (
                self.plan.owner_id,
                self.plan.request_key,
                self.startup.contract.generation,
            ),
        ).fetchone()
        return dict(row) if row is not None else None

    async def attest(self):
        return await self.completion._owned(
            ("ready", self.startup.contract.sha256), self._attest
        )

    def _current(self, epoch):
        if (
            self.actor._epoch != epoch
            or not self.actor.may_start_ranks()
            or self.provider._busy()
            or self.actor.intents.closing(self.plan.owner_id, self.plan.request_key)
        ):
            raise BindingRefused("current all-member startup authority required")

    async def _attest(self):
        if self.provider._busy():
            raise Conflict("readiness cannot replace running or uncertain inference")
        self.actor.wake()
        await self.actor.wait()
        epoch = self.actor._epoch
        self._current(epoch)
        contract = self.startup.contract
        submitted = self.startup.starts.snapshot(
            self.plan.owner_id, self.plan.request_key
        )
        if submitted["contract_sha256"] != contract.sha256:
            raise BindingRefused(
                "ready instance has no matching once-submitted startup"
            )
        reports = await self.provider.receipts._observe()
        wanted = {launch.lease_id for launch in self.startup.launches}
        if set(reports) != wanted or any(
            report.started is None
            or report.cleanup_bytes is not None
            or report.phase != "spawning"
            for report in reports.values()
        ):
            raise BindingRefused("all three actual live host-start receipts required")
        hosts = {lease: asdict(report.started) for lease, report in reports.items()}
        runtime = await self.startup.verify_runtime()
        state = await self.startup.private.state()
        contract.check_idle(
            self.plan, self.startup.launches, state, runtime_sha256=runtime
        )
        if (
            type(state.get("lastEventAppliedIdx")) is not int
            or state["lastEventAppliedIdx"] < submitted["state_event_idx"]
        ):
            raise BindingRefused("private ready state predates the submitted instance")
        self._current(epoch)
        # This exact first ready observation stays immutable. A later benign
        # runtime event may change state history; it never replaces the original
        # attestation or changes the independently pinned host child identities.
        identity = {
            "group_id": contract.group_id,
            "generation": contract.generation,
            "contract_sha256": contract.sha256,
            "runtime_sha256": runtime,
            "hosts": hosts,
        }
        candidate = {
            **identity,
            "private_state_sha256": digest(state),
            "state_event_idx": state["lastEventAppliedIdx"],
            "startup_state_event_idx": submitted["state_event_idx"],
        }
        with self.journal.transaction():
            saved = self.saved()
            if saved is not None:
                old = json.loads(saved["evidence_json"])
                if (
                    canonical(old).decode() != saved["evidence_json"]
                    or digest(old) != saved["evidence_sha256"]
                    or saved["evidence_ref"]
                    != "model-service-journal:exo_group_readiness:" + digest(old)
                ):
                    raise Conflict(
                        "saved serving attestation bytes differ from their digest"
                    )
                if any(
                    canonical(old.get(k)) != canonical(v) for k, v in identity.items()
                ):
                    raise Conflict("serving attestation identity is immutable")
            else:
                raw = canonical(candidate).decode()
                sha = digest(candidate)
                self.journal.db.execute(
                    "INSERT INTO exo_group_readiness VALUES(?,?,?,?,?,?,0)",
                    (
                        self.plan.owner_id,
                        self.plan.request_key,
                        contract.generation,
                        raw,
                        sha,
                        "model-service-journal:exo_group_readiness:" + sha,
                    ),
                )
                saved = self.saved()
        proof = {k: saved[k] for k in ("evidence_ref", "evidence_sha256")}
        group = await self.completion._group(contract.generation)
        self._current(epoch)
        if group.get("phase") != "serving":
            if group.get("phase") != "admitted" or group.get("release_requested"):
                raise BindingRefused("ready group no longer admitted")
            await self.actor.plane.mark_serving(
                contract.group_id, contract.generation, proof
            )
        for _ in range(2):
            group = await self.completion._group(contract.generation)
            acknowledged = group.get("serving_evidence")
            if (
                group.get("phase") != "serving"
                or group.get("release_requested")
                or not isinstance(acknowledged, dict)
                or acknowledged.get("generation") != contract.generation
                or acknowledged.get("attested_by") != self.plan.owner_id
                or any(acknowledged.get(k) != v for k, v in proof.items())
            ):
                raise BindingRefused("exact serving attestation is not acknowledged")
        self.actor.wake()
        await self.actor.wait()
        if not self.actor.may_serve_requests():
            raise BindingRefused("fresh serving authority is unavailable")
        with self.journal.transaction():
            self.journal.db.execute(
                "UPDATE exo_group_readiness SET confirmed=1 "
                "WHERE owner=? AND request_key=? AND generation=?",
                (self.plan.owner_id, self.plan.request_key, contract.generation),
            )
        self.provider.wake()
        return {
            "state": "exact_group_readiness_confirmed",
            "group_id": contract.group_id,
            "generation": contract.generation,
            **proof,
        }
