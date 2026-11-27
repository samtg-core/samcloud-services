"""Journaled, once-submitted cleanup artifact publication with exact readback.

No inference retry, timer or host process effects. File POST has no idempotency
key: a lost reply reconciles the saved name/owner/audience/hash, never re-POSTs.
"""

import copy
import hashlib
import re

from ..inference_arbiter import Conflict
from .cleanup import validate_cleanup
from .plan import BindingRefused
from .prepare import digest


class PublicationUnknown(RuntimeError):
    """Keep the physical claim; explicit metadata reconciliation is required."""


class GroupPublication:
    def __init__(self, completion, *, group_scope):
        if not isinstance(group_scope, str) or not re.fullmatch(
            r"[a-zA-Z0-9_-]{1,100}", group_scope
        ):
            raise ValueError("explicit existing file audience required")
        self.completion, self.group_scope = completion, group_scope
        self.journal, self.plane, self.plan = (
            completion.journal,
            completion.plane,
            completion.plan,
        )
        with self.journal.transaction():
            self.journal.db.execute("""CREATE TABLE IF NOT EXISTS exo_publication_policy (
                owner TEXT NOT NULL,request_key TEXT NOT NULL,group_scope TEXT NOT NULL,
                PRIMARY KEY(owner,request_key),
                FOREIGN KEY(owner,request_key) REFERENCES exo_group_intents(owner,request_key)
            )""")
            self.journal.db.execute("""CREATE TABLE IF NOT EXISTS exo_cleanup_publications (
                owner TEXT NOT NULL,request_key TEXT NOT NULL,lease_id TEXT NOT NULL,
                file_owner TEXT NOT NULL,group_scope TEXT NOT NULL,name TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL,raw BLOB NOT NULL,
                upload_started INTEGER NOT NULL DEFAULT 0 CHECK(upload_started IN (0,1)),
                file_id INTEGER,confirmed INTEGER NOT NULL DEFAULT 0 CHECK(confirmed IN (0,1)),
                PRIMARY KEY(owner,request_key,lease_id),
                FOREIGN KEY(owner,request_key,lease_id)
                    REFERENCES exo_group_cleanup_pins(owner,request_key,lease_id)
            )""")

    def _key(self, lease_id):
        return self.plan.owner_id, self.plan.request_key, lease_id

    def _saved(self, lease_id):
        row = self.journal.db.execute(
            "SELECT * FROM exo_cleanup_publications "
            "WHERE owner=? AND request_key=? AND lease_id=?",
            self._key(lease_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def _validate(self, lease_id, raw):
        if not isinstance(raw, bytes) or not 0 < len(raw) <= 128 * 1024:
            raise BindingRefused("exact bounded host evidence bytes required")
        validate_cleanup(raw, self.completion._pin(lease_id))
        return hashlib.sha256(raw).hexdigest()

    async def publish(self, lease_id, raw):
        sha = self._validate(lease_id, raw)
        return await self.completion._owned(
            ("publish", lease_id, sha, self.group_scope),
            lambda: self._publish(lease_id, raw),
        )

    async def complete_member(self, lease_id, raw, *, generation):
        sha = self._validate(lease_id, raw)
        self.completion.intents.begin_drain(
            self.plan.owner_id, self.plan.request_key, generation
        )
        self.completion.actor.wake()

        async def finish():
            publication = await self._publish(lease_id, raw)
            return await self.completion._settle(
                lease_id, generation, publication["file_id"], publication["sha256"]
            )

        return await self.completion._owned(
            ("publish_settle", lease_id, sha, self.group_scope, generation), finish
        )

    async def _publish(self, lease_id, raw):
        sha = self._validate(lease_id, raw)
        file_owner = await self.plane.file_owner()
        if not isinstance(file_owner, str) or not file_owner:
            raise BindingRefused("verified publication owner required")
        run = self.completion._pin(lease_id).binding
        name = (
            "exo-cleanup-"
            + digest([self.plan.owner_id, run.group_id, lease_id, sha])
            + ".json"
        )
        with self.journal.transaction():
            policy = self.journal.db.execute(
                "SELECT group_scope FROM exo_publication_policy "
                "WHERE owner=? AND request_key=?",
                self._key(lease_id)[:2],
            ).fetchone()
            if policy is not None and policy["group_scope"] != self.group_scope:
                raise Conflict("group cleanup publication audience is already bound")
            self.journal.db.execute(
                "INSERT OR IGNORE INTO exo_publication_policy VALUES(?,?,?)",
                (*self._key(lease_id)[:2], self.group_scope),
            )
            saved = self._saved(lease_id)
            if saved is not None:
                expected = (file_owner, self.group_scope, name, sha, raw)
                if (
                    tuple(
                        saved[k]
                        for k in ("file_owner", "group_scope", "name", "sha256", "raw")
                    )
                    != expected
                ):
                    raise Conflict(
                        "exact cleanup publication identity/audience/bytes are immutable"
                    )
            else:
                self.journal.db.execute(
                    """INSERT INTO exo_cleanup_publications
                    (owner,request_key,lease_id,file_owner,group_scope,name,sha256,raw)
                    VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        *self._key(lease_id),
                        file_owner,
                        self.group_scope,
                        name,
                        sha,
                        raw,
                    ),
                )
                saved = self._saved(lease_id)
        if saved["file_id"] is None:
            reply = await self.plane.artifacts_named(name)
            rows = reply.get("files") if isinstance(reply, dict) else None
            if (
                not isinstance(rows, list)
                or len(rows) >= 200
                or any(not isinstance(row, dict) for row in rows)
            ):
                raise PublicationUnknown(
                    "cleanup file listing is unconfirmed or truncated"
                )
            matches = [row for row in rows if row.get("name") == name]
            if len(matches) > 1:
                raise PublicationUnknown(
                    "cleanup file identity is ambiguous; no upload"
                )
            if matches:
                metadata = matches[0]
                self._metadata(saved, metadata)
                self._bind_file(lease_id, metadata["id"])
            else:
                with self.journal.transaction():
                    current = self._saved(lease_id)
                    if current["upload_started"] or current["file_id"] is not None:
                        raise PublicationUnknown(
                            "submitted cleanup upload has no exact visible receipt"
                        )
                    self.journal.db.execute(
                        "UPDATE exo_cleanup_publications SET upload_started=1 "
                        "WHERE owner=? AND request_key=? AND lease_id=?",
                        self._key(lease_id),
                    )
                # The committed decision outlives cancellation/crash before or
                # after the POST. Neither a timeout nor an empty list resets it.
                metadata = await self.plane.upload_artifact(
                    name=name, group_scope=self.group_scope, raw=raw
                )
                self._metadata(saved, metadata)
                self._bind_file(lease_id, metadata["id"])
        saved = self._saved(lease_id)
        metadata = await self.plane.artifact(saved["file_id"])
        self._metadata(saved, metadata)
        data = await self.plane.artifact_bytes(saved["file_id"])
        if data != raw or hashlib.sha256(data).hexdigest() != sha:
            raise PublicationUnknown("published cleanup bytes differ; retain ownership")
        self._metadata(saved, await self.plane.artifact(saved["file_id"]))
        with self.journal.transaction():
            self.journal.db.execute(
                "UPDATE exo_cleanup_publications SET confirmed=1 "
                "WHERE owner=? AND request_key=? AND lease_id=?",
                self._key(lease_id),
            )
        return copy.deepcopy(
            {
                "state": "exact_cleanup_publication_confirmed",
                "file_id": saved["file_id"],
                "sha256": sha,
                "bytes": len(raw),
                "name": name,
                "group_scope": self.group_scope,
                "owner": file_owner,
            }
        )

    @staticmethod
    def _metadata(saved, metadata):
        if (
            not isinstance(metadata, dict)
            or type(metadata.get("id")) is not int
            or metadata["id"] < 1
            or saved["file_id"] is not None
            and metadata["id"] != saved["file_id"]
            or any(
                type(metadata.get(k)) is not type(v) or metadata.get(k) != v
                for k, v in {
                    "name": saved["name"],
                    "owner": saved["file_owner"],
                    "group_scope": saved["group_scope"],
                    "sha256": saved["sha256"],
                    "bytes": len(saved["raw"]),
                    "content_type": "application/octet-stream",
                }.items()
            )
        ):
            raise PublicationUnknown(
                "exact cleanup publication metadata is unconfirmed"
            )

    def _bind_file(self, lease_id, file_id):
        with self.journal.transaction():
            saved = self._saved(lease_id)
            if saved["file_id"] not in (None, file_id):
                raise Conflict("cleanup publication already names another file id")
            self.journal.db.execute(
                "UPDATE exo_cleanup_publications SET file_id=? "
                "WHERE owner=? AND request_key=? AND lease_id=?",
                (file_id, *self._key(lease_id)),
            )
