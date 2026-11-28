"""Cleanup publication crash boundaries; injected hosts, real local journal."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
import unittest

from . import test_exo_group_completion as completion_controls
from .exo_groups.completion import GroupCompletion
from .exo_groups.install import GroupProvider
from .exo_groups.intents import GroupIntents
from .exo_groups.plan import BindingRefused
from .exo_groups.publication import GroupPublication, PublicationUnknown
from .inference_arbiter import Arbiter, Conflict


class PublicationControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.parts = completion_controls.CompletionControls()
        await self.parts.asyncSetUp()
        self.addAsyncCleanup(self.parts.base.asyncTearDown)
        p = self.parts
        self.lease = p.lease_id
        self.raw = p.files[p.proofs[self.lease][0]]
        self.records, self.uploads = {}, []
        self.owner = "codex"
        self.lose = self.hide = self.corrupt = self.fence = False
        self.extra = []
        self.entered, self.proceed = asyncio.Event(), asyncio.Event()
        self.proceed.set()

        async def file_owner():
            return self.owner

        async def listing(name):
            return {
                "files": self.extra
                + (
                    []
                    if self.hide
                    else [
                        copy.deepcopy(m)
                        for m, _ in self.records.values()
                        if m["name"] == name
                    ]
                )
            }

        async def upload(*, name, group_scope, raw):
            saved = self.publisher._saved(self.lease)
            self.assertEqual(saved["upload_started"], 1)
            self.assertEqual(saved["raw"], raw)
            self.assertFalse(p.base.journal.db.in_transaction)
            file_id = 100 + len(self.uploads)
            meta = {
                "id": file_id,
                "name": name,
                "owner": self.owner,
                "group_scope": group_scope,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "content_type": "application/octet-stream",
            }
            self.records[file_id] = (meta, raw)
            self.uploads.append(file_id)
            self.entered.set()
            await self.proceed.wait()
            if self.fence:
                p.group.update(generation=2, phase="releasing", release_requested=True)
            if self.lose:
                self.lose = False
                raise OSError("reply lost after store")
            return copy.deepcopy(meta)

        async def artifact(file_id):
            return copy.deepcopy(self.records[file_id][0])

        async def artifact_bytes(file_id):
            return self.records[file_id][1] + (b" " if self.corrupt else b"")

        p.plane.file_owner, p.plane.artifacts_named = file_owner, listing
        p.plane.upload_artifact = upload
        p.plane.artifact, p.plane.artifact_bytes = artifact, artifact_bytes
        self.publisher = GroupPublication(p.completion, group_scope="projects")

    async def publish(self):
        return await self.publisher.publish(self.lease, self.raw)

    async def complete(self, generation=1):
        return await self.publisher.complete_member(
            self.lease, self.raw, generation=generation
        )

    async def test_publication_commits_before_post_and_reads_exact_bytes(self):
        result = await self.publish()
        self.assertEqual(result["state"], "exact_cleanup_publication_confirmed")
        self.assertEqual(result["owner"], "codex")
        self.assertEqual(self.uploads, [100])
        self.assertEqual((await self.publish())["file_id"], 100)
        self.assertEqual(self.uploads, [100])
        self.assertEqual(self.publisher._saved(self.lease)["confirmed"], 1)
        self.assertEqual(self.parts.releases(), [])

    async def test_lost_reply_reconciles_same_file_after_journal_reopen(self):
        self.lose = True
        with self.assertRaises(OSError):
            await self.publish()
        p = self.parts
        b = p.base
        b.journal.close()
        b.journal = Arbiter(Path(b.temp.name) / "queue.sqlite")
        b.intents = p.actor.intents = GroupIntents(b.journal)
        p.completion = GroupCompletion(p.actor)
        self.publisher = GroupPublication(p.completion, group_scope="projects")
        self.assertEqual((await self.publish())["file_id"], 100)
        self.assertEqual(self.uploads, [100])

    async def test_unknown_upload_with_no_visible_file_never_posts_again(self):
        self.lose = True
        with self.assertRaises(OSError):
            await self.publish()
        self.hide = True
        for _ in range(2):
            with self.assertRaises(PublicationUnknown):
                await self.publish()
        self.assertEqual(self.uploads, [100])
        self.assertEqual(self.publisher._saved(self.lease)["upload_started"], 1)
        self.hide = False
        self.assertEqual((await self.publish())["file_id"], 100)

    async def test_matching_preexisting_file_is_reused_without_post(self):
        self.lose = True
        with self.assertRaises(OSError):
            await self.publish()
        # Simulate recovery finding a matching record before its upload flag.
        self.parts.base.journal.db.execute(
            "UPDATE exo_cleanup_publications SET upload_started=0"
        )
        self.assertEqual((await self.publish())["file_id"], 100)
        self.assertEqual(self.uploads, [100])

    async def test_two_observers_share_upload_and_disconnect_preserves_settlement(self):
        self.parts.end_call()
        self.proceed.clear()
        first = asyncio.create_task(self.complete())
        await self.entered.wait()
        second = asyncio.create_task(self.complete())
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertFalse(self.parts.completion._task.done())
        self.proceed.set()
        self.assertEqual((await second)["state"], "exact_member_settlement_confirmed")
        self.assertEqual(self.uploads, [100])
        self.assertEqual(len(self.parts.releases()), 1)
        self.assertEqual(
            sum(
                lease["status"] == "active" for lease in self.parts.base.leases.values()
            ),
            2,
        )

    async def test_fence_during_upload_keeps_file_but_never_releases_stale_generation(
        self,
    ):
        self.parts.end_call()
        self.fence = True
        with self.assertRaises(BindingRefused):
            await self.complete()
        self.assertEqual(self.parts.releases(), [])
        self.assertEqual(self.publisher._saved(self.lease)["file_id"], 100)
        self.fence = False
        self.assertEqual((await self.complete(generation=2))["launch_generation"], 1)
        self.assertEqual(self.uploads, [100])

    async def test_pending_compute_blocks_upload_and_authority_withdrawal(self):
        p = self.parts
        for state in ("queued", "running", "uncertain"):
            p.base.journal.db.execute(
                "UPDATE executions SET state=? WHERE id=?", (state, p.base.call["id"])
            )
            with self.assertRaises(Conflict):
                await self.complete()
            self.assertFalse(
                p.base.intents.closing(p.base.plan.owner_id, p.base.plan.request_key)
            )
        self.assertEqual(self.uploads, [])
        self.assertEqual(p.mutations, [])

    async def test_metadata_owner_scope_size_hash_and_mime_must_match(self):
        await self.publish()
        meta = self.records[100][0]
        for key, value in (
            ("owner", "peer"),
            ("group_scope", "services"),
            ("bytes", True),
            ("sha256", "f" * 64),
            ("content_type", "application/json"),
            ("id", 101),
        ):
            old = meta[key]
            meta[key] = value
            with self.assertRaises(PublicationUnknown):
                await self.publish()
            meta[key] = old
        self.assertEqual(self.uploads, [100])
        self.assertEqual(self.parts.releases(), [])

    async def test_changed_raw_bytes_never_confirm_or_settle(self):
        self.parts.end_call()
        self.corrupt = True
        with self.assertRaises(PublicationUnknown):
            await self.complete()
        self.assertEqual(self.publisher._saved(self.lease)["confirmed"], 0)
        self.assertEqual(self.parts.releases(), [])
        self.corrupt = False
        self.assertEqual(
            (await self.complete())["state"], "exact_member_settlement_confirmed"
        )
        self.assertEqual(self.uploads, [100])

    async def test_immutable_observation_even_if_new_bytes_validate(self):
        await self.publish()
        value = json.loads(self.raw)
        value["observation"]["observed_at"] = "2026-11-14T10:06:00+00:00"
        with self.assertRaises(Conflict):
            await self.publisher.publish(self.lease, json.dumps(value).encode())
        self.owner = "renamed"
        with self.assertRaises(Conflict):
            await self.publish()
        self.assertEqual(self.uploads, [100])

    async def test_audience_is_immutable_across_members_and_reopen(self):
        await self.publish()
        other = next(lease for lease in self.parts.pins if lease != self.lease)
        raw = self.parts.files[self.parts.proofs[other][0]]
        changed = GroupPublication(self.parts.completion, group_scope="services")
        with self.assertRaises(Conflict):
            await changed.publish(other, raw)
        self.assertEqual(self.uploads, [100])

    async def test_ambiguous_or_truncated_list_never_uploads(self):
        self.lose = True
        with self.assertRaises(OSError):
            await self.publish()
        self.extra = [copy.deepcopy(self.records[100][0])]
        with self.assertRaises(PublicationUnknown):
            await self.publish()
        self.extra = [{"name": "unrelated"}] * 200
        with self.assertRaises(PublicationUnknown):
            await self.publish()
        self.assertEqual(self.uploads, [100])

    async def test_provider_requires_explicit_audience_and_owned_completion(self):
        # Exercise the actual provider methods without unrelated runtime startup.
        p = GroupProvider.__new__(GroupProvider)
        p.completion, p.publication = self.parts.completion, None
        with self.assertRaises(RuntimeError):
            await p.settle_observed_member(self.lease, self.raw, generation=1)
        self.assertIs(
            p.bind_publication(group_scope="projects"),
            p.bind_publication(group_scope="projects"),
        )
        with self.assertRaises(ValueError):
            p.bind_publication(group_scope="services")
        self.parts.end_call()
        self.assertEqual(
            (await p.settle_observed_member(self.lease, self.raw, generation=1))[
                "state"
            ],
            "exact_member_settlement_confirmed",
        )
