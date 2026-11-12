# Foreground compaction on the captured HS client

This packet applies the existing foreground-owned compaction implementation to
the exact 99-module client package captured on Hermes Samcloud. It changes six
client modules. The gateway, native SDK, model/context configuration, registry,
image service and accepted requests are unchanged. Merging this packet activates
nothing; use the ordinary client-release workflow at a real maintenance boundary.

The source implementation is pinned at
`0f55441781f956a4f800b3f7e84bc418ba66796f`. This is a client overlay, not an
instruction to deploy that entire historical gateway tree. The busy-intake
consumer change in PR91 is deliberately excluded.

## Evidence and behavior

On 12 November, HS's exact compaction `4a933d1…` completed with a normal,
nonempty result. Its next call was accepted 3.75 seconds later on its original
grant `404ba420…`, but queued at order2957 after that grant had been observed at
2945. No dependency-return record existed for this compaction. The installed
client used a separate execution-lifetime auxiliary call. This is different from
the earlier natural held-parent round trip that preserved Creative's order2939.

For a fresh compaction, the updated context adapter binds to the exact live
native foreground turn. A same-model summary uses the conversation's work-session
grant, with its own durable call identity and acknowledged intake fence. Existing
fairness and explicit yield still apply. Background forks, foreign or closed
owners, and different summary models cannot borrow that foreground ownership.
The summary algorithm, thresholds and native code do not change.

Already-bound legacy compactions retain their original execution policy and keys.
Changed inputs refuse; a model-only recovery of a newly acknowledged summary
reads the original receipt. It does not submit a replacement inference. This
packet cannot repair the scheduling position of an existing accepted legacy call.
Keep that call intact and let it finish through the normal queue.

## Stage and verify

`manifest.json` binds the complete input/output Python file sets and the three
native source seams used by the qualification. It does not attest the whole
native environment. Verify the intended interpreter and dependency environment
through the normal release checks too. A matching identity module alone does
not establish that another seat's complete bundle matches these inputs.
The patch has zero context; complete input-manifest verification is required
before applying it. Output verification checks the exact resulting bytes.

Use a new, versioned copy of the static client package. Credentials, profile
state and identity databases remain in their existing locations and are not
copied or edited. `verify.py` reads source bytes only; it creates no state,
imports no service/native modules and makes no network/process calls.

```sh
python3 /path/to/packet/verify.py --root /path/to/staged-client \
  --native-root /path/to/pinned-native --phase input
git -C /path/to/staged-client init -q
git -C /path/to/staged-client apply --check --unidiff-zero /path/to/packet/client.patch
git -C /path/to/staged-client apply --unidiff-zero /path/to/packet/client.patch
python3 /path/to/packet/verify.py --root /path/to/staged-client \
  --native-root /path/to/pinned-native --phase output
```

These commands stage source only. Before activating it, account for foreground
calls, background forks, pending compaction and queued native input. Preserve the
same profile home, stored conversation, identity database and accepted keys.
Do not terminate current work, delete bindings, replay input, change models or
reprioritize requests to make a maintenance boundary. Activate through the
normal owner-controlled runtime/profile release and record its actual path and
hashes. The image-service PR15 rollout is a separate existing leg.

Rollback also needs a schema-aware client: legacy code ignores the new
compaction bindings and can allocate a different execution key for the same
source. Do not restore an earlier identity database over later accepted work or
switch a pending continuation to the old client. Preserve the new bindings,
receipts and native history; any downgrade needs an explicit compatible
migration, not merely an empty cloud queue. This packet performs no downgrade.

## Reproduce the source qualification

Use the matching pinned native checkout's isolated test environment. Start from
the verified staged client. Copy only the test sources below (and their fixture
modules `test_inference_sdk.py` and `test_hermes_queue_client.py`) from the pinned
source commit into its `ollama/` directory, in a separate test copy. Source
verification applies to the production copy, before adding test files.

Run each file in a fresh interpreter, as the native canonical per-file test
runner does. Put the test copy first on `PYTHONPATH`, followed by the pinned
native checkout. Use synthetic profiles/HTTP fixtures and the guarded CPU-only
lane; block external network, model imports and process signals. The single
fresh-process SQLite reopen fixture needs only its owned Python child, with
the same guards inside that child. No live profile or service token is needed.

```
ollama/test_compaction_affinity_candidate.py
ollama/test_compaction_sdk_candidate.py
ollama/test_compaction_native_owner_candidate.py
ollama/test_compaction_model_recovery_candidate.py
ollama/test_inference_sdk.py
ollama/test_inference_identity.py
ollama/test_hermes_queue_identity.py
ollama/test_hermes_usage_handoff.py
```

The captured input fails all12 foreground-compaction controls. The overlay
passes79 distinct controls:26 focused compaction cases and53 related identity,
SDK and handoff cases. One fresh-process fixture was initially refused by the
guard, then passed with only that exact guarded child allowed. Two optional
loopback-bind probes and optional process probes were blocked; no external-IP
attempt, model import or process signal occurred. See `qualification.json`.

This is source/native-interface compatibility evidence. A future natural
compaction must still show its actual wire identity, return and resumed tool
use after activation. No production inference, native load, lease operation,
worker restart or profile change was performed for this packet.
