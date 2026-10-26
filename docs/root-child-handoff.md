# Child-service handoff — remaining integration, not an advertised capability

A native text parent can retain one physical root across its own model/tool/
compaction requests. A separate image service cannot simply request another
exclusive claim on the same resource while that parent waits: it would queue
behind its own parent. RootQueueStartup does not solve cross-service transfer.
The public inference capability remains `cross_service_delegation: false`.
First standalone native image/endpoint delivery stays independent; these are the
later integration requirements, not another permission gate for Hermes.

## Existing registry contract

The unmounted registry handoff router offers an exact current full envelope,
accepts only the named registered child service's owner, and hands it back only
with that child's exact lease and absence observation. Transfer is an atomic
successor on the same root; the old member becomes released and the root's
current lease/actor changes. It neither creates a second capacity envelope nor
promotes a foreign waiter in an internal release/insert gap. Reads/replays return
current mapped state, not cached admission. The root lifetime stays explicit-end
with finite health and hold-on-expiry; no elapsed job time ends it.

These primitives exist and have temporary-registry tests. They are not mounted
in production, not wired into the model/image service boundary, and not a claim
that delegated image generation is ready. RegistryRootClient.begin submits its
own root; it is not an accept operation for a borrowed child allocation.

## Principals and job binding

The agent authenticates as itself to each service. Its parent `job_key` and the
service URL identify logical work, not authority to borrow a physical lease.
The text service must verify that requester owns the live parent and that the
specific pending tool/child operation belongs to it. Reading root/lease IDs from
status is not consent. Service credentials never pass to the requesting agent.

The text provider owns its own physical-root credential and known backend
handles. Only it can drain its execution, observe its backend/cache absent and
offer that exact current lease to a permitted named target service. A caller's
claim that compute is idle is insufficient. Root suspension provides relevant
cold-drain checks but is not the complete child handoff/liveness protocol.

The image provider authenticates with its own registered service identity. It
must bind the offered context to one durable accepted image job, expected parent/
root/resource/source lease and requester before consenting. It must recover the
same offer/acceptance after lost acknowledgements, never submit an ordinary
claim as a fallback. Parent and child job/context binding is still to implement;
the registry's named-service consent alone does not supply it.

## Required complete flow

1. Persist the exact parent/job/child-operation intent. Withdraw new text dispatch,
   drain owned work and freshly prove the offering backend/cache absent. Unknown
   execution or absence retains ownership and refuses transfer.
2. Persist and send the same registry offer key. Bind its acknowledgement to that
   intent. The child persists its matching accepted job/consent before accept.
   Lost replies recover the same identities; acknowledged missing records are
   reconciliation, not permission to submit replacements.
3. Before loading, the child verifies fresh current-root/current-lease admission,
   its own actor/service, resource policy, full envelope and actual headroom.
   It maintains health as the current holder. The parent cannot renew a lease
   now held by the child, load text in parallel or treat the absence of its own
   model as root completion. Parent control remains alive for return/recovery.
4. Generate/publish through the real image job's durable receipt boundaries.
   Disconnect/cancellation cannot substitute for executor absence. Observe child
   backend/cache gone and persist exact handback intent before registry mutation.
5. If the parent remains open, confirm the returned successor and fresh physical
   admission before restoring text. If the parent was explicitly sealed meanwhile,
   do not resurrect it: settle the exact held child under the root end/recovery
   contract. Unknown cleanup or a lost release reply stays durable charged debt.
6. Confirm original parent/grant/call identity, intentional image lineage and text
   handback. A queued foreign requester remains outside the root until its actual
   completion/yield boundary. No time-based eviction or actor substitution.

The native client/tool bridge must discover these service capabilities and carry
authorized job context. A tool name or shared GPU hostname is not enough to infer
a valid handoff. Parent API authorization, child SDK/adapter, native bridge and
failure/restart coverage must ship together before advertising delegation.

## Limits that must stay explicit

The present handoff row freezes its initial `child_id`; this is not a qualified
nested delegation stack. After a deeper transfer/return there can be a different
current successor. Supporting image-to-text-to-image nesting requires exact
frame/ancestry/current-member recovery and logical job authorization, not simply
accepting any same-actor lease or claiming the single-level path is generic.
No nested service support is advertised here.

Required evidence includes source/child crash before and after each ACK, a lost
handback reply, unknown executor/cache absence, parent sealing during child work,
expired held health and same-ID recovery, concurrent child requests, owner/context
mismatch refusal, and no duplicate charged envelope. Current local native vision
tool tests are same-service children; they are not this image-service qualification.
