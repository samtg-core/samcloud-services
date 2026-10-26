# Native queue components — staged source

These plugins connect Hermes to the adjacent durable inference API. This branch
is a review candidate, not an installation or production-state record. The prior
component notes are preserved in the baseline commit; live deployment state must
be checked independently.

- samcloud-queue-provider registers main execution, main stint, auxiliary and
  originating-tool provider profiles. Every accepted call keeps its exact durable
  grant/execution receipt across observation failure and reconnect.
- samcloud-queue-hooks binds native source ownership through existing hooks and
  middleware. It does not mutate old messages, prompts, memory or model policy.
- samcloud-queue-context uses the stock compressor with an explicitly persisted
  source. Its summary is an inference request, not a pre-existing summary that
  must be produced before admission. Native foreground compaction can bind its
  parent before the first main-model call.

The provider and context plugins require these ollama modules on Python's import
path and matching native Hermes hook/provider interfaces. Configure credentials
through the registered provider's own binding. Do not substitute another seat's
identity or bypass the durable SDK with an anonymous custom transport.

## Parent job opt-in

The explicit owning profile's config.yaml can contain:

```yaml
samcloud_queue:
  job_continuity: true
```

Absent means legacy behavior. This candidate is not yet ready for live activation.
Both main queued-provider types and every Hermes role use the same policy. A
verified native foreground owner links text, originating tools and compaction to
one parent. Concurrent tools have distinct calls under that parent; detached
background work stays independent. Error/interruption retains the parent for
recovery. Only authoritative successful native completion finishes it through
the durable metadata outbox. A tool finishing does not finish the parent.

Legacy sc-queue-wait cannot silently yield a linked parent. Explicit native yield
and registry-authorized cross-service image delegation remain unfinished. In
particular, sharing a job-key string does not authorize another service to use a
physical registry claim. Capability discovery advertises that limitation.

See [the requester-job contract](../docs/requester-jobs.md) for HTTP operations,
physical recovery limits, synthetic qualification and rollback. Preserve old
accepted identities during maintenance; never delete journals or resubmit work
to make a guard pass. No live profile or runtime is changed by these source files.
