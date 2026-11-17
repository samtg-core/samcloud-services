# Resuming an owned host operation before its effects

A host preparation or launch has one durable request identity. An inference
request has its own identity; this mechanism never resubmits inference.

`exo_host_controls.prefix` identifies an explicitly saved read-only prefix:

| Prefix | Reads that may resume | Irreversible boundary |
| --- | --- | --- |
| `prepare` | Prior exact cleanup and settlement, then current own-member admission | The row clears `prefix` transactionally before `HostDeployment.bind` can create files or bind an owner |
| `start` | Current sealed runtime and event epoch, then current own-member authority | `exo_host_rank_attempts.phase` changes from `prepared` to `launch_committed` before any launch intent or `Popen` |

Host metadata reconciliation can wake the supervisor's **existing** operation.
It never creates a new request or starts a second task while that task is live.
Every resumed prefix repeats the real validation; a saved successful read or
the metadata request itself grants no authority. This adds no timer, automatic
reconnection, priority, or default compute deadline.

The start prefix remains pending only for the typed `LaunchPreflightPending`
case, including an authority read superseded by a registry event. A configuration
or boot mismatch, arbitrary exception, partial bind, committed launch, unknown
spawn, fenced attempt, withdrawn authority, or paused service cannot enter this
continuation path. Existing rows gain a nullable column with **no** inferred
prefix: an old incomplete operation stays held for exact recovery.

The ordinary status report exposes `prefix` alongside the original operation's
request ID, state, and exception type. It exposes no raw exception, credential,
prompt, or output. `owned` with a prefix means the read stage is pending; it is
not proof of a worker, admission, readiness, or completed generation.

Controls use the real host HTTP routes, supervisor and SQLite journal, and real
sealed fixture files. They cover persist-before-read, transient historical-read
failure, concurrent observers, unchanged request identity, one binding, no
repeated bind after an unknown effect, no resumed committed or legacy operation,
and an authority event superseding the final read before the same runtime
performs exactly one launch. These are control tests, not distributed numerical
qualification.

## Isolate receipt reads from the gateway's SQLite locks

The receipt collector pins the journal inode with a raw file descriptor. On
Darwin, closing any descriptor for that inode in the same process can drop
SQLite's POSIX locks held by another connection in that process. Moving the
collector to a thread does not isolate those locks.

Local host receipts and historical-cleanup checks therefore use the existing
bounded probe protocol in an isolated standard-library Python child. The
command is fixed service code; the exact receipt specification travels on
stdin, with a fresh nonce and collector hash checked on return. The child gets
an explicit minimal environment with no credential inheritance. All original
inode, ownership, boot, immutable-receipt, SQLite snapshot and size checks stay
in place. A failed observation is still unknown, never cleanup or launch proof.

A real SQLite control holds a parent writer lock and asks a second process to
acquire it before and after the actual receipt observation. The isolated read
preserves that lock; the old in-process collector demonstrably drops it. Other
controls check identical receipt metadata, nonce/source/spec/uid/boot matching,
output bounds, credential exclusion and no automatic observer retry. No model,
fleet request or physical lease is used in these controls.
