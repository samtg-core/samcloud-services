# Durable queue HTTP shutdown

The `python -m ollama.server` entry point uses `QueueServer` to detach HTTP
observers before Uvicorn waits for their connections. A queued compatibility
request can otherwise hold shutdown indefinitely: it waits for a saved result,
but the native worker has already drained and admission is paused. ASGI lifespan
shutdown happens after Uvicorn's connection wait, so closing observers there is
too late.

On shutdown, the runtime fences dispatch and closes observation. A pending
non-streaming response returns HTTP 503 with `service_restarting` and its existing
execution ID. A pending SSE/NDJSON observer gets the same typed error and closes;
it does not get an invented successful finish. Resume reading that exact saved
execution rather than making a new submission. Already saved terminal results
remain readable. Authentication and owner isolation are unchanged.

This does not cancel workers, change durable jobs/grants, release capacity, or
infer backend death. The existing lifespan still waits for known compute and
refuses manager shutdown on unresolved ownership. Ordinary physical-resource
handoffs do not close observation. There is no new workload or graceful-shutdown
deadline.

Other ASGI launch methods do not automatically select `QueueServer`; they need
the equivalent hook before connection draining. Merely importing `server:app`
does not enable it. A source upgrade also cannot change an already running old
process that is waiting for connections. Its current deployment remains the
operator's responsibility; this change is not a prerequisite for adopting PR94.

CPU validation includes a real loopback HTTP connection with the installed
Uvicorn 0.54.0, queued journal reopen, a blocked fake worker completing once,
uncertain-claim retention, successful saved-stream replay, and ordinary handoff.
The old server fails the loopback shutdown check. Candidate qualification covers
224 related tests; no native provider, external network, subprocess or signal is
used. Native or fleet shutdown qualification is not claimed.
