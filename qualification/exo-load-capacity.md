# Hold the group while physical memory becomes available

Ordinary host-queue claims and physical RAM headroom are separate observations.
An admitted exclusive member protects its slot from another queued GPU load.
It does not prove that desktop applications and other host processes have left
enough unified memory for its assigned model rank.

Model-service retains the original queued inference and every exact member
claim while it waits for the full pinned per-rank memory envelopes. Its existing
assembly owner reads each configured host's canonical capacity collector through
an authenticated, source/configuration/identity/boot-bound metadata route. Each
rank must fit independently; spare Obelisk memory cannot cover a short Wafer.

Positive insufficient headroom publishes `unavailable / exo_rank_memory` and
`assembly_phase: waiting_capacity`. Missing or invalid evidence remains unknown.
Neither case fails inference, releases a member, changes queue position, starts
a second watchdog, nor adds an assembly/compute deadline. Existing host/resource
events can resume the same finite assembly pass. It repeats all actual reads.

The gate runs after preparation and before the first host process is started,
then again after host-start verification and before the first instance-create
submission. Once an instance has been submitted, its normal once-only startup
and readiness path remains in charge: a lost response is not replayed, and RAM
held by an already resident model is not mistaken for foreign unavailability.

This is a final provider admission guard. It does not change the registry's
static host capacity, ordinary lease promotion, renewal limits or HOLD cleanup
rules. Publishing fresh physical capacity into ordinary host-queue admission
and continuous claim lifetime are separate contracts; this guard must not be
presented as having implemented them. Existing legacy unknown submissions are
not retrospectively declared safe or cleared by this change.

Controls cover a short Wafer with spare memory on the other ranks, preserving
the original call and all claims over repeated observations, no start/instance
POST before fit, a drop during host launch, same-operation recovery on a fresh
external wake, no hot retry, and no denial of an already resident instance.
Host ASGI controls check fresh readings, exact caller/identity and failed reads.
These are offline control tests, not physical residency qualification.

## Qualification

529 offline EXO controls across 48 modules pass on this change. No model,
physical lease or fleet mutation is performed by that run. The fixed-session
endpoint and full source/configuration pins change and require a separately
qualified deployment; this is not installed into an active group.
