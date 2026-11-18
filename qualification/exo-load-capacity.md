# Fixed queue envelopes for the leased EXO placement

The ordinary host queues own the slots. A member claim allocates no OS pages,
so subtracting our own claim from physical telemetry would count it twice.
The leased rank placement must use those same exact envelopes, with fresh
all-member authority, instead of introducing another allocator in EXO.

The first draft required each full member envelope in instantaneous available
RAM. That predicate was wrong for the raised Metal policy and is superseded
here. The live attempt's native HTTP400 compared the whole parent checkpoint
with discovery's current `ram_available`; it was not an actual allocation or
full-window fit failure. Sam restored Slice 62,259 MiB and Wafer 30,720 MiB, and fresh
kernel plus `MTLDevice` queries confirmed those effective recommendations.
Obelisk remains 124,518 MiB. Old successful Wafer trials also reached 26.49 GiB wired
with 1.89 GiB available and no critical pressure: free RAM is useful telemetry,
not an absolute wired-allocation ceiling.

## Before the first load

Model-service retains the original queued inference and all exact member claims.
Its existing finite assembly pass reads each host through the authenticated,
source/configuration/request/identity-bound metadata route. The host checks its
actual boot and reads the current explicit wired limit, physical memory,
kernel wired pages, and the existing canonical accelerator in-use counter.
It imports no Metal/MLX, allocates no model, and changes no kernel policy.

Each queue envelope must be within the explicit wired limit and leave physical
space for the already non-pageable host/device memory. Wired and accelerator
counters may overlap; without an overlap proof the gate conservatively reserves
both. It never subtracts the plane claim or treats pageable CPU/cache memory as
permanently unavailable. This is still a pre-load guard: compressing/swapping
other pageable work can cost time, and only the measured load/full-window peak
can prove the actual working envelope.

A positive shortage publishes `unavailable / exo_rank_memory` and
`assembly_phase: waiting_capacity`. Bad/missing readings remain unknown.
The same call and member claims are kept. Existing resource events resume the
same assembly; no hot loop, watchdog, deadline, release/reacquire or inference
retry is added. Checks run before host starts and immediately before the first
instance-create POST. Already submitted/resident instances are not denied by
counting their own allocations again.

## Native fixed placement

`HostDeployment` supplies the pipeline-ordered `EXO_LEASED_MEMORY_MB` vector
from its frozen group plan, not from public request input or discovery. It is
part of the launch environment/source/configuration pins. The paired native
change validates the exact pinned staged manifest and local loading view,
checks every layer range, and requires each rank's assigned file bytes plus its
full served-window F16 GQA cache to fit its own queue envelope. Extra memory on
another rank cannot cover a short host. Workspace/allocator/process overhead is
not falsely included in that lower-bound calculation; measured peaks remain a
separate required gate.

Only a leased profile with this explicit vector replaces the generic parent
checkpoint/current-free-RAM check. Ordinary automatic placement and legacy
leased profiles retain the existing guard. A changed manifest, context,
placement, invalid/missing cache geometry or short envelope refuses before
`CreateInstance` is sent. This environment is configuration, not authority;
model-service still performs the live exact member checks before starting.

This requires the paired native source and a separately qualified frozen
runtime. Neither candidate is installed. It does not retrospectively clear the
old unknown submission or authorize a start under expired HOLD claims.

## Lifetime is separate

The first qualification's explicit six-hour cap elapsed. Its capacity stays
charged under HOLD, but it is not active start authority. The intended production
lifetime is the active turn/session with renewable authority and positive
cleanup/release. [Core issue 21](https://github.com/samtg-core/samcloud/issues/21)
proposes that lifetime and an explicit, audited atomic continuation after exact
old-worker cleanup. No expired lease was edited, resurrected or reacquired.

## Qualification scope

CPU/offline controls exercise zero-current-free-RAM with valid wired envelopes,
per-rank/cache refusal, foreign locked-memory waits, exact identity/boot checks,
unknown telemetry, preserved calls/claims, and unchanged legacy behaviour. The
unchanged native create method fails the new low-free-RAM regression with the
original HTTP400; the paired change passes. These are software control tests,
not a 235B resident or full-context model qualification.

The paired native source is local commit `2b7d860`, carried in
`patches/exo-queue-memory-envelope.patch` with its digest and software gate
results in `exo-memory-envelope-evidence.json`. The native default-selection
CPU run passes 653 tests (3 existing skips, 190 slow deselections), strict type
checking and Ruff pass, and repository Nix formatting processes 806 files with
zero changes. All three actual staged 235B metadata views agree: remaining
workspace/allocator margins are 0.951 GB on Slice, 2.113 GB on Wafer and 2.379 GB on
Obelisk. These margins are estimates, not measured peak headroom.

The zero-context patch applies with `git apply --unidiff-zero` on the exact
recorded native parent (verified with `--check`). All 537 provider EXO controls
across 49 modules pass, with changed-file Ruff clean. The paired runtime is
still uninstalled; valid group continuation and live allocation remain required.
