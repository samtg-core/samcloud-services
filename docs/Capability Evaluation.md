# RFC / PRD: versioned model-service evaluation baseline

Status: implementing capability-playground v1. Sam clarified useful agent capability first; general inference/performance benchmarks follow. Owner: Codex. Requested by Sam 4 November 2026. Implementation belongs to samcloud-services, not a separate Obelisk benchmark product. Operational discovery remains task954; shared compute lifecycle remains task940; image-service996 retains its own usable-service delivery.

## Outcome

When we add a model, host, client or configuration, run the same versioned tests and compare useful capability and integration reliability. The immediate deliverable is a small baseline, not an exhaustive leaderboard. Sam's interest is a consistent environment and repeatable operation; raw generation acceleration is not the premise.

## Run identity

Every run binds fixture revision/hash, harness commit, compute resource/hardware, backend and model digest/format, served context, requester identity, executor/service attribution and client/config revision. These are configuration values, never special cases in harness code. Secrets stay in explicit private token files, out of argv, reports and repository. Claimed pins and measured/verified pins are distinguished.

## Primary capability baseline (clarification)

The first delivery is a testing playground of useful agent jobs: implementation with edge cases, repair of an existing repository, and source-grounded operational reasoning. Each gets a clean controlled workspace, the same task, independently held acceptance checks and explicit intervention accounting. Score requirement adherence, task correctness, edge cases, useful artifacts and unattended completion. Infrastructure health and timing are diagnostic context, not the primary outcome.

Validate the playground on Obelisk first. Reuse it retroactively on current 35B and v1 Hermes rail configurations using matched task revisions. Existing historical CSV passes remain labelled historical; they cannot substitute for running the other tasks. Model/client/config/path are comparison dimensions. General inference, throughput, context stress and concurrency benchmarks come after this baseline.

## Common dimensions

Report per fixture: exact correctness or explicit task oracle, tool-call validity and useful artifact checks, input/context size, provider-reported token usage, observed first payload and completion time, queue/reconnect state, memory/pressure evidence and cleanup/return evidence. Missing measurements are null, not zero. Queue-inclusive time is not decode speed; sampled residency is not kernel peak; model fitting is not useful task success. Do not invent a composite agent score from unlike tests.

Existing material to reuse now:
- The handed-over deterministic arithmetic/JSON/code/sequence prompts in bench_fixture.py, with strict checks and no silent answer cleanup.
- Native Hermes CSV rollup with the four independent checks already used for 35B/122B; preserve failed GPT-OSS and old harness attempts.
- Existing model-service durable consumer and its accepted-ID/reconnect semantics, rather than a second queue client.
- Existing shared process/claim lifecycle for any direct physical work; no outer claim around a resident model-service.

## Execution paths

1. Service baseline v1: authenticated caller -> existing durable model-service queue -> backend. Persist run/call keys before submit, preserve accepted IDs, reconnect the same execution, stop later dispatch on an actual failed case. An observer disconnect never authorizes regeneration or release.
2. Agent evidence: import exact existing native task outcomes with source/artifact hashes and provenance; distinguish imported historical observations from new execution. Next run the same useful task through a configured native client. The current native saved-session binding defect remains a separate integration failure, not a negative model-quality score.
3. Direct/backend comparison: supported adapter later, same fixtures and dimensions, retaining physical compute admission. V1 must refuse unsupported execution modes rather than bypass the queue or pretend a direct comparison exists.
4. Rail path: the same worker/config/fixtures through standard Samcloud delivery. Transport/worker configuration is another recorded dimension; commissioning a rail is not a prerequisite for measuring the model-service baseline.

## Baseline implementation and delivery

A versioned model_eval package, private deployment JSON plus generic example, CLI run/compare, durable local records, independent task checkers and concise README in samcloud-services. It invokes a configured native agent/client; that client uses the existing shared model-service/queue lifecycle, with no embedded allocator or replacement queue. Current repository main predates the live queue source: record the runtime pin rather than silently copying another allocator/client into the harness.

First execution on isolated Obelisk: the passed 35B model performs the three useful capability jobs. Subsequent configurations use those exact tasks; GPT-OSS native resume and tool-frame regression remain separate integration repairs. Keep 122B's completed native CSV task as existing evidence, not a new unmeasured performance claim. Use serial runs and normal service-owned admission/cooldown. No model downloads, new groups, external providers or host-wide scheduling changes required for this baseline.

Then capture current Slice/Wafer agent/compute configurations using the same schema and fixtures at their ordinary admitted boundaries. Imported results remain labelled; comparable fresh runs require the same fixture/context/settings. Do not interrupt other owners to fill a table. Context expansion, sustained load, mixed concurrency and long-session recovery become separate tests once the baseline works.

## Acceptance

- Offline tests prove exact fixture checks, final-frame tool handling, no fabricated measurements, immutable config binding, lost-reply/same-key recovery and failure stopping later cases.
- A real authenticated Obelisk run produces versioned manifest, accepted execution IDs and checked results under its own identity; run/compare commands work from documented configuration.
- Model-service README/module notes describe the harness, current deployment source differences and the evidence limits. Cloud RFC, Obelisk runbook and benchmark ledger link code/results.
- Failures stay in the comparison with their layer: configuration, transport, service protocol, backend, model task or native-client integration. Existing successful results are not erased by later failures.

## Boundaries and decisions

No new allocator, host-specific fork, benchmark-driven global upgrade or automatic model eviction. No arbitrary compute deadline or retry-as-new-work. Unknown/uncertain work retains its accepted identity. Direct mode, context/concurrency stress and a full rail campaign are subsequent adapters/cases, not claims of this first delivery. Group-callable image-service remains an operational product requirement and is not replaced by this harness.
