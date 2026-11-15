# RFC / PRD — model-service capability playground

Status: baseline v1 implemented; first Obelisk validation running. Owner: Codex. Requested by Sam, 4 November 2026. Scope clarification: **useful agent capability first; general model/performance benchmarks later**.

Code: [samcloud-services PR81](https://github.com/samtg-core/samcloud-services/pull/81), model_eval package. Operations: task954. Shared compute remains task940; group-callable image-service remains996. This is tooling inside model-services, not a separate Obelisk product or allocator.

## User outcome

When we add capacity, a model, client or configuration, answer: what useful work can it complete, how much better is it than an existing setup, and where does integration fail? Validate the testing playground on Obelisk first. Then run identical tasks retroactively against the current 35B and v1 Hermes rail configurations. Host health, successful admission and high GPU use do not count as agent capability.

## Baseline v1

Three small jobs with fixed prompts/inputs and independently held acceptance checks:

1. **CSV rollup:** implement a standard-library utility, deliver the requested result and test edge cases. Reuses the actual Obelisk task and its four checks: example, empty, quoting, invalid integers.
2. **Repository repair:** fix a small existing scheduling-window module while preserving API and input immutability. Check empty/nested/touching/negative/separate windows and invalid input types, plus a change handback.
3. **Evidence audit:** reconcile conflicting source files, classify supported/unsupported claims, cite sources and write the next checks. Check exact classifications/schema/citations; explanation quality needs separate human review.

Dimensions are task correctness, requirement adherence, edge-case handling, useful artifact delivery, source grounding, test/tool discipline and unattended completion. No composite ranking from three small jobs. A code-exit zero, a nonempty handback or a claimed test pass is insufficient. Automated presence/citations do not qualify prose quality.

The same task revision, allowed tools and assistance policy define a comparison cohort. Preserve original failures. Existing historical CSV passes are historical evidence, not new passes for all three tasks. Operator repair invalidates unattended success; account for help explicitly.

## Configuration and evidence

Run manifests bind model name/digest/context, hardware/compute resource, backend/gateway/client revisions, configured Samcloud identity, fixture/source/profile hashes, delivery path and tool permissions. Values belong in private deployment JSON, not host-specific code. Preflight verifies the actual requester and installed model digest; Hermes profiles additionally check model/context. Hardware/version labels remain operator declarations where not independently measured.

Each task has a clean workspace and fresh profile carrying only configured config/SOUL/plugin sources. Do not copy sessions, memory, skills, auth caches or cron. Explicit token files feed named process environment variables; no ambient seat-token fallback, secrets in argv or repository. The native account still has OS permissions: controlled workspaces are not an adversarial sandbox.

Save intent before launch, PID/exit, agent log, artifact hashes and acceptance results. A repeated run directory never relaunches an existing task. Changed fixture/config/profile refuses resume. Unknown launch/process state needs inspection of the exact operation. The existing native/queue/rail client retains inference identity and ownership; this controller is not a replacement supervisor. It neither infers absence nor releases physical capacity.

Missing measurements stay absent/unknown. Elapsed time includes queue/tools and is diagnostic, not a capability score or decode rate. Do not infer kernel peak, token efficiency, cleanup or broad model quality from task pass. General throughput/context/concurrency benchmarking follows this baseline.

## Execution paths

V1 invokes a configured existing native client through its normal model-service/queue. A real rail comparison uses the existing rail/delivery client with the same tasks and records its actual receipts; changing a label does not create a rail run. No outer claim around a resident leased service and no embedded second queue client. Direct-backend and general inference benchmark adapters are deferred until the capability baseline is useful.

Current repository main predates some live queue/native releases. Pin those actual deployed sources; installing the standalone harness must not silently upgrade or downgrade a gateway. The GPT-OSS terminal-frame fix and native saved-session binding defect remain separately recorded integration issues, not negative model-quality conclusions.

## Delivery sequence

1. Publish this RFC and maintained model-services source/docs. Baseline implementation at52334d5;12 offline tests pass.
2. Validate on isolated Obelisk: Hermes818c13be / queue client75cc974 / qwen3.5:35b-mlx,65536context, own hermes-obelisk identity and normal resident admission. First run d1516e05-6e93-4c43-9320-b22698b0bf73 is executing, not yet accepted as a task pass.
3. Review the artifacts and agent trace, repair harness defects separately from answers, then run the same cohort on selected Obelisk alternatives. No need to complete a model survey first.
4. Apply the exact cohort to the current35B and v1 Hermes rail setups at ordinary admitted boundaries. Record any differences in tools, profile, assistance or fixture; do not silently compare mismatched jobs. No interruption of another owner's workload to fill a table.
5. Publish a small capability comparison with successful and failed examples. Add broader tasks and general performance testing only when they answer a concrete next question.

## Acceptance and progress

- Offline checks reject hard-coded example-only solutions, invalid repairs, unsupported audit claims, successful process exits without artifacts, profile/config changes and automatic relaunch. A positive full orchestration fixture verifies copied profile isolation and a real checked task result.
- At least one real model-service/native run produces checked useful artifacts under its configured identity, with exact runtime pins and operation/execution receipts.
- Compare command preserves case/configuration/evidence, rather than emitting a misleading aggregate score.
- README, model-service module notes, Obelisk runbook and cloud benchmark ledger link the source/RFC/results and state limitations.
- Rollout and subsequent task dispatch use standing authorized scope. No new host, model download, credential broadening, service migration or human setup step is needed for this baseline.
