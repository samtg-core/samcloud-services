# Native empty-response recovery

Hermes 818c13be adds a request-only assistant `(empty)` / user recovery nudge
after an empty response to tools. SessionDB intentionally excludes both rows.
They can remain between persisted rows after another tool round. Requiring a
row ID for them prevents the native queue client from continuing.

`HeldSource.capture` accepts at most one exact native pair in a main call with
explicit turn and API request IDs. It requires a directly preceding persisted
tool row, both literal native flags, the pinned nudge text and the qualified
empty assistant shape. Changed text, nonempty reasoning, extra fields, invalid
row IDs, a second pair and arbitrary unstored messages still refuse. Other native
scaffolding, including thinking-prefill, is not qualified by this change.

The source retains only genuine positive row IDs. Its call boundary includes a
derived marker at the pair's position in the complete ordered row sequence, plus
the explicit native turn/API identity. This covers persisted rows before and
after the pair without inventing transcript rows or adding a snapshot store.
Reopening the same source reuses the same call; a new API intention is distinct,
and the existing request digest rejects changed input for the same boundary.
Ordinary persisted-only sources retain their existing boundary and keys.

This does not reconstruct a cold native turn from an incomplete transcript.
Reopen evidence supplies the same held history explicitly. Authentication,
foreground/job ownership, accepted-intake fences, compaction lineage and physical
admission policies are unchanged. No fallback is added around a source refusal.

`test_native_empty_recovery` checks middle placement, process reopen, mutation
refusal, ordinary-key compatibility and malformed shapes. The native client test
uses real Hermes recovery, SessionDB, hooks, SDK and file tools over a fake service:
tool → empty → nudge → another tool → final answer, with no synthetic rows saved.
Run that test through the matching Hermes checkout's `scripts/run_tests.sh` with
its qualified test interpreter. Removing the identity delta makes it fail.

The installed legacy client and this job-aware branch have separate source
lineages. Backport the additive identity change and the identical recovery helper;
do not replace the branch module with the legacy copy. A source commit or offline
pass is not a live installation receipt. Adopt at a completed native boundary and
preserve existing failed runs as client failures rather than regrading them.
