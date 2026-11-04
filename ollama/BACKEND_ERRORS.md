# Ollama backend errors

The generation client raises `OllamaBackendError` for HTTP error responses
and error-only JSON records inside an HTTP 200 stream. Synchronous `chat` and
`generate` and the asynchronous iterator use the same diagnostic fields:

| Field | Meaning |
| --- | --- |
| `status` | HTTP status; 200 for an error inside an accepted stream |
| `phase` | `status` or `in_stream` |
| `chunks_before` | Parsed records yielded before the error, not a token count |
| `subtype` | Fixed prefix classification, otherwise `unclassified` |
| `detail_len` | Byte length of the captured detail, not necessarily the full response |
| `detail_sha256` | First 12 hex characters of the SHA-256 of at most 4096 detail bytes |
| `detail_truncated` | Capture was incomplete or reached the byte limit |

Recognized prefixes are `model runner has unexpectedly stopped` (`runner_stopped`),
`error parsing tool call` (`tool_parse`),
`an error was encountered while running the model` (`model_runtime`), and
`context canceled` (`context_cancelled`). These literal strings were checked
against Ollama 0.33.0. A tag identifies a message family, not its root cause.
Unknown, malformed and truncated status bodies stay `unclassified`.

Status-body capture is limited to 4096 bytes. The asynchronous path gives that
capture two seconds, independently of the existing generation deadlines.
The synchronous path retains its existing HTTP per-read timeout; it does not
gain a two-second overall capture deadline. A truncated hash describes only
the captured prefix and cannot identify the complete backend response.

Raw backend detail can contain generated text. It is neither stored on the
exception nor included in its string, diagnostic logs or failure frames.
The client logs safe fields so direct iterator consumers have diagnostics;
the OpenAI streaming route also logs those fields with elapsed seconds.
This can produce two diagnostic lines for one streaming failure.

The OpenAI chat stream emits a `backend_error` failure and `[DONE]`, preserving
any earlier content. `[DONE]` closes the response; it does not turn the failure
into a successful result. Consumers must inspect the error frame.

The Ollama-backed `/v1/completions` stream retains its NDJSON format. On a typed
backend error it emits a final safe line with `error: "backend_error"`,
`diagnostics`, `done: true`, and `done_reason: "error"`. The error remains explicit even after partial
output; `done` closes the transport and does not acknowledge compute cleanup.

Neither an HTTP error, a subtype nor `[DONE]` proves backend termination,
process absence or returned capacity. Durable execution adapters must retain
uncertainty and ownership until their existing reconciliation rules establish
otherwise. This change adds no automatic retry, lease release or admission
policy. Non-streaming chat and completions collection propagate the typed error
as HTTP 500 rather than manufacturing a result. Previously an in-band error could
be returned as truncated successful chat content or a raw error dictionary with
HTTP 200. These non-streaming routes do not yet return the safe diagnostic fields
in their HTTP error body; those fields are available in the client log.

Run the offline client-to-SSE tests with:

```sh
python -m unittest ollama.test_backend_error_diagnostics
python -m ollama.test_stream_deadlines
python ollama/test_ollama_terminal_delta.py
```

When porting this change to a separately pinned gateway, preserve that
gateway's existing pre-dispatch refusal handling and service-owned execution
semantics. Replacing its entire client or server with an older shared version
is not a safe backport.
