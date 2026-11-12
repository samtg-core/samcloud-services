"""Ollama API client for model management."""

import re
import time
import httpx
import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Iterator

from . import config

OLLAMA_BASE = config.OLLAMA_BASE

log = logging.getLogger("ollama-client")


ERROR_BODY_LIMIT = 4096
ERROR_BODY_TIMEOUT = 2.0
# These literal prefixes are present in the installed 0.33.0 backend. Anything
# else remains unclassified; model-generated text is never a diagnostic tag.
_ERROR_PREFIXES = (
    ("model runner has unexpectedly stopped", "runner_stopped"),
    ("error parsing tool call", "tool_parse"),
    ("an error was encountered while running the model", "model_runtime"),
    ("context canceled", "context_cancelled"),
)
# 0.33.0 raises tool_parse only in the harmony parser's done branch
# (harmony/harmonyparser.go:445), and the chat route reports it after the
# runner's Completion call has returned (server/routes.go:2892). Unlike every
# other tag it is affirmative evidence that generation ended.
_COMPLETED_GENERATION = frozenset({"tool_parse"})
_TOOL_PARSE_PREFIX = "error parsing tool call"
_TOOL_PARSE_STATUS_HEAD = b'{"error":"' + _TOOL_PARSE_PREFIX.encode()
# Go encoding/json messages, so the shape of a rejected tool call is recorded
# without its text: a fixed kind and the class of the offending character.
_GO_JSON_ERRORS = (
    (re.compile(r"unexpected end of JSON input$"), "eof"),
    (re.compile(r"invalid character (.+) looking for beginning of value$"), "not_json"),
    (re.compile(r"invalid character (.+) after top-level value$"), "trailing_data"),
    (re.compile(r"invalid character (.+) after object key(?::value pair)?$"), "bad_separator"),
    (re.compile(r"invalid character (.+) looking for beginning of object key string$"), "bad_key"),
    (re.compile(r"invalid character (.+) in string literal$"), "control_in_string"),
    (re.compile(r"invalid character (.+) in string escape code$"), "bad_escape"),
    (re.compile(r"invalid character (.+) in numeric literal$"), "bad_number"),
    # Valid JSON that is not an object fails in wk8/go-ordered-map v2.1.8 via
    # buger/jsonparser.ObjectEach, after encoding/json's own syntax check.
    (re.compile(r"Value looks like object, but can't find closing '\}' symbol$"), "not_object"),
)


def _char_class(c):
    if not c:
        return "none"
    if c in "{}":
        return "brace"
    if c in "[]":
        return "bracket"
    if c in "\"'`":
        return {"\"": "quote", "'": "apostrophe", "`": "backtick"}[c]
    if c == "\\":
        return "backslash"
    if c in "\n\r\t":
        return "newline_or_tab"
    if c.isspace():
        return "space"
    if c.isalpha():
        return "letter"
    if c.isdigit():
        return "digit"
    if not c.isprintable():
        return "control"
    return "punct"


def _go_quoted_char(token):
    # Go's quoteChar: 'x', '\'', '\\', '\n', '\x01' or '\u00a0'.
    inner = token[1:-1] if len(token) >= 2 and token[0] == token[-1] == "'" else ""
    if inner.startswith("\\") and len(inner) > 1:
        return {"n": "\n", "r": "\r", "t": "\t", "'": "'", "\\": "\\"}.get(inner[1], "\x00")
    return inner[:1]


def _tool_parse_shape(detail):
    """Content-free shape of `error parsing tool call: raw='...', err=...`."""
    body = detail[len(_TOOL_PARSE_PREFIX):]
    head, sep, err = body.rpartition("', err=")
    raw = head[len(": raw='"):] if sep and head.startswith(": raw='") else None
    shape = {"raw_len": None if raw is None else len(raw),
             "raw_first": _char_class(raw.lstrip()[:1]) if raw is not None else "unknown",
             "raw_last": _char_class(raw.rstrip()[-1:]) if raw is not None else "unknown",
             "json_error": "unknown", "char": "unknown"}
    if raw is None:
        return shape
    for pattern, kind in _GO_JSON_ERRORS:
        m = pattern.search(err)
        if m:
            shape["json_error"] = kind
            if m.groups():
                shape["char"] = _char_class(_go_quoted_char(m.group(1)))
            else:
                shape["char"] = "none"
            return shape
    shape["json_error"] = "other"
    return shape


class OllamaBackendError(Exception):
    """Safe provider diagnostics, NOT evidence that execution/compute ended."""
    def __init__(self, status, phase, chunks_before, detail=b"", *, truncated=False):
        raw = detail.encode("utf-8", errors="replace") if isinstance(detail, str) else bytes(detail)
        self.status, self.phase, self.chunks_before = status, phase, chunks_before
        self.detail_len = len(raw)
        self.detail_truncated = truncated or len(raw) > ERROR_BODY_LIMIT
        bounded = raw[:ERROR_BODY_LIMIT]
        self.detail_sha256 = hashlib.sha256(bounded).hexdigest()
        self.subtype = "unclassified"
        if isinstance(detail, str) and not self.detail_truncated:
            self.subtype = next((tag for prefix, tag in _ERROR_PREFIXES
                                 if detail.lower().startswith(prefix)), "unclassified")
        # A tool call's arguments can exceed the retained bound. The fixed prefix
        # comes first, so it is still identifiable; the shape is then unknown.
        elif isinstance(detail, str) and detail.startswith(_TOOL_PARSE_PREFIX):
            self.subtype = "tool_parse"
        elif phase == "status" and raw.startswith(_TOOL_PARSE_STATUS_HEAD):
            self.subtype = "tool_parse"
        self.parse_shape = None
        if self.subtype == "tool_parse":
            self.parse_shape = (_tool_parse_shape(detail) if isinstance(detail, str)
                                and not self.detail_truncated else
                                {"raw_len": None, "raw_first": "unknown", "raw_last": "unknown",
                                 "json_error": "unknown", "char": "unknown"})
        super().__init__(f"Ollama backend error: status={status} phase={phase} "
                         f"subtype={self.subtype} chunks_before={chunks_before}")

    @property
    def completed_generation(self):
        return self.subtype in _COMPLETED_GENERATION

    def diagnostics(self):
        out = dict(status=self.status, phase=self.phase, subtype=self.subtype,
                   chunks_before=self.chunks_before, detail_len=self.detail_len,
                   detail_sha256=self.detail_sha256[:12], detail_truncated=self.detail_truncated)
        if self.parse_shape is not None:
            out["parse_shape"] = dict(self.parse_shape)
        return out


def _backend_error(status, phase, chunks_before, body, *, truncated=False, parsed=False):
    detail = body
    if not parsed:
        try:
            value = json.loads(body)
            if isinstance(value, dict) and isinstance(value.get("error"), str):
                detail = value["error"]
        except (ValueError, TypeError):
            pass
    if not isinstance(detail, (str, bytes, bytearray)):
        detail = b""
    error = OllamaBackendError(status, phase, chunks_before, detail, truncated=truncated)
    # The durable queue may bypass server.stream's compatibility framing.
    # Keep the same safe diagnostic available on that path as well.
    log.warning("Ollama backend_error %s", json.dumps(error.diagnostics(), sort_keys=True))
    return error


async def _async_status_error(resp):
    import asyncio
    body = bytearray()
    async def read_bounded():
        while len(body) < ERROR_BODY_LIMIT:
            part = await resp.content.read(ERROR_BODY_LIMIT - len(body))
            if not part:
                break
            body.extend(part)
    incomplete = False
    try:
        await asyncio.wait_for(read_bounded(), timeout=ERROR_BODY_TIMEOUT)
    except Exception:
        incomplete = True
    return _backend_error(resp.status, "status", 0, bytes(body),
                          truncated=incomplete or len(body) >= ERROR_BODY_LIMIT)


def _sync_status_error(resp):
    # Preserve the existing httpx per-read timeout. Bound retained bytes; unlike
    # the async path, this read uses the synchronous request's configured timeout.
    body = bytearray()
    try:
        for part in resp.iter_bytes(chunk_size=1024):
            body.extend(part[:ERROR_BODY_LIMIT-len(body)])
            if len(body) >= ERROR_BODY_LIMIT:
                break
    except Exception:
        return _backend_error(resp.status_code, "status", 0, bytes(body), truncated=True)
    return _backend_error(resp.status_code, "status", 0, bytes(body),
                          truncated=len(body) >= ERROR_BODY_LIMIT)


def _checked_chunk(chunk, seen):
    if isinstance(chunk, dict) and "error" in chunk and "message" not in chunk and "done" not in chunk:
        raise _backend_error(200, "in_stream", seen, chunk["error"], parsed=True)
    return chunk


class OllamaRefused(Exception):
    """Complete 400/404 JSON refusal received BEFORE any generation frames.

    Not a timeout, load error, 5xx or an in-stream error. Error bodies can contain
    prompts/paths, so only status and fixed safe copy cross the queue boundary.
    https://docs.ollama.com/api/errors
    """
    def __init__(self, status):
        if status not in (400, 404):
            raise ValueError('not a qualified Ollama refusal')
        self.result = {'kind': 'error', 'status': status, 'code': 'ollama_refused',
            'detail': 'Ollama rejected an invalid request before streaming' if status == 400
                      else 'Ollama could not find the requested model'}
        super().__init__(self.result['detail'])


def _raise_complete_refusal(status, raw):
    if status not in (400, 404):
        return
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError):
        return
    if isinstance(body, dict) and isinstance(body.get('error'), str) and body['error']:
        raise OllamaRefused(status)


class OllamaStalled(Exception):
    """A generation went silent past its deadline (#904).

    Carries which deadline fired, because two stalls with the same symptom need
    telling apart in a log a week later: `first_token` is a prefill that never
    produced, `inter_token` is a decode that stopped mid-answer. Same shape as
    `ExoStalled`, so the two backends report a stall the same way.
    """

    def __init__(self, detail: str, tokens: int = 0, silent_for: float = 0.0,
                 phase: str = "inter_token", deadline: float = 0.0):
        super().__init__(detail)
        self.detail = detail
        self.tokens = tokens
        self.silent_for = silent_for
        self.phase = phase
        self.deadline = deadline


class OllamaRequestTimeout(Exception):
    """The whole-request budget ran out (#904).

    Distinct from OllamaStalled on purpose. Both used to surface as the same
    bare `asyncio.TimeoutError`, and a single `except` cannot tell them apart —
    which is how 17 requests in one day were logged as `Stream error for
    qwen3.8:27b-mlx:` with nothing after the colon. A request that was
    producing tokens the whole time and simply ran long is not a stall, and
    saying so is the difference between "raise the budget" and "the model is
    stuck".
    """

    def __init__(self, detail: str, tokens: int = 0, elapsed: float = 0.0,
                 deadline: float = 0.0):
        super().__init__(detail)
        self.detail = detail
        self.tokens = tokens
        self.elapsed = elapsed
        self.deadline = deadline

# Approximate VRAM requirements (MB) for common model sizes.
# Ollama reports actual size once pulled - these are planning estimates.
MODEL_MEMORY_ESTIMATES = {
    "1b": 1200,
    "3b": 2500,
    "7b": 5000,
    "8b": 5500,
    "13b": 8500,
    "14b": 9500,
    "32b": 20000,
    "70b": 42000,
    "72b": 44000,
}


# Match a parameter-count tag only on a digit boundary. Plain substring
# matching read "qwen3.8:27b-mlx" as 7b and leased 5000MB for a model that
# actually resides at 17530MB — the registry under-counted GPU memory by 3.5x,
# so a second service could be told there was room that did not exist.
# "13b" hit "3b" the same way. Anchor it.
_SIZE_TAG_RE = re.compile(r"(?<![0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])")


def estimate_memory_mb(model_name: str) -> int:
    """Estimate memory needed based on model name/size tag.

    Only a fallback — prefer OllamaClient.memory_estimate_mb(), which reads the
    real weight size from Ollama instead of guessing from the name.
    """
    name = model_name.lower()
    m = _SIZE_TAG_RE.search(name)
    if m:
        raw = m.group(1)
        # "1.0" -> "1", but never turn "70" into "7".
        tag = f"{raw[:-2] if raw.endswith('.0') else raw}b"
        if tag in MODEL_MEMORY_ESTIMATES:
            return MODEL_MEMORY_ESTIMATES[tag]
        # Unlisted size: interpolate at ~0.65 GB per billion params (Q4-ish).
        try:
            return max(1024, round(float(m.group(1)) * 650))
        except ValueError:
            pass
    # Default conservative estimate for unknown models
    return 4000


@dataclass
class OllamaClient:
    base_url: str = OLLAMA_BASE
    _http: httpx.Client = field(default=None, repr=False)
    # Models whose configured num_ctx we have already warned about clamping,
    # so a typo is reported once rather than on every request.
    _clamped: set = field(default_factory=set, repr=False)

    def __post_init__(self):
        # Used for non-streaming requests only (health, list, show, etc.)
        self._http = httpx.Client(base_url=self.base_url, timeout=600)

    def version(self) -> str:
        r = self._http.get("/api/version")
        r.raise_for_status()
        return r.json().get("version", "unknown")

    def memory_estimate_mb(self, model_name: str) -> int:
        """Memory to lease for a model, from its real weight size where known.

        Ollama reports on-disk size in /api/tags; resident size tracks it
        closely (18GB on disk -> 17530MB resident for qwen3.8:27b-mlx). That
        beats guessing from the name, which is only used for models we have
        not pulled yet.
        """
        try:
            for m in self.list_models():
                name = m.get("name", "")
                if name == model_name or name == f"{model_name}:latest":
                    size = int(m.get("size", 0))
                    if size > 0:
                        return max(1024, round(size / 1024 / 1024))
        except Exception:
            pass
        return estimate_memory_mb(model_name)

    # {(name, digest): native_context_length|None} — one /api/show per model
    # per digest. `/api/tags` does not carry the window, so /api/show is the
    # only source, and a model's native window only changes when the weights
    # do, which the digest catches.
    _native_ctx: dict = field(default_factory=dict, repr=False)
    # {name: digest} with an expiry, so resolving a digest costs one /api/tags
    # per NATIVE_CTX_TTL_S and not one per model per call. This matters: the
    # clamp below is reached from `offering()`, which serves the auth-exempt
    # `/warm` for every model in the catalogue. Per-call tag reads there would
    # turn an anonymous request rate into an Ollama request rate — the same
    # mistake `/warm`'s own docstring records about subprocess spawns.
    _digests: dict = field(default_factory=dict, repr=False)
    _digests_at: float = 0.0

    NATIVE_CTX_TTL_S = 300.0

    def _digest_of(self, model_name: str) -> str:
        now = time.monotonic()
        if now - self._digests_at > self.NATIVE_CTX_TTL_S or not self._digests:
            try:
                self._digests = {
                    m.get("name", ""): m.get("digest", "")
                    for m in self.list_models()
                }
                self._digests_at = now
            except Exception as e:
                log.debug(f"digest refresh failed: {e}")
                # Keep whatever we had. A stale digest costs a stale window,
                # which is far cheaper than re-reading /api/show on every call
                # because the catalogue briefly would not answer.
                self._digests_at = now
        return (self._digests.get(model_name)
                or self._digests.get(f"{model_name}:latest") or "")

    def native_context_length(self, model_name: str) -> Optional[int]:
        """The window the WEIGHTS declare, from /api/show. None if unreadable.

        The key is architecture-prefixed (`qwen3_5.context_length`), so match on
        the suffix rather than naming an architecture we would then have to
        keep a list of.
        """
        key = (model_name, self._digest_of(model_name))
        if key in self._native_ctx:
            return self._native_ctx[key]
        ctx = None
        try:
            info = self.show_model(model_name).get("model_info", {}) or {}
            for k, v in info.items():
                if k.endswith(".context_length") and isinstance(v, int) and v > 0:
                    ctx = v
                    break
        except Exception as e:
            log.debug(f"native_context_length({model_name}): {e}")
        self._native_ctx[key] = ctx
        return ctx

    def num_ctx_for(self, model_name: str) -> Optional[int]:
        """The num_ctx we will ask for, clamped to what the weights declare.

        Returns None when nothing is configured, which means "send no num_ctx
        and let Ollama derive one" — see `config.OLLAMA_NUM_CTX`.

        The clamp is not politeness. A configured value above the native window
        is a typo or a stale copy of another box's setting, and Ollama answers
        it by allocating for a window the weights cannot address; refusing to
        pass it on keeps a config mistake from becoming a memory one. We log it
        once per model so the typo is still visible.
        """
        want = config.ollama_num_ctx(model_name)
        if want <= 0:
            return None
        native = self.native_context_length(model_name)
        if native and want > native:
            if model_name not in self._clamped:
                self._clamped.add(model_name)
                log.warning(
                    f"num_ctx for {model_name}: configured {want} exceeds the "
                    f"model's native window {native} — serving {native}"
                )
            return native
        return want

    def _with_num_ctx(self, model_name: str, payload: dict) -> dict:
        """Merge our num_ctx into an Ollama payload's `options`.

        EVERY path that touches a model goes through here, because Ollama keys
        a loaded instance by its options: a request whose num_ctx differs from
        the resident instance's RELOADS the model. One call site that forgets
        is not a cosmetic inconsistency, it is a 30s reload in the middle of
        somebody's job, and `/api/ps` would then report a window the next
        request does not get.

        A caller's own explicit num_ctx is left alone — `/models/load` takes
        one, and an operator naming a window should get the one they named.
        """
        n = self.num_ctx_for(model_name)
        if n is None:
            return payload
        opts = dict(payload.get("options") or {})
        opts.setdefault("num_ctx", n)
        payload["options"] = opts
        return payload

    def running_context_length(self, model_name: str) -> Optional[int]:
        """The window the RESIDENT instance actually has, from /api/ps.

        This is the only authoritative answer: it is what Ollama decided at
        load time, whether we pinned it or it derived one from free VRAM.
        """
        try:
            for m in self.list_running():
                name = m.get("name", "")
                if name == model_name or model_name in name:
                    ctx = m.get("context_length")
                    return int(ctx) if ctx else None
        except Exception:
            pass
        return None

    def list_models(self) -> list[dict]:
        r = self._http.get("/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])

    def list_running(self) -> list[dict]:
        r = self._http.get("/api/ps")
        r.raise_for_status()
        body = r.json()
        if not isinstance(body,dict) or not isinstance(body.get('models'),list):
            raise ValueError('Ollama inventory is unknown')
        return body['models']

    def pull_model(self, model: str, stream: bool = True) -> Iterator[dict]:
        """Pull a model. Yields progress dicts if stream=True."""
        with self._http.stream(
            "POST", "/api/pull", json={"model": model, "stream": stream}, timeout=None
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def load_model(self, model: str, keep_alive: str | int = "1h",
                   num_ctx: Optional[int] = None, *, service_owned=False) -> dict:
        """Load a model into memory without generating (warm-up).
        keep_alive: duration string ("1h", "30m") or -1 for indefinite.

        `num_ctx` overrides the configured window for this one call. It exists
        for the ADOPTION path, which re-applies keep_alive to a model somebody
        else already loaded: passing that instance's own window makes the
        re-apply a no-op, where passing ours would reload a model mid-job to
        change a number. Everywhere else leaves it None and gets the config.
        """
        payload = {"model": model, "prompt": "", "keep_alive": keep_alive}
        if num_ctx and num_ctx > 0:
            payload["options"] = {"num_ctx": int(num_ctx)}
        else:
            payload = self._with_num_ctx(model, payload)
        if service_owned:
            payload['stream'] = False
        r = self._http.post("/api/generate", json=payload,
            timeout=self._owned_stream_timeout if service_owned else 300)
        r.raise_for_status()
        return r.json()

    def unload_model(self, model: str, *, service_owned=False) -> dict:
        """Unload a model from memory by setting keep_alive to 0."""
        payload = {"model": model, "prompt": "", "keep_alive": "0"}
        if service_owned:
            payload['stream'] = False
        r = self._http.post(
            "/api/generate",
            json=payload,
            timeout=self._owned_stream_timeout if service_owned else 60,
        )
        r.raise_for_status()
        return r.json()

    # UNCHANGED, AND THAT IS THE POINT (#904). httpx `read` is a per-chunk
    # IDLE bound — 300s of silence, not 300s of elapsed time — so this line was
    # always the right shape and needs nothing. The async methods below were
    # written by transcribing this `300` into aiohttp's `total=`, which is
    # wall-clock: the intent was always "300s of silence" and the SHAPE was
    # lost in the httpx -> aiohttp translation. That is the whole bug.
    #
    # Left at 300 and NOT wired to the new constants. It covers a prefill as
    # one long read, so the inter-token budget would be the wrong value here —
    # and changing a correct line to look like the fix is how the next reader
    # loses track of which one was broken.
    _stream_timeout = httpx.Timeout(connect=10, read=300, write=10, pool=10)
    # Only an admitted durable worker selects this. Its lifetime is independent
    # of consumer sockets, with ambiguous termination quarantined by the queue.
    _owned_stream_timeout = httpx.Timeout(connect=10, read=None, write=10, pool=10)

    def generate(self, model: str, prompt: str, *, service_owned=False, **kwargs) -> Iterator[dict]:
        """Generate completion, streaming."""
        payload = self._with_num_ctx(
            model, {"model": model, "prompt": prompt, "stream": True, **kwargs})
        with self._http.stream(
            "POST", "/api/generate", json=payload,
            timeout=self._owned_stream_timeout if service_owned else self._stream_timeout
        ) as resp:
            if resp.status_code in (400, 404):
                _raise_complete_refusal(resp.status_code, resp.read())
            if resp.status_code >= 400:
                raise _sync_status_error(resp)
            resp.raise_for_status()
            seen = 0
            for line in resp.iter_lines():
                if line.strip():
                    yield _checked_chunk(json.loads(line), seen)
                    seen += 1

    def chat(self, model: str, messages: list[dict], *, service_owned=False, **kwargs) -> Iterator[dict]:
        """Chat completion, streaming (sync, for non-streaming collection)."""
        payload = self._with_num_ctx(
            model, {"model": model, "messages": messages, "stream": True, **kwargs})
        with self._http.stream(
            "POST", "/api/chat", json=payload,
            timeout=self._owned_stream_timeout if service_owned else self._stream_timeout
        ) as resp:
            if resp.status_code in (400, 404):
                _raise_complete_refusal(resp.status_code, resp.read())
            if resp.status_code >= 400:
                raise _sync_status_error(resp)
            resp.raise_for_status()
            seen = 0
            for line in resp.iter_lines():
                if line.strip():
                    yield _checked_chunk(json.loads(line), seen)
                    seen += 1

    async def _stream_with_deadlines(self, path: str, payload: dict,
                                     prompt_tokens: Optional[int] = None, *, service_owned=False):
        """POST a streaming Ollama request under three separate deadlines (#904).

        WHY NOT aiohttp's own `total=`, which is what this used to be.
        `total` bounds ELAPSED TIME, so it kills a generation for being long —
        and being long is the one thing that is not a fault.

        Measured on slice 2026-10-08, counting `Stream error` in `server.log`
        and NOT Ollama's status column: **18 cut-offs out of 81 `/api/chat`
        requests, 22%**. The status column undercounts, because our abort races
        the response completion: 11 of the 18 were recorded `500 | 5m0s` and
        **6 were recorded `200 | 5m0s`** — the same event, logged as a success.
        The longest request that actually COMPLETED today ran 285.0s, which is
        **15 seconds** under the wall. The cap was the failure, and not a rare
        one.

        What actually wants bounding is SILENCE, and silence means two different
        things at two points in a request:

          before the first token   prefill, legitimately silent, for longer the
                                   longer the prompt — so the budget scales with
                                   the prompt (`ollama_first_token_deadline`)
          between tokens           a gap here is a stopped generation, not a
                                   slow one, so it can be tight
                                   (`OLLAMA_STALL_TIMEOUT`)

        A single silence budget cannot be both: it must cover the worst prefill,
        which makes it useless at catching a stall during decode.

        The whole-request budget is kept as a leak bound and tracked HERE rather
        than handed to aiohttp, because aiohttp's `total` and our own
        `wait_for` both raise `asyncio.TimeoutError` and a single `except`
        cannot tell them apart. Owning it makes "ran long" and "went silent"
        different facts with different exceptions — which is the whole point,
        since they call for opposite responses.

        Same structure, constants and reasoning as `ExoClient.chat_stream`
        (#830). Asserted by `test_stream_deadlines`.
        """
        import aiohttp
        import asyncio
        session = aiohttp.ClientSession()
        try:
            async with session.post(
                f"{self.base_url}{path}",
                json=payload,
                # `total=None` is NOT "no ceiling". OLLAMA_GENERATE_TIMEOUT
                # is a real ceiling, enforced in the loop below — because
                # aiohttp's `total` and our own `wait_for` both raise
                # asyncio.TimeoutError and one `except` cannot tell them apart.
                # Owning it is what makes the cause nameable instead of an
                # empty `{e}`.
                #
                # The slot still matters (#97): an unbounded generation can
                # occupy the single inference slot for minutes and starve every
                # other route. And raising the ceiling WITHOUT the inter-token
                # bound would trade a 5-minute truncation for a 30-minute
                # single-slot occupation on a wedge — so the stall bound is
                # what makes a generous ceiling affordable, not a nicety.
                # sock_connect still bounds reaching a dead Ollama, which is
                # neither a stall nor a long generation.
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            ) as resp:
                if getattr(resp, 'status', None) in (400, 404):
                    _raise_complete_refusal(resp.status, await resp.read())
                if resp.status >= 400:
                    raise await _async_status_error(resp)
                resp.raise_for_status()
                if service_owned:
                    # This connection is owned by the durable scheduler, NOT
                    # its HTTP observer. Silence or elapsed queue/generation
                    # time does not establish failure. Real transport errors
                    # still propagate for uncertainty/reconciliation; terminal
                    # output is validated by the owning adapter before release.
                    seen = 0
                    async for raw in resp.content:
                        line = raw.decode(errors='replace').strip()
                        if line:
                            yield _checked_chunk(json.loads(line), seen)
                            seen += 1
                    return
                stream = resp.content.__aiter__()
                seen = 0
                started = time.monotonic()
                last = started
                first_budget = config.ollama_first_token_deadline(prompt_tokens)
                # Logged ARMED, not only on failure. A deadline visible only
                # when it fires cannot be told apart from one that was never
                # wired up — which is how the old flat cap went unnoticed for as
                # long as it did.
                #
                # KEYED ON WHETHER IT IS ACTUALLY ARMED, not on whether we
                # counted the prompt. An earlier version branched on
                # `if prompt_tokens:` and printed "armed ... / 0 tok/s" on a
                # box with no measured rate — claiming a state it was not in,
                # and dividing by a zero it then displayed. Caught in the live
                # log within a minute of the #904 restart, which is the exact
                # class of defect #903 and #904 are both about: a line whose
                # name promises more than it measures.
                armed = (prompt_tokens
                         and config.OLLAMA_FIRST_TOKEN_RATE_TPS > 0)
                if armed:
                    log.info(
                        f"ollama deadlines armed for {payload.get('model')}: "
                        f"first-token {first_budget:.0f}s "
                        f"({prompt_tokens} tokens / "
                        f"{config.OLLAMA_FIRST_TOKEN_RATE_TPS} tok/s + "
                        f"{config.OLLAMA_FIRST_TOKEN_QUEUE_ALLOWANCE_S}s queue + "
                        f"{config.OLLAMA_FIRST_TOKEN_MARGIN_S}s margin), "
                        f"inter-token {config.OLLAMA_STALL_TIMEOUT}s, whole "
                        f"request {config.OLLAMA_GENERATE_TIMEOUT}s"
                    )
                else:
                    why = ("OLLAMA_FIRST_TOKEN_RATE_TPS is unset"
                           if prompt_tokens else "prompt size unknown")
                    log.info(
                        f"ollama first-token deadline NOT armed ({why}) — a "
                        f"silent prefill is bounded only by the whole-request "
                        f"budget {config.OLLAMA_GENERATE_TIMEOUT}s; inter-token "
                        f"{config.OLLAMA_STALL_TIMEOUT}s"
                        + (f"; counted {prompt_tokens} prompt tokens"
                           if prompt_tokens else "")
                    )
                while True:
                    line_budget = (
                        config.OLLAMA_STALL_TIMEOUT if seen else first_budget
                    )
                    remaining = (
                        started + config.OLLAMA_GENERATE_TIMEOUT
                        - time.monotonic()
                    )
                    # Whichever runs out first, and we record which so the
                    # handler reports the cause it actually hit rather than the
                    # one it assumed.
                    bounded_by_request = remaining <= line_budget
                    budget = max(0.0, min(line_budget, remaining))
                    try:
                        raw = await asyncio.wait_for(
                            stream.__anext__(), timeout=budget
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        elapsed = time.monotonic() - started
                        if bounded_by_request:
                            # With no token count the first-token budget IS the
                            # request budget, so this branch is the one an
                            # unarmed deadline always reaches. An earlier
                            # version put that explanation in a separate
                            # `if not seen and not prompt_tokens` below, which
                            # could therefore never run — caught by
                            # `test_stream_deadlines` step 6 looking for the
                            # reason in the message and not finding it.
                            unarmed = (
                                " No first-token deadline was armed, because "
                                "the prompt size was unknown."
                                if not prompt_tokens else ""
                            )
                            raise OllamaRequestTimeout(
                                f"the model did not finish within "
                                f"{config.OLLAMA_GENERATE_TIMEOUT}s "
                                f"({seen} chunks received in {elapsed:.0f}s). "
                                f"This is the whole-request budget, NOT a "
                                f"stall — the generation may have been "
                                f"producing the whole time. Raise "
                                f"OLLAMA_GENERATE_TIMEOUT if prompts this "
                                f"large are expected.{unarmed}",
                                tokens=seen, elapsed=elapsed,
                                deadline=float(config.OLLAMA_GENERATE_TIMEOUT),
                            )
                        if not seen:
                            if prompt_tokens:
                                raise OllamaStalled(
                                    f"no first token within {first_budget:.0f}s "
                                    f"for a {prompt_tokens}-token prompt "
                                    f"(tokens/"
                                    f"{config.OLLAMA_FIRST_TOKEN_RATE_TPS} + "
                                    f"{config.OLLAMA_FIRST_TOKEN_MARGIN_S}s). "
                                    f"Prefill never produced, so this is a "
                                    f"stuck runner rather than a slow one.",
                                    tokens=0, silent_for=first_budget,
                                    phase="first_token", deadline=first_budget,
                                )
                            # Unreachable when the prompt size is unknown —
                            # that case is handled above, where the request
                            # budget bound us. Kept as a guard rather than
                            # deleted: if the floor or the clamp ever make the
                            # first-token budget strictly smaller than the
                            # request budget with no token count, silence
                            # before the first token must still not be reported
                            # as a measured stall.
                            raise OllamaRequestTimeout(
                                f"no answer within {first_budget:.0f}s and the "
                                f"prompt size was unknown, so no first-token "
                                f"deadline was armed",
                                tokens=0, elapsed=elapsed,
                                deadline=first_budget,
                            )
                        silent = time.monotonic() - last
                        raise OllamaStalled(
                            f"the generation stopped mid-answer: {seen} chunks, "
                            f"then nothing for {silent:.0f}s (limit "
                            f"{config.OLLAMA_STALL_TIMEOUT}s). Stuck, not slow.",
                            tokens=seen, silent_for=silent,
                            phase="inter_token",
                            deadline=float(config.OLLAMA_STALL_TIMEOUT),
                        )
                    line = raw.decode(errors="replace").strip()
                    if line:
                        chunk = _checked_chunk(json.loads(line), seen)
                        seen += 1
                        last = time.monotonic()
                        yield chunk
        finally:
            await session.close()

    async def chat_stream(self, model: str, messages: list[dict],
                          prompt_tokens: Optional[int] = None, *, service_owned=False, **kwargs):
        """Chat completion, async streaming. Kills connection on cancel."""
        payload = self._with_num_ctx(
            model, {"model": model, "messages": messages, "stream": True, **kwargs})
        async for chunk in self._stream_with_deadlines(
            "/api/chat", payload, prompt_tokens, service_owned=service_owned
        ):
            yield chunk

    async def generate_stream(self, model: str, prompt: str,
                              prompt_tokens: Optional[int] = None, *, service_owned=False, **kwargs):
        """Generate completion, async streaming. Kills connection on cancel."""
        payload = self._with_num_ctx(
            model, {"model": model, "prompt": prompt, "stream": True, **kwargs})
        async for chunk in self._stream_with_deadlines(
            "/api/generate", payload, prompt_tokens, service_owned=service_owned
        ):
            yield chunk

    def show_model(self, model: str) -> dict:
        """Get model metadata."""
        r = self._http.post("/api/show", json={"model": model})
        r.raise_for_status()
        return r.json()

    def delete_model(self, model: str) -> dict:
        r = self._http.request("DELETE", "/api/delete", json={"model": model})
        r.raise_for_status()
        return r.json()
