"""Opt-in OpenAI SDK adapter for the durable queue; not installed in Hermes.

Requires explicit, caller-persisted grant/call keys. Missing keys fail at request
time (never fall through to direct inference). Work-session release remains an
explicit caller operation; execution grants hand off at service-confirmed end.
Host deadlines and logical-call persistence are NOT
implemented by this transport.
"""
import asyncio
from concurrent.futures import CancelledError as FutureCancelled
from contextlib import suppress
from contextlib import nullcontext
import json
import inspect
import logging
import threading
import time
from types import SimpleNamespace

from anyio.from_thread import start_blocking_portal
import httpx

from .inference_consumer import DurableConsumer, ExecutionStopped


def _error_response(request, error):
    """Preserve affirmative errors and reject contradictory terminal evidence.

    This changes the observer's error type only. It neither releases admission
    nor manufactures backend termination. Saved error bodies must come from the
    trusted service adapter, which is responsible for sanitizing provider text.
    Canonical uncertain jobs keep waiting in DurableConsumer; they do not reach
    this mapper merely because the service has not reconciled their worker.
    """
    if isinstance(error, httpx.HTTPStatusError):
        return httpx.Response(error.response.status_code, request=request,
            content=error.response.content, headers={'content-type': 'application/json',
                                                   'x-should-retry': 'false'})
    result = error.result
    if (error.state == 'failed' and isinstance(result, dict) and
            result.get('kind') == 'error' and type(result.get('status')) is int and
            400 <= result['status'] <= 599 and isinstance(result.get('detail'), str)):
        status = result['status']
        body = {'error': {'message': result['detail'],
                         'type': result.get('type') or 'provider_error',
                         'code': result.get('code') or 'inference_failed'}}
    else:
        status = 409
        body = {'error': {'message': 'Inference requires reconciliation' if error.state == 'uncertain'
                         else 'Inference execution is ' + error.state,
                         'type': 'inference_state', 'code': 'inference_' + error.state}}
    return httpx.Response(status, request=request, json=body,
        headers={'x-should-retry': 'false', 'X-Inference-Execution': error.job.execution_id,
                 'X-Inference-Grant': error.job.grant_id})


def _stream_error(request, error):
    # Headers have already been emitted. Raise the same public SDK exception
    # type as the non-stream response rather than inventing an answer delta.
    import openai
    response = _error_response(request, error)
    body = response.json()
    classes = {400: openai.BadRequestError, 401: openai.AuthenticationError,
               403: openai.PermissionDeniedError, 404: openai.NotFoundError,
               409: openai.ConflictError, 422: openai.UnprocessableEntityError,
               429: openai.RateLimitError}
    cls = classes.get(response.status_code, openai.InternalServerError
                      if response.status_code >= 500 else openai.APIStatusError)
    detail = body.get('error', body)
    return cls(str(detail.get('message', 'Inference failed')), response=response, body=detail)


async def _status_pages(pages, job, model):
    """Ephemeral control frames from successful service reads, never saved deltas.

    Cursor and provider events are unchanged. This layer emits nothing from
    failed polls; the outer observer reports explicit local-adapter/unknown
    state instead, never a counterfeit healthy service waiting state.
    """
    try:
        async for page in pages:
            if not page['events'] and page['state'] in ('queued', 'running', 'uncertain'):
                frame = {'id': job.execution_id, 'object': 'chat.completion.chunk',
                         'created': int(time.time()), 'model': model, 'choices': [],
                         'model_service': {'kind': 'queue_status', 'state': page['state'],
                                           'execution_id': job.execution_id}}
                yield {**page, 'events': [{'data': 'data: ' + json.dumps(frame) + '\n\n', 'control': True}]}
            else:
                yield page
    finally:
        await pages.aclose()


async def _intent(request, key_provider):
    if request.method != 'POST' or not request.url.path.endswith('/chat/completions'):
        raise ValueError('queue adapter supports chat completions only')
    grant_key = request.headers.get('X-Inference-Grant-Key')
    call_key = request.headers.get('Idempotency-Key')
    body = json.loads(request.content)
    yield_grant_keys = ()
    execution_only = False
    if key_provider is not None:
        # Runs before any service write, in the submitting caller's ContextVars.
        # Only request content is passed, never credentials or HTTP headers.
        # The host owns durable allocation; exceptions must NOT fall through.
        keys = key_provider(body)
        if inspect.isawaitable(keys):
            keys = await keys
        supplied = (keys.grant_key, keys.call_key)
        if not all(isinstance(key, str) and key.strip() for key in supplied):
            raise ValueError('host returned invalid durable call keys')
        if (grant_key and grant_key != supplied[0]) or (call_key and call_key != supplied[1]):
            raise ValueError('explicit request identity conflicts with host identity')
        grant_key, call_key = supplied
        yield_grant_keys = getattr(keys, 'yield_grant_keys', ())
        execution_only = getattr(keys,'execution_only',False)
        if type(execution_only) is not bool:
            raise ValueError('host returned invalid execution-only policy')
        if (not isinstance(yield_grant_keys, tuple) or len(yield_grant_keys) > 32 or
                any(not isinstance(key, str) or not key.strip() for key in yield_grant_keys)):
            raise ValueError('host returned invalid dependency handoff keys')
    if not grant_key or not call_key:
        raise ValueError('durable inference requires explicit grant and logical call keys')
    intent = (body, grant_key, call_key, yield_grant_keys)
    return intent+('execution',) if execution_only else intent


async def _open_intent(consumer, intent, on_state=None, state_frames=False,
                       grant_lifetime='work_session', require_stint_policy=False, observation=None):
    body, grant_key, call_key, yield_grant_keys = intent[:4]
    if len(intent)>4:
        if len(intent)!=5 or intent[4]!='execution':
            raise ValueError('invalid host execution-only policy')
        grant_lifetime='execution'
    identity = {'grant_key': grant_key, 'call_key': call_key}
    if on_state is not None or observation is not None:
        async def report(state):
            if observation is not None:
                observation(state, identity)
            if on_state is None:
                return
            try:
                kind = 'transport' if state in ('transport_unavailable', 'transport_available',
                    'rate_limited', 'service_unavailable') else 'execution'
                result = on_state({**identity, 'kind': kind, 'state': state})
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                logging.getLogger(__name__).warning('Queue state observer failed (%s)', type(error).__name__)
        consumer = DurableConsumer(consumer.http, progress=report,
            poll_interval=consumer.poll_interval, request_timeout=consumer.request_timeout)
    if require_stint_policy or yield_grant_keys:
        capabilities = await consumer.capabilities()
        policy = capabilities.get('stints') if isinstance(capabilities, dict) else None
        if (not isinstance(policy, dict) or type(policy.get('version')) is not int
                or policy['version'] != 1 or policy.get('enabled') is not True):
            # An older explicit-release server could accept this session and
            # retain it forever. Refuse BEFORE creating any service-owned job.
            raise ValueError('service does not advertise automatic conversation stints')
        if require_stint_policy and policy.get('lifecycle_yield') is not True:
            raise ValueError('service does not advertise native lifecycle handoff')
        if yield_grant_keys and policy.get('dependency_yield') is not True:
            raise ValueError('service does not advertise atomic dependency handoff')
    job = await consumer.submit(body['model'], grant_key=grant_key, call_key=call_key,
                                request=body, lifetime=grant_lifetime,
                                yield_grant_keys=yield_grant_keys)
    identity.update(job.as_dict())
    if observation is not None:
        observation('accepted', identity)
    headers = {'X-Inference-Execution': job.execution_id, 'X-Inference-Grant': job.grant_id}
    if body.get('stream'):
        pages = consumer.events(job)
        if state_frames:
            pages = _status_pages(pages, job, body['model'])
        return headers, pages, None
    result = await consumer.result(job)
    if result.get('kind') != 'json':
        raise ValueError('saved result does not match the requested response mode')
    return headers, None, result['body']


async def _open(consumer, request, on_state=None, state_frames=False, key_provider=None,
                grant_lifetime='work_session', require_stint_policy=False):
    intent = await _intent(request, key_provider)
    if state_frames and intent[0].get('stream'):
        # Return observation before service acceptance. These headers identify
        # LOCAL durable intent, never an invented service grant/execution.
        snapshot = {'grant_key': intent[1], 'call_key': intent[2], 'phase': 'submission_unconfirmed'}
        def observation(state, identity):
            snapshot.update(identity)
            snapshot['phase'] = ('contact_unknown' if state in ('transport_unavailable', 'service_unavailable')
                                 else 'awaiting_service_observation')
        opening = asyncio.create_task(_open_intent(consumer, intent, on_state, True,
            grant_lifetime, require_stint_policy, observation))
        return {'X-Inference-Grant-Key': intent[1], 'X-Inference-Call-Key': intent[2],
                'X-Inference-Observation': 'local-intent'}, _ObservedPages(
                    opening, snapshot, intent[0]['model'], max(.05, min(1., consumer.poll_interval))), None
    return await _open_intent(consumer, intent, on_state, state_frames, grant_lifetime, require_stint_policy)


class _ObservedPages:
    """Adapter liveness is not service progress; heartbeats never end a job.

    A pending read remains the SAME task across pulses. No timeout cancels it.
    Closing cancels observation only, never releases/resubmits durable compute.
    """
    def __init__(self, opening, snapshot, model, interval):
        self.opening, self.snapshot, self.model, self.interval = opening, snapshot, model, interval
        self.pages = self.read = None
        self.closed = False

    def __aiter__(self):
        return self

    def pulse(self):
        state = {'kind': 'adapter_status', 'source': 'local_adapter',
                 'state': self.snapshot['phase'], 'service_confirmation': 'unknown',
                 **{k: v for k, v in self.snapshot.items() if k != 'phase'}}
        return self.control(state)

    def control(self, state):
        frame = {'id': state.get('execution_id') or 'intent:' + self.snapshot['call_key'],
                 'object': 'chat.completion.chunk',
                 'created': int(time.time()), 'model': self.model, 'choices': [], 'model_service': state}
        return {'events': [{'data': 'data: ' + json.dumps(frame) + '\n\n', 'control': True}]}

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        if self.pages is None:
            done, _ = await asyncio.wait({self.opening}, timeout=self.interval)
            if not done:
                return self.pulse()
            headers, self.pages, _ = self.opening.result()
            return self.control({'kind': 'job_accepted', 'source': 'service', 'state': 'accepted',
                'grant_id': headers['X-Inference-Grant'], 'execution_id': headers['X-Inference-Execution'],
                'grant_key': self.snapshot['grant_key'], 'call_key': self.snapshot['call_key']})
        if self.read is None:
            self.read = asyncio.create_task(anext(self.pages))
        done, _ = await asyncio.wait({self.read}, timeout=self.interval)
        if not done:
            return self.pulse()
        read, self.read = self.read, None
        page = read.result()
        self.snapshot['last_confirmed_state'] = page['state']
        self.snapshot['last_confirmed_at'] = time.time()
        self.snapshot['phase'] = 'awaiting_service_observation'
        return page

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        if not self.opening.done():
            self.opening.cancel()
        try:
            _, pages, _ = await self.opening
        except (Exception, asyncio.CancelledError):
            pages = None
        if self.read is not None:
            self.read.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await self.read
        if pages is not None:
            await pages.aclose()


class _SyncOutput(httpx.SyncByteStream):
    def __init__(self, portal, pages, request):
        self.portal, self.pages, self.request = portal, pages, request
        self.lock = threading.Lock()
        self.pending = None
        self.closed = False

    async def _next(self):
        try:
            return await anext(self.pages)
        finally:
            if self.closed:
                await self.pages.aclose()

    def __iter__(self):
        try:
            while True:
                with self.lock:
                    if self.closed:
                        return
                    pending = self.pending = self.portal.start_task_soon(self._next)
                try:
                    page = pending.result()
                except FutureCancelled:
                    if self.closed:
                        return
                    raise
                finally:
                    with self.lock:
                        self.pending = None
                for event in page['events']:
                    if self.closed:
                        return
                    yield event['data'].encode()
        except StopAsyncIteration:
            return
        except (ExecutionStopped, httpx.HTTPStatusError) as error:
            raise _stream_error(self.request, error) from None
        finally:
            self.close()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            pending = self.pending
        # Close the observation, not the saved job. A running async generator
        # cannot be aclosed; cancel its pending read so its own frame unwinds.
        if pending is None or not pending.cancel():
            self.portal.call(self.pages.aclose)


class _AsyncOutput(httpx.AsyncByteStream):
    def __init__(self, pages, request):
        self.pages, self.request = pages, request
        self.pending = None
        self.closed = False

    async def __aiter__(self):
        try:
            while not self.closed:
                self.pending = asyncio.create_task(anext(self.pages))
                try:
                    page = await self.pending
                except StopAsyncIteration:
                    return
                except asyncio.CancelledError:
                    if self.closed:
                        return
                    raise
                finally:
                    self.pending = None
                for event in page['events']:
                    if self.closed:
                        return
                    yield event['data'].encode()
        except (ExecutionStopped, httpx.HTTPStatusError) as error:
            raise _stream_error(self.request, error) from None
        finally:
            await self.aclose()

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        pending = self.pending
        if pending is not None:
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
        await self.pages.aclose()


class QueueTransport(httpx.BaseTransport):
    def __init__(self, http_factory, *, poll_interval=1.0, on_state=None, state_frames=False,
                 key_provider=None, grant_lifetime='work_session', require_stint_policy=False,
                 event_source=None):
        # This loop belongs to the SDK client, not to an observer or worker turn.
        self.context = start_blocking_portal()
        self.portal = self.context.__enter__()
        self.closed = False
        self.on_state = on_state
        self.state_frames = state_frames
        self.key_provider = key_provider
        self.grant_lifetime = grant_lifetime
        self.require_stint_policy = require_stint_policy
        async def initialize():
            self.http = http_factory()
            self.consumer = DurableConsumer(self.http, poll_interval=poll_interval)
            self.events_task = None
            if event_source is not None:
                from .inference_events import relay_events
                self.events_task = asyncio.create_task(relay_events(self.consumer, event_source))
        try:
            self.portal.call(initialize)
        except BaseException:
            self.context.__exit__(None, None, None)
            raise

    def handle_request(self, request):
        try:
            headers, pages, body = self.portal.call(_open, self.consumer, request, self.on_state,
                self.state_frames, self.key_provider, self.grant_lifetime, self.require_stint_policy)
        except (ExecutionStopped, httpx.HTTPStatusError) as error:
            return _error_response(request, error)
        if pages is not None:
            return httpx.Response(200, headers={**headers, 'content-type': 'text/event-stream'},
                                  stream=_SyncOutput(self.portal, pages, request))
        return httpx.Response(200, headers=headers, json=body)

    def close(self):
        if not self.closed:
            self.closed = True
            async def shutdown():
                if self.events_task is not None:
                    self.events_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self.events_task
                await self.http.aclose()
            try:
                self.portal.call(shutdown)
                self.portal.call(self.portal.stop, True)
            finally:
                self.context.__exit__(None, None, None)


class AsyncQueueTransport(httpx.AsyncBaseTransport):
    def __init__(self, http_factory, *, poll_interval=1.0, on_state=None, state_frames=False,
                 key_provider=None, grant_lifetime='work_session', require_stint_policy=False):
        self.http_factory, self.poll_interval = http_factory, poll_interval
        self.on_state = on_state
        self.state_frames = state_frames
        self.key_provider = key_provider
        self.grant_lifetime = grant_lifetime
        self.require_stint_policy = require_stint_policy
        self.http = self.consumer = None

    async def handle_async_request(self, request):
        if self.consumer is None:
            self.http = self.http_factory()
            self.consumer = DurableConsumer(self.http, poll_interval=self.poll_interval)
        try:
            headers, pages, body = await _open(self.consumer, request, self.on_state,
                self.state_frames, self.key_provider, self.grant_lifetime, self.require_stint_policy)
        except (ExecutionStopped, httpx.HTTPStatusError) as error:
            return _error_response(request, error)
        if pages is not None:
            return httpx.Response(200, headers={**headers, 'content-type': 'text/event-stream'},
                                  stream=_AsyncOutput(pages, request))
        return httpx.Response(200, headers=headers, json=body)

    async def aclose(self):
        if self.http is not None:
            await self.http.aclose()


class QueuedClient:
    """Dual sync/async SDK facade for Hermes's supported custom-client seam.

    Does not read a profile home, choose credentials, synthesize call keys or
    rewrite context. Those remain the host's responsibility. The async facade is
    essential: merely setting SKIP_ASYNC_WRAP on a sync client would break aux.
    """
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key, base_url, http_factory=None, poll_interval=1.0, on_state=None,
                 state_frames=False, key_provider=None, complete_responses=False,
                 grant_lifetime='work_session', require_stint_policy=False, event_source=None,
                 call_scope=None, **kwargs):
        # Optional host diagnostics: service/transport states, not model tokens or
        # a consumer-completion receipt. Keys must be opaque caller-owned IDs.
        # Sync calls run the callback on the transport's portal thread. Callbacks
        # must return promptly; this is NOT a native watchdog integration.
        from openai import OpenAI, AsyncOpenAI
        self.api_key, self.base_url = api_key, base_url
        # Explicit auxiliary-client policy. Hermes accepts a complete response
        # from its auxiliary create() even when it requested streaming. Keeping
        # that wait inside the transport avoids treating service state frames
        # as summary tokens. Main streaming clients leave this disabled.
        self.complete_responses = complete_responses
        if grant_lifetime not in ('work_session', 'execution'):
            raise ValueError('invalid grant lifetime')
        self.grant_lifetime = grant_lifetime
        if require_stint_policy and grant_lifetime != 'work_session':
            raise ValueError('automatic stint policy requires work_session lifetime')
        self.require_stint_policy = require_stint_policy
        self.event_source = event_source
        self.call_scope = call_scope
        if http_factory is None:
            # The supplied seat credential is used only at its configured service.
            # No redirects, environment proxy or alternate provider is selected.
            # SDK endpoints end in /v1; DurableConsumer owns that API prefix.
            # httpx appends even paths beginning with '/', so keeping both
            # would silently target /v1/v1/inference. Preserve proxy prefixes.
            service_url = httpx.URL(base_url)
            path = service_url.path.rstrip('/')
            if path.endswith('/v1'):
                path = path[:-3]
            service_url = service_url.copy_with(path=path + '/')
            http_factory = lambda: httpx.AsyncClient(base_url=service_url,
                headers={'Authorization': 'Bearer ' + api_key}, follow_redirects=False, trust_env=False)
        common = {'api_key': api_key, 'base_url': base_url, 'max_retries': 0}
        if kwargs.get('default_headers'):
            common['default_headers'] = kwargs['default_headers']
        self.sync = OpenAI(**common, http_client=httpx.Client(
            transport=QueueTransport(http_factory, poll_interval=poll_interval, on_state=on_state,
                                     state_frames=state_frames, key_provider=key_provider,
                                     grant_lifetime=grant_lifetime,
                                     require_stint_policy=require_stint_policy, event_source=event_source)))
        self.async_client = AsyncOpenAI(**common, http_client=httpx.AsyncClient(
            transport=AsyncQueueTransport(http_factory, poll_interval=poll_interval, on_state=on_state,
                                          state_frames=state_frames, key_provider=key_provider,
                                          grant_lifetime=grant_lifetime,
                                          require_stint_policy=require_stint_policy)))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        if self.complete_responses and kwargs.get('stream'):
            kwargs = dict(kwargs, stream=False)
            kwargs.pop('stream_options', None)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            with self.call_scope() if self.call_scope else nullcontext():
                return self.sync.chat.completions.create(**kwargs)
        async def execute():
            with self.call_scope() if self.call_scope else nullcontext():
                return await self.async_client.chat.completions.create(**kwargs)
        return execute()

    def with_options(self, **kwargs):
        copy = object.__new__(type(self))
        copy.api_key, copy.base_url = self.api_key, self.base_url
        copy.complete_responses = self.complete_responses
        copy.grant_lifetime = self.grant_lifetime
        copy.require_stint_policy = self.require_stint_policy
        copy.event_source = self.event_source
        copy.call_scope = self.call_scope
        copy.sync = self.sync.with_options(**kwargs)
        copy.async_client = self.async_client.with_options(**kwargs)
        copy.chat = SimpleNamespace(completions=SimpleNamespace(create=copy.create))
        return copy

    def close(self):
        self.sync.close()

    async def aclose(self):
        await self.async_client.close()
        self.sync.close()
