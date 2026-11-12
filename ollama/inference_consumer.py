"""Consumer protocol for durable inference, independent of Hermes/SDK wrappers.

Callers persist logical grant/call keys BEFORE submit, and reuse them after a
lost response or process restart. Never derive idempotency from prompt equality:
two intentional identical requests are still different work. A saved Job is a
convenient cursor, not the only way to recover the job (submit with the same keys).

Short HTTP timeouts bound individual observations, NOT inference or queue age.
Cancellation stops observation only. Callers explicitly choose an execution
grant or release a work-session grant at their own safe boundary.
Uncertain compute is nonterminal: observers retain the same job and wait for
explicit service reconciliation, never resubmit it or infer failure from age.
"""
import asyncio
from dataclasses import asdict, dataclass
import inspect
import json
import math

import httpx


@dataclass(frozen=True)
class Job:
    grant_id: str
    execution_id: str

    def as_dict(self):
        return asdict(self)


class ExecutionStopped(RuntimeError):
    def __init__(self, job, state, result=None):
        self.job, self.state, self.result = job, state, result
        super().__init__('inference execution is ' + state)


class ReconciliationRequired(ExecutionStopped):
    """Do not replace this job: backend completion has not been established."""


class DurableConsumer:
    def __init__(self, http, *, progress=None, poll_interval=1.0, request_timeout=10.0):
        """http is an authenticated httpx.AsyncClient owned by the caller.

        progress receives only job/transport states, never prompts or secrets.
        Observer cancellation does not close a shared HTTP client or cancel jobs.
        """
        if poll_interval <= 0 or request_timeout <= 0:
            raise ValueError('positive polling and HTTP observation intervals required')
        self.http = http
        self.progress = progress
        self.poll_interval = poll_interval
        self.request_timeout = request_timeout

    async def _progress(self, state):
        if self.progress:
            result = self.progress(state)
            if inspect.isawaitable(result):
                await result

    async def _request(self, method, path, **kwargs):
        delay = self.poll_interval
        recovering = False
        while True:
            state = 'transport_unavailable'
            retry_delay = delay
            try:
                response = await self.http.request(method, '/v1/inference' + path,
                                                   timeout=self.request_timeout, **kwargs)
                if response.status_code not in (408, 429, 500, 502, 503, 504):
                    # Authentication, ownership, invalid calls and conflicting
                    # keys are actionable failures, not endless retries.
                    response.raise_for_status()
                    body = response.json()
                    if recovering:
                        await self._progress('transport_available')
                    return body
                state = 'rate_limited' if response.status_code == 429 else 'service_unavailable'
                # Respect an explicit numeric retry interval without making it
                # a job deadline. asyncio cancellation still stops this wait.
                try:
                    retry_after = float(response.headers.get('Retry-After', ''))
                    if math.isfinite(retry_after) and retry_after >= 0:
                        retry_delay = max(delay, retry_after)
                except ValueError:
                    pass
            except httpx.TransportError:
                pass  # response unknown; retry the SAME operation/key
            await self._progress(state)
            recovering = True
            await asyncio.sleep(retry_delay)
            delay = min(max(self.poll_interval, delay * 2), 5.0)

    async def capabilities(self):
        """Authenticated service policy, not a backend/inference probe."""
        return await self._request('GET', '/capabilities')

    async def submit(self, model, *, grant_key, call_key, request, lifetime='work_session',
                     yield_grant_keys=()):
        if not grant_key or not call_key:
            raise ValueError('persisted logical grant and call keys are required')
        if lifetime not in ('work_session', 'execution'):
            raise ValueError('invalid grant lifetime')
        # Freeze this new logical input before any await. Waiting or a lost reply
        # must not turn caller mutation into a different payload under this key.
        body = {'key': call_key, 'request': request}
        if yield_grant_keys:
            body['yield_grant_keys'] = list(yield_grant_keys)
        body = json.loads(json.dumps(body, allow_nan=False))
        grant = await self._request('POST', '/grants', json={
            'model': model, 'key': grant_key, 'lifetime': lifetime})
        ended_predecessors = set()
        while True:
            try:
                call = await self._request('POST', '/grants/' + grant['id'] + '/executions', json=body)
                break
            except httpx.HTTPStatusError as error:
                # Only a typed, owner-scoped refusal identifies work to observe.
                # Ordinary conflicts (including old servers) remain actionable.
                if error.response.status_code != 409 or lifetime != 'work_session':
                    raise
                try:
                    detail = error.response.json()['detail']
                    predecessor = detail['execution_id']
                    valid = (detail['code'] == 'grant_busy' and detail['grant_id'] == grant['id']
                        and isinstance(predecessor, str) and len(predecessor) == 32
                        and all(c in '0123456789abcdef' for c in predecessor))
                except (ValueError, TypeError, KeyError):
                    valid = False
                if not valid:
                    raise
                if predecessor in ended_predecessors:
                    raise ValueError('busy predecessor was already terminal')
                await self._progress('waiting_for_predecessor')
                while True:
                    previous = await self._request('GET', '/executions/' + predecessor)
                    if (previous.get('id') != predecessor or previous.get('grant_id') != grant['id']
                            or previous.get('state') not in
                            ('queued', 'running', 'uncertain', 'complete', 'failed', 'cancelled')):
                        raise ValueError('busy predecessor receipt does not match the service')
                    if previous['state'] in ('complete', 'failed', 'cancelled'):
                        ended_predecessors.add(predecessor)
                        break
                    await asyncio.sleep(self.poll_interval)
                # The predecessor's output is NOT this request's result. Offer
                # the same unaccepted intake again, under the same stable keys.
                # A release/race is checked afresh by the journal transaction.
        return Job(grant['id'], call['id'])

    async def reattach(self, grant_id, execution_id):
        """Read an exact acknowledged job under the caller's auth; no submit/admission.

        A receipt mismatch or inaccessible job is not permission to replace it.
        """
        if not all(isinstance(v,str) and v.strip() for v in (grant_id,execution_id)):
            raise ValueError('exact acknowledged job identity required')
        row = await self._request('GET','/executions/'+execution_id)
        if row.get('id') != execution_id or row.get('grant_id') != grant_id:
            raise ValueError('saved execution receipt does not match the service')
        return Job(grant_id,execution_id)

    async def release(self, job):
        """Explicit safe-boundary handoff; running compute drains on the service."""
        return await self._request('POST', '/grants/' + job.grant_id + '/release')

    async def yield_work(self, job):
        """Long external work yields affinity, not saved calls or the conversation."""
        return await self._request('POST', '/grants/' + job.grant_id + '/yield')

    async def lifecycle_yield(self, grant_key, expected_call_key):
        return await self._request('POST', '/yield', json={'grant_key': grant_key,
                                                         'expected_call_key': expected_call_key})

    async def result(self, job):
        previous = None
        while True:
            call = await self._request('GET', '/executions/' + job.execution_id)
            state = call['state']
            if state != previous:
                await self._progress(state)
                previous = state
            if state == 'complete':
                return call['result']
            if state not in ('queued', 'running', 'uncertain'):
                raise ExecutionStopped(job, state, call.get('result'))
            await asyncio.sleep(self.poll_interval)

    async def events(self, job, *, after=0):
        """Yield durable pages; caller checkpoints next_cursor AFTER consuming.

        A replay after a crash may repeat an unacknowledged page, never inference.
        Empty pages communicate queued/running/terminal states without inventing
        model tokens. Partial deltas are provisional until complete is reported.
        """
        previous = None
        while True:
            page = await self._request('GET', '/executions/' + job.execution_id + '/events',
                                       params={'after': after})
            if page['state'] != previous:
                await self._progress(page['state'])
                previous = page['state']
            yield page
            after = page['next_cursor']
            if page['has_more']:
                continue
            if page['state'] not in ('queued', 'running', 'uncertain'):
                if page['state'] != 'complete':
                    # Events contain no error body. Read this same owner's saved
                    # execution, never infer a refusal from an empty stream.
                    call = await self._request('GET', '/executions/' + job.execution_id)
                    if call['state'] != page['state']:
                        raise ReconciliationRequired(job, 'uncertain')
                    raise ExecutionStopped(job, page['state'], call.get('result'))
                return
            await asyncio.sleep(self.poll_interval)
