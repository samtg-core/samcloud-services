"""Retry native lifecycle metadata separately from inference and callbacks.

The outbox contains identifiers, never prompt/response text. An observation
failure retains the event. It cannot fail, cancel or resubmit compute.
"""
import asyncio
import logging

import httpx


async def source_call(callback, *args):
    # Cancellation does not stop a SQLite worker. Drain the metadata operation
    # before closing the relay/home, just as compute must own its real lifetime.
    task = asyncio.create_task(asyncio.to_thread(callback, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def relay_once(consumer, source, *, cursor=None, through=None, wrap=True):
    cursor = [0] if cursor is None else cursor
    events = await source_call(source.pending_events, cursor[0])
    if not events and cursor[0] and wrap:
        cursor[0] = 0
        events = await source_call(source.pending_events, 0)
    if through is not None:
        events = [event for event in events if event['cursor'] <= through]
    if not events:
        return 0
    cursor[0] = events[-1]['cursor']
    # One bounded observation, not the consumer's unbounded job-observation loop:
    # an unavailable old event must not hide every later event in this batch.
    http = consumer.http
    capabilities = await http.get('/v1/inference/capabilities', timeout=consumer.request_timeout)
    capabilities.raise_for_status()
    document = capabilities.json()
    policy = document.get('stints', {}) if isinstance(document, dict) else None
    jobs = document.get('jobs', {}) if isinstance(document, dict) else None
    legacy_supported = isinstance(policy, dict) and policy.get('lifecycle_yield') is True
    jobs_supported = (isinstance(jobs, dict) and type(jobs.get('version')) is int
                      and jobs['version'] == 1 and jobs.get('explicit_finish') is True)
    acknowledged = 0
    for event in events:
        try:
            if event.get('kind') == 'job_finish':
                if not jobs_supported:
                    continue
                response = await http.post('/v1/inference/jobs/end', json={
                    'key': event['job_key'], 'action': 'finish'}, timeout=consumer.request_timeout)
                if response.status_code == 202:
                    body = response.json()
                    if (isinstance(body, dict) and body.get('key') == event['job_key']
                            and body.get('action') == 'finish' and isinstance(body.get('jobs'), list)):
                        await source_call(source.acknowledge_event, event, 'accepted')
                        acknowledged += 1
                continue
            if event.get('kind', 'grant_yield') != 'grant_yield' or not legacy_supported:
                continue
            response = await http.post('/v1/inference/yield', json={
                'grant_key': event['grant_key'], 'expected_call_key': event['expected_call_key']},
                timeout=consumer.request_timeout)
            verdict = None
            if response.status_code == 200:
                body = response.json()
                if (isinstance(body, dict) and body.get('client_key') == event['grant_key']
                        and body.get('state') in ('queued', 'granted', 'uncertain', 'released', 'cancelled')):
                    verdict = 'accepted'
            elif response.status_code == 409:
                document = response.json()
                detail = document.get('detail') if isinstance(document, dict) else None
                if isinstance(detail, dict) and detail.get('code') == 'stale_lifecycle_event':
                    verdict = 'stale'
            if verdict is not None:
                await source_call(source.acknowledge_event, event, verdict)
                acknowledged += 1
        except (httpx.HTTPError, ValueError):
            continue  # same event remains pending; no inferred backend state
    return acknowledged or -1


async def flush_events(consumer, source):
    """Observe the finite pending prefix once before closing the HTTP client.

    Failed pages never hide later finishes and never loop back during shutdown.
    This sends only persisted intent; it cannot create a finish from client exit.
    An observation failure retains its exact event for the next client/operator.
    """
    highwater = getattr(source, 'pending_event_highwater', None)
    if highwater is None:
        return await relay_once(consumer, source)  # legacy custom source protocol
    through = await source_call(highwater)
    if type(through) is not int or through < 0:
        raise ValueError('explicit lifecycle outbox high-water mark required')
    cursor = [0]
    acknowledged = 0
    while cursor[0] < through:
        previous = cursor[0]
        try:
            result = await relay_once(consumer, source, cursor=cursor, through=through, wrap=False)
            acknowledged += max(result, 0)
        except Exception as error:
            logging.getLogger(__name__).warning('Closing lifecycle observation failed (%s)', type(error).__name__)
        if cursor[0] <= previous:
            break  # concurrent acknowledgement or unreadable metadata; never spin
    return acknowledged


async def relay_events(consumer, source, interval=1.0):
    delay = interval
    previous_error = None
    cursor = [0]
    while True:
        try:
            result = await relay_once(consumer, source, cursor=cursor)
            delay = min(30.0, max(interval, delay * 2)) if result < 0 else interval
            previous_error = None
        except Exception as error:
            kind = type(error).__name__
            if kind != previous_error:
                logging.getLogger(__name__).warning('Lifecycle relay observation failed (%s)', kind)
            previous_error = kind
            delay = min(30.0, max(interval, delay * 2))
        await asyncio.sleep(delay)
