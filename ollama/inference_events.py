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


async def relay_once(consumer, source, *, cursor=None):
    cursor = [0] if cursor is None else cursor
    events = await source_call(source.pending_events, cursor[0])
    if not events and cursor[0]:
        cursor[0] = 0
        events = await source_call(source.pending_events, 0)
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
    if not isinstance(policy, dict) or policy.get('lifecycle_yield') is not True:
        return -1
    acknowledged = 0
    for event in events:
        try:
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
