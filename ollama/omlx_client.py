"""Native oMLX chat transport for the service-owned loopback worker."""
from contextlib import asynccontextmanager

import httpx

async def model_status(port, api_key, model, *, transport=None):
    """Read this private worker's exact model entry before configuring its chats."""
    if type(port) is not int or not 1 <= port <= 65535 or not api_key:
        raise ValueError('Missing local oMLX backend binding')
    async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=10) as client:
        response = await client.get(f'http://127.0.0.1:{port}/v1/models/status',
                                    headers={'Authorization': 'Bearer ' + api_key})
        response.raise_for_status()
        rows = [row for row in response.json()['models'] if row['id'] == model]
    if len(rows) != 1:
        raise ValueError('Expected one exact oMLX model in worker inventory')
    return rows[0]


@asynccontextmanager
async def chat(port, api_key, model, request, *, transport=None, reasoning_levels=None):
    """Keep the backend connection owned until the gateway finishes consuming it."""
    if type(port) is not int or not 1 <= port <= 65535 or not api_key:
        raise ValueError('Missing local oMLX backend binding')
    body = dict(request, model=model)
    effort = body.pop('reasoning_effort', None)
    template = dict(body.get('chat_template_kwargs') or {})
    if effort is not None:
        if 'reasoning_effort' in template and template['reasoning_effort'] != effort:
            raise ValueError('Conflicting reasoning_effort values')
        template['reasoning_effort'] = effort
        body['chat_template_kwargs'] = template
    effort = template.get('reasoning_effort')
    # An oMLX model can advertise additional levels in /v1/models/status.
    # The caller supplies that inventory from the service-owned worker.
    if effort is not None:
        if effort != 'none' and (type(body.get('max_tokens')) is not int or body['max_tokens'] <= 0):
            raise ValueError('reasoning_effort requires a positive integer max_tokens')
        if reasoning_levels is None:
            status = await model_status(port, api_key, model, transport=transport)
            reasoning_levels = status.get('reasoning_levels') or []
        if not isinstance(effort, str) or effort not in reasoning_levels:
            raise ValueError('Unsupported oMLX reasoning_effort')
    async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=None) as client:
        async with client.stream('POST', f'http://127.0.0.1:{port}/v1/chat/completions',
                                 headers={'Authorization': 'Bearer ' + api_key}, json=body) as response:
            response.raise_for_status()
            yield response
