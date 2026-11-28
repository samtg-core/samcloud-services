"""Native oMLX chat transport for the service-owned loopback worker."""
from contextlib import asynccontextmanager

import httpx

from .reasoning_request import reasoning_request_error


@asynccontextmanager
async def chat(port, api_key, model, request, *, transport=None):
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
    problem = reasoning_request_error(dict(body, reasoning_effort=template.get('reasoning_effort')))
    if problem:
        raise ValueError(problem)
    async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=None) as client:
        async with client.stream('POST', f'http://127.0.0.1:{port}/v1/chat/completions',
                                 headers={'Authorization': 'Bearer ' + api_key}, json=body) as response:
            response.raise_for_status()
            yield response
