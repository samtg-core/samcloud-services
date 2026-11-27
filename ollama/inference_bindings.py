"""Resolve a service-operator binding for a verified runtime on durable inference routes."""
import json
import re
from pathlib import Path

from fastapi import HTTPException

HEADER = 'X-Samcloud-Inference-Binding'


# Only the existing durable inference contract is available through a binding.
_ROUTES = {
    'GET': re.compile(r'/v1/inference/(?:capabilities|grants/[a-f0-9]{32}|executions/[a-f0-9]{32}(?:/events)?)'),
    'POST': re.compile(r'/v1/inference/(?:grants|yield|grants/[a-f0-9]{32}/(?:release|yield|executions))'),
}


def resolve(request, directory):
    values = request.headers.getlist(HEADER)
    if not values:
        return None
    route = _ROUTES.get(request.method)
    if (len(values) != 1 or not re.fullmatch(r'[a-z][a-z0-9-]{0,62}', values[0])
            or route is None or not route.fullmatch(request.url.path)):
        raise HTTPException(403, 'Invalid inference binding')
    caller = getattr(request.state, 'caller', None)
    if not directory or not isinstance(caller, dict) or caller.get('valid') is not True:
        raise HTTPException(403, 'Inference binding unavailable')
    binding_id = values[0]
    try:
        # The provider owns this directory. Atomic replacement/removal takes effect
        # on the next request, independently of the ordinary bearer verification cache.
        policy = json.loads((Path(directory) / (binding_id + '.json')).read_bytes())
    except FileNotFoundError:
        raise HTTPException(403, 'Inference binding unavailable') from None
    except (ValueError, OSError):
        raise HTTPException(503, 'Inference binding policy unavailable') from None
    if not isinstance(policy, dict) or set(policy) != {'issuer', 'subject_id', 'subject', 'models'}:
        raise HTTPException(503, 'Invalid inference binding policy')
    if (not isinstance(policy['subject'], str) or not re.fullmatch(r'[a-z][a-z0-9-]{0,62}', policy['subject'])
            or not isinstance(policy['subject_id'], str) or not re.fullmatch(r'user:[1-9][0-9]*', policy['subject_id'])
            or not isinstance(policy['models'], list) or not policy['models']
            or not all(isinstance(m, str) and m for m in policy['models'])):
        raise HTTPException(503, 'Invalid inference binding policy')
    if not caller.get('owner_id') or caller['owner_id'] != policy['issuer']:
        raise HTTPException(403, 'Inference binding unavailable')
    return dict(binding=binding_id, **policy)
