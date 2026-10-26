"""Opt-in durable inference API. Not mounted by server.py yet.

Uses the gateway's already-verified caller. All resource/model mapping and call
validation are injected by the service, never supplied by a consumer. HTTP
requests return promptly; they do not own the worker or its inference lifetime.
"""
import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from .inference_arbiter import Conflict, NotFound, LifecycleStale


class GrantRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    key: str = Field(min_length=1, max_length=512)
    model: str = Field(min_length=1, max_length=512)
    lifetime: Literal['work_session', 'execution'] = 'work_session'


class CallRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    key: str = Field(min_length=1, max_length=512)
    request: dict[str, Any]
    yield_grant_keys: list[str] = Field(default_factory=list, max_length=32)
    job_key: str | None = Field(default=None, min_length=1, max_length=512)


class JobBoundary(BaseModel):
    model_config = ConfigDict(extra='forbid')
    key: str = Field(min_length=1, max_length=512)
    action: Literal['finish', 'yield'] = 'finish'


class LifecycleYield(BaseModel):
    model_config = ConfigDict(extra='forbid')
    grant_key: str = Field(min_length=1, max_length=512)
    expected_call_key: str = Field(min_length=1, max_length=512)


def owner(request):
    caller = getattr(getattr(request, 'state', None), 'caller', None)
    name = caller.get('username') if isinstance(caller, dict) else None
    if not isinstance(name, str) or not name.strip():
        # Even a development gateway with AUTH_ENABLED=False may not expose
        # private queue records without an authenticated owner.
        raise HTTPException(401, 'Verified identity required')
    return name


def public_call(row):
    # Input prompts remain in the private journal, not incidental status reads.
    result = {k: row[k] for k in ('id', 'grant_id', 'state', 'created', 'updated')}
    result['wait_reason'] = row['wait_reason']
    result['job_id'] = row['job_id']
    result['result'] = json.loads(row['result']) if row['result'] is not None else None
    return result


def build_router(supervisor, resolve_model, validate_call):
    """resolve_model(model)->(resource, canonical_model); validation is local.

    Neither callback may perform inference. Model loading belongs after admission.
    validate_call(model, request) returns a validated, canonical request dict.
    """
    router = APIRouter(prefix='/v1/inference')

    def invoke(fn, *args):
        try:
            return fn(*args)
        except NotFound:
            raise HTTPException(404, 'Not found') from None
        except LifecycleStale:
            raise HTTPException(409, {'code': 'stale_lifecycle_event'}) from None
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError:
            raise HTTPException(422, 'Invalid inference request') from None

    @router.get('/capabilities')
    async def capabilities(request: Request):
        owner(request)
        policy = supervisor.journal.stint_policy
        result = {'stints': {'version': 1, 'enabled': policy is not None,
                          'dependency_yield': True,
                          'lifecycle_yield': True,
                          'idle_grace_seconds': policy.idle_grace_seconds if policy else None,
                          'fairness_seconds': policy.fairness_seconds if policy else None},
                'jobs': {'version': 1, 'enabled': policy is not None,
                         'same_service_children': True, 'cross_service_delegation': False,
                         'explicit_finish': True, 'explicit_yield': True}}
        if getattr(supervisor.journal,'supports_single_call_jobs',False):
            result['jobs']['implicit_single_call_finish']=True
        return result

    @router.get('/jobs')
    async def jobs(request: Request, key: str = Query(min_length=1, max_length=512)):
        return {'jobs': invoke(supervisor.journal.jobs, owner(request), key)}

    @router.post('/jobs/end', status_code=202)
    async def end_job(body: JobBoundary, request: Request):
        identity = owner(request)
        rows = invoke(lambda: supervisor.end_job(identity, body.key, finish=body.action == 'finish'))
        return {'key': body.key, 'action': body.action, 'jobs': rows}

    @router.post('/grants', status_code=202)
    async def submit(body: GrantRequest, request: Request):
        identity = owner(request)
        resource, model = invoke(resolve_model, body.model)
        return invoke(supervisor.journal.submit, identity, body.key, resource, model, body.lifetime)

    @router.post('/yield')
    async def lifecycle_yield(body: LifecycleYield, request: Request):
        return invoke(supervisor.yield_by_key, owner(request), body.grant_key, body.expected_call_key)

    @router.get('/grants/{grant_id}')
    async def grant(grant_id: str, request: Request):
        return invoke(supervisor.journal.get, grant_id, owner(request))

    @router.post('/grants/{grant_id}/release')
    async def release(grant_id: str, request: Request):
        return invoke(supervisor.release, grant_id, owner(request))

    @router.post('/grants/{grant_id}/yield')
    async def yield_grant(grant_id: str, request: Request):
        return invoke(supervisor.yield_grant, grant_id, owner(request))

    @router.post('/grants/{grant_id}/executions', status_code=202)
    async def enqueue(grant_id: str, body: CallRequest, request: Request):
        identity = owner(request)
        grant = invoke(supervisor.journal.get, grant_id, identity)
        canonical = invoke(validate_call, grant['model'], body.request)
        call = invoke(lambda: supervisor.journal.enqueue(grant_id, identity, body.key, canonical,
                                                        yield_grant_keys=body.yield_grant_keys,
                                                        job_key=body.job_key))
        supervisor.pump(grant['resource'])
        return public_call(supervisor.journal.execution(call['id'], identity))

    @router.get('/executions/{execution_id}')
    async def execution(execution_id: str, request: Request):
        return public_call(invoke(supervisor.journal.execution, execution_id, owner(request)))

    @router.get('/executions/{execution_id}/events')
    async def events(execution_id: str, request: Request,
                     after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        identity = owner(request)
        return invoke(lambda: supervisor.journal.events(execution_id, identity, after=after, limit=limit))

    return router
