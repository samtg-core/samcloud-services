"""Explicit owning-profile parent inspection/finish; no inference submission.

Run in the shipping Hermes environment: python -m ollama.requester_job --help.
Local intent commits before network observation. Repeating the exact revision
resends the same finish and cannot finish a later job or cancel accepted work.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from .hermes_queue_provider import host_for_home
from .hermes_queue_source import owner_scope
from .inference_arbiter import Conflict
from .inference_consumer import DurableConsumer
from .inference_events import relay_once
from .inference_identity import IdentityStore, LocalSourceRefused


def local_operation(home, session, *, revision=None):
    home = Path(home)
    if not home.is_absolute() or not (home / 'samcloud-queue/identity.sqlite').is_file():
        raise LocalSourceRefused('existing absolute owning profile home required')
    host = host_for_home(home)
    if not host.job_continuity:
        raise LocalSourceRefused('owning profile has not enabled requester jobs')
    owner = host.owner_session(session)
    store = IdentityStore(host.path)
    try:
        scope = owner_scope(host.scope, owner)
        result = (store.native_job_status(scope) if revision is None else
                  store.request_native_job_finish(scope, revision))
        return dict(session=session, owner_session=owner, parent=result)
    finally:
        store.close()


class FinishDelivery:
    """A single explicit end event; never flush another conversation's work."""
    def __init__(self, path, event_id):
        self.path, self.event_id = path, event_id

    def read(self):
        store = IdentityStore(self.path)
        try:
            return store.operator_finish_delivery(self.event_id)
        finally:
            store.close()

    def pending_events(self, after=0):
        event = self.read()
        return [event] if event['state'] == 'pending' else []

    def acknowledge_event(self, event, verdict):
        if event['id'] != self.event_id:
            raise LocalSourceRefused('operator delivery changed its exact event')
        store = IdentityStore(self.path)
        try:
            store.acknowledge_event(event, verdict)
        finally:
            store.close()


async def deliver_finish(consumer, source):
    try:
        await relay_once(consumer, source)
        return dict(delivery=source.read()['state'])
    except Exception as error:
        # No credentials, service response bodies or prompts in CLI diagnostics.
        return dict(delivery=source.read()['state'], observation_error=type(error).__name__)


async def deliver_configured(source, endpoint, token):
    try:
        url = httpx.URL(endpoint)
        if url.scheme != 'https' or not url.host or url.username or url.password:
            raise LocalSourceRefused('own configured HTTPS provider endpoint required')
        path = url.path.rstrip('/')
        if path.endswith('/v1'):
            path = path[:-3]
        url = url.copy_with(path=path + '/', query=None, fragment=None)
        async with httpx.AsyncClient(base_url=url, headers={'Authorization':'Bearer ' + token},
                                     follow_redirects=False, trust_env=False) as http:
            return await deliver_finish(DurableConsumer(http), source)
    except Exception as error:
        return dict(delivery=source.read()['state'], observation_error=type(error).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', required=True, type=Path, help='exact owning Hermes profile home')
    parser.add_argument('--session', required=True, help='original session or persisted compression alias')
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('status', help='inspect this session parent without network I/O')
    end = sub.add_parser('finish', help='seal this parent and drain accepted children without cancellation')
    end.add_argument('--revision', required=True, help='revision returned by status; stale attempts refuse')
    args = parser.parse_args(argv)
    try:
        result = local_operation(args.home, args.session,
                                 revision=args.revision if args.action == 'finish' else None)
        if args.action == 'finish':
            source = FinishDelivery(args.home / 'samcloud-queue/identity.sqlite',
                                    result['parent']['event_id'])
            token = os.environ.get('SAMCLOUD_QUEUE_API_KEY')
            endpoint = os.environ.get('SAMCLOUD_QUEUE_BASE_URL')
            if not token or not endpoint:
                result.update(delivery=source.read()['state'], observation_error='ProviderCredentialsUnavailable')
            else:
                result.update(asyncio.run(deliver_configured(source, endpoint, token)))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, Conflict) as error:
        print(json.dumps({'error':type(error).__name__, 'detail':str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
