"""Explicit root-runtime launcher; never adopts a legacy live queue or claim."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import ipaddress
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit

from .inference_bootstrap import QueueSettings
from .inference_root_bootstrap import RootQueueStartup
from .inference_root_client import RegistryRootClient, RootIntentStore, _base, _identifier, _resource
from .manager import Backend


@dataclass(frozen=True)
class RootLaunchPlan:
    path: Path
    sha256: str
    registry: str
    actor: str
    service: str
    resource: str
    journal: Path
    models: Mapping[str, Backend]

    @classmethod
    def load(cls, path, settings):
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError('duplicate key')
                value[key] = item
            return value
        try:
            requested = Path(path)
            if not requested.is_absolute():
                raise ValueError('absolute configuration required')
            source = requested.resolve(strict=True)
            raw = source.read_bytes()
            body = json.loads(raw, object_pairs_hook=unique)
            if (not isinstance(body, dict) or set(body) != {
                'version', 'registry', 'actor', 'service', 'resource', 'journal', 'backends'}
                or type(body['version']) is not int or body['version'] != 1):
                raise ValueError('explicit versioned configuration required')
            registry = _base(body['registry'])
            actor, service = _identifier(body['actor']), _identifier(body['service'])
            resource = _resource(body['resource'])
            if not isinstance(body['journal'], str) or not isinstance(body['backends'], dict):
                raise ValueError('explicit journal and model map required')
            journal = Path(body['journal'])
            if (not journal.is_absolute() or journal.resolve() == settings.journal.resolve()
                or journal.exists() and settings.journal.exists() and journal.samefile(settings.journal)
                or set(body['backends']) != set(settings.models)
                or set(settings.models.values()) != {resource}
                or settings.reservation_mb or settings.shared_queue_wait_s is not None):
                raise ValueError('root configuration differs from queue scope')
            models = {name: Backend(value) for name, value in body['backends'].items()}
            if any(value not in (Backend.OLLAMA, Backend.VLM) for value in models.values()):
                raise ValueError('only qualified local backends allowed')
            return cls(source, hashlib.sha256(raw).hexdigest(), registry, actor,
                service, resource, journal, MappingProxyType(models))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            # Never echo file contents, identifiers supplied in a malformed file,
            # or credential-looking values into argparse/operational logs.
            raise ValueError('Invalid root startup configuration') from None


def confirm_cold(service):
    """Fresh read-only whole-local-backend absence, not a capacity grant."""
    from .inference_shared_resource import cold_vision_absent
    try:
        manager = service.mgr
        if manager is None or manager.models:
            return False
        running = manager.ollama.list_running()
        return isinstance(running, list) and not running and cold_vision_absent() is True
    except Exception:
        return False


def local_backends(configuration):
    """The local process/headroom sensor cannot qualify a remote backend."""
    def loopback(host):
        if host == 'localhost':
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            return False
    try:
        url = urlsplit(configuration.OLLAMA_BASE)
        return (url.scheme in ('http', 'https') and loopback(url.hostname)
            and url.username is None and url.password is None and not url.query
            and not url.fragment and url.path in ('', '/') and url.port is not None
            and loopback(configuration.VLM_HOST))
    except (ValueError, TypeError, AttributeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-config', required=True)
    parser.add_argument('--root-config', required=True)
    parser.add_argument('--work-permission', required=True, choices=('allow', 'deny'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--check', action='store_true',
        help='validate exact files and service configuration; no journal/listener/registry/backend I/O')
    args = parser.parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    settings = QueueSettings.from_file(args.queue_config)
    plan = RootLaunchPlan.load(args.root_config, settings)
    from . import config
    if (plan.registry != _base(config.SC_BASE) or plan.service != config.SC_SERVICE_ID
        or plan.resource != config.SC_RESOURCE_ID or config.EXO_ENABLED or not local_backends(config)):
        raise ValueError('Root configuration differs from this local service')
    if args.check:
        # Source hashes are public provenance, never an identity/capacity receipt.
        print(json.dumps({'queue_sha256': settings.source_sha256,
            'root_sha256': plan.sha256, 'checked': True, 'activated': False}))
        return
    from . import server
    import uvicorn
    if server.mgr is not None or any(getattr(server, name) is not None for name in (
        'inference_startup', 'inference_runtime', 'inference_dispatcher')):
        raise RuntimeError('Root launcher requires a fresh service process')
    if not isinstance(server.SC_TOKEN, str) or not server.SC_TOKEN.strip():
        raise ValueError('Explicit own service credential is required')
    store = RootIntentStore(plan.journal, registry=plan.registry,
        actor=plan.actor, service=plan.service)
    client = RegistryRootClient(store, token=server.SC_TOKEN)
    try:
        server.inference_startup = RootQueueStartup(settings, client, plan.models,
            lambda: confirm_cold(server), args.work_permission == 'allow')
        logging.getLogger('root-launch').info('Root config sha256=%s queue sha256=%s',
            plan.sha256, settings.source_sha256)
        uvicorn.run(server.app, host=args.host,
            port=args.port if args.port is not None else server.SERVICE_PORT, workers=1)
    finally:
        # Once attached, safe runtime shutdown owns journal/client close. An
        # observer exception must not close a live controller's control channel.
        if server.inference_runtime is None:
            client.close()


if __name__ == '__main__':
    main()
