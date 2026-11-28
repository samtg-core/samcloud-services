"""Explicit single-process durable gateway launcher; never a hot conversion.

No deployed launch agent selects this module yet. The config, work permission,
own registry identity and cold provider inventory must all pass independently.
"""
import argparse

from .inference_bootstrap import QueueSettings, QueueStartup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-config',required=True,
                        help='absolute path to exact model/resource/reservation configuration')
    parser.add_argument('--work-permission',required=True,choices=('allow','deny'),
                        help='explicit operator permission; backend availability is not permission')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int)
    args = parser.parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    settings = QueueSettings.from_file(args.queue_config)
    # Validate the file before importing the service or opening a listener.
    from . import server
    import uvicorn
    if (server.inference_startup is not None or server.inference_runtime is not None or
            server.inference_dispatcher is not None):
        raise RuntimeError('Durable launcher requires a fresh service process')
    server.inference_startup = QueueStartup(settings,work_permitted=args.work_permission=='allow')
    # No reload/multi-worker fork: one journal/manager owns this local runner.
    uvicorn.run(server.app,host=args.host,
                port=args.port if args.port is not None else server.SERVICE_PORT,workers=1)


if __name__ == '__main__':
    main()
