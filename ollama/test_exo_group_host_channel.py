"""Read-only host channel, including actual isolated stdlib subprocess probes."""

import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

from .exo_groups import staging_files
from .exo_groups.host_channel import (
    AuthenticatedHostReader, HostRoute, MAX_DIAGNOSTICS, MAX_FRAME, PREFIX, run_probe,
)
from .exo_groups.staging import HostFiles, StagingUnavailable


class HostChannelControls(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.source = Path(staging_files.__file__).read_bytes()
        self.pin = hashlib.sha256(self.source).hexdigest()
        self.route = HostRoute('slice', 'slice/gpu-0', os.getuid(),
                               ('/explicit/own/ssh-route', 'slice --'), sys.executable)
        policy = {'root': '/explicit/model', 'files': [], 'views': {}}
        self.host = HostFiles('slice', 'slice/gpu-0', os.getuid(), 100, policy, policy)
        self.calls = []
        self.reply = lambda request: {'request_id': request['request_id'],
            'collector_sha256': request['collector_sha256'], 'report': {
                'rank': 'slice', 'resource_id': 'slice/gpu-0', 'uid': os.getuid()}}

    async def runner(self, argv, raw, *, environment):
        self.calls.append((argv, raw, environment))
        request = json.loads(raw)
        return b'route banner\n' + PREFIX + json.dumps(self.reply(request)).encode() + b'\n'

    def reader(self, **kwargs):
        return AuthenticatedHostReader((self.route,), collector_sha256=self.pin,
            environment={}, runner=self.runner, **kwargs)

    async def test_exact_route_and_static_isolated_probe_with_policy_only_on_stdin(self):
        reader = self.reader()
        report = await reader(self.host, None)
        self.assertEqual(report['uid'], os.getuid())
        argv, raw, environment = self.calls[0]
        self.assertEqual(argv[:-1], self.route.command)
        command = shlex.split(argv[-1])
        self.assertEqual(command[:5], [sys.executable, '-I', '-S', '-B', '-c'])
        self.assertTrue(command[5].startswith(self.source.decode()))
        self.assertNotIn('/explicit/model', command[5])
        self.assertEqual(environment, {})
        self.assertEqual(json.loads(raw)['spec'], asdict(self.host))
        await reader(self.host, report)
        self.assertEqual(len(self.calls), 2)
        self.assertNotEqual(json.loads(self.calls[0][1])['request_id'],
                            json.loads(self.calls[1][1])['request_id'])

    async def test_complete_runtime_report_and_previous_cache_exceed_old_source_only_bound(self):
        original = self.reply
        def full_report(request):
            response = original(request)
            response['report']['identity_cache'] = 'x' * (9 * 1024 * 1024)
            return response
        self.reply = full_report
        reader = self.reader()
        first = await reader(self.host,None)
        self.assertGreater(len(self.calls[0][1]),0)
        self.assertGreater(len(json.dumps(first)),8 * 1024 * 1024)
        await reader(self.host,first)
        self.assertGreater(len(self.calls[1][1]),8 * 1024 * 1024)
        self.assertLess(len(self.calls[1][1]),MAX_FRAME)

    async def test_wrong_route_or_principal_refuses_before_any_process(self):
        for host in (replace(self.host, uid=os.getuid() + 1),
                     replace(self.host, rank='wafer'),
                     replace(self.host, resource_id='foreign/gpu-0')):
            with self.subTest(host=host), self.assertRaises(StagingUnavailable):
                await self.reader()(host, None)
        self.assertEqual(self.calls, [])

    async def test_stale_reply_changed_collector_or_host_never_produces_observation(self):
        original = self.reply
        for field, value in [('request_id', 'old'), ('collector_sha256', 'f' * 64),
                             ('report', {'rank': 'slice', 'resource_id': 'slice/gpu-0', 'uid': -1})]:
            self.reply = lambda request, field=field, value=value: original(request) | {field: value}
            with self.subTest(field=field), self.assertRaises(StagingUnavailable):
                await self.reader()(self.host, None)
        self.assertEqual(len(self.calls), 3)  # No failure-driven resubmission.

    async def test_ambiguous_oversized_nonfinite_missing_or_duplicate_frames_refuse(self):
        raw_cases = [b'', PREFIX + b'{"x":NaN}', PREFIX + b'{"x":1,"x":2}',
                     PREFIX + b'{}\n' + PREFIX + b'{}', b'x' * (MAX_FRAME + 1)]
        for output in raw_cases:
            count = []
            async def runner(*_args, **_kwargs):
                count.append(1)
                return output
            reader = AuthenticatedHostReader((self.route,), collector_sha256=self.pin,
                                             environment={}, runner=runner)
            with self.subTest(size=len(output)), self.assertRaises(StagingUnavailable) as error:
                await reader(self.host, None)
            self.assertEqual(str(error.exception), 'authenticated host staging unavailable')
            self.assertEqual(count, [1])

    def test_source_pin_and_environment_are_explicit_and_frozen(self):
        with self.assertRaises(ValueError):
            AuthenticatedHostReader((self.route,), collector_sha256='a' * 64, environment={})
        for key in ('SAMCLOUD_AUTH_HDR', 'PRIVATE_TOKEN', 'SSH_AUTH_SOCK'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                AuthenticatedHostReader((self.route,), collector_sha256=self.pin,
                                         environment={key: 'never forward'})
        with self.assertRaises(ValueError):
            HostRoute('slice', 'slice/gpu-0', 0, ('relative-route',), sys.executable)
        with self.assertRaises(ValueError):
            HostRoute('slice', 'slice/gpu-0', 0, ('/absolute',), '/path/../python')

    async def test_transport_error_exposes_no_command_or_private_diagnostic_and_does_not_retry(self):
        count = []
        async def runner(*_args, **_kwargs):
            count.append(1)
            raise RuntimeError('private host diagnostic and bearer must not escape')
        reader = AuthenticatedHostReader((self.route,), collector_sha256=self.pin,
                                         environment={}, runner=runner)
        with self.assertRaises(StagingUnavailable) as error:
            await reader(self.host, None)
        self.assertEqual(str(error.exception), 'authenticated host staging unavailable')
        self.assertEqual(count, [1])

    @unittest.skipUnless(sys.platform == 'darwin', 'Actual Darwin filesystem/boot collector')
    async def test_actual_isolated_probe_hashes_sealed_files_and_cache_over_pipes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            route = root / 'test_route.py'
            route.write_text('import os,shlex,sys\na=shlex.split(sys.argv[-1]);os.execv(a[0],a)\n')
            host = root / 'host'
            host.mkdir()
            (host / 'pinned').write_bytes(b'exact bytes')
            (host / 'pinned').chmod(0o400)
            host.chmod(0o500)
            policy = {'root': str(host), 'files': [{'path': 'pinned', 'bytes': 11,
                'sha256': hashlib.sha256(b'exact bytes').hexdigest(), 'link': None}], 'views': {}}
            endpoint = replace(self.route, command=(sys.executable, str(route)))
            reader = AuthenticatedHostReader((endpoint,), collector_sha256=self.pin, environment={})
            try:
                spec = replace(self.host, model=policy, runtime=policy)
                report = await reader(spec, None)
                self.assertTrue(report['model']['readonly'])
                self.assertEqual(report['model']['hashed_bytes'], 11)
                # A complete runtime's prior identities can exceed the old
                # 8MiB source-only request bound. Exercise the actual remote
                # stdlib reader, not only the injected transport.
                report['extra_identity_cache'] = 'x' * (9 * 1024 * 1024)
                cached = await reader(spec, report)
                self.assertEqual(cached['model']['hashed_bytes'], 0)
                self.assertEqual(cached['boot_id'], report['boot_id'])
            finally:
                host.chmod(0o700)

    async def test_actual_pipe_bounds_stop_only_the_owned_observer(self):
        processes = []
        async def spawn(*args, **kwargs):
            process = await asyncio.create_subprocess_exec(*args, **kwargs)
            processes.append(process)
            return process
        for channel, limit in [('stdout', MAX_FRAME), ('stderr', MAX_DIAGNOSTICS)]:
            code = ('import sys;sys.stdin.buffer.read();'
                    f'sys.{channel}.buffer.write(b"x" * {limit + 1});sys.{channel}.flush()')
            with self.subTest(channel=channel), self.assertRaises(StagingUnavailable):
                await run_probe((sys.executable, '-I', '-S', '-c', code), b'{}',
                                environment={}, spawn=spawn)
            self.assertIsNotNone(processes[-1].returncode)

    async def test_actual_probe_cancellation_does_not_relaunch_or_leave_local_observer(self):
        processes, started = [], asyncio.Event()
        loop, errors = asyncio.get_running_loop(), []
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        async def spawn(*args, **kwargs):
            process = await asyncio.create_subprocess_exec(*args, **kwargs)
            processes.append(process)
            started.set()
            return process
        # Blocking on stdin after consuming one byte, no timer or model process.
        code = 'import os;os.read(0,1);r,w=os.pipe();os.read(r,1)'
        try:
            task = asyncio.create_task(run_probe((sys.executable, '-I', '-S', '-c', code), b'x',
                                                 environment={}, spawn=spawn))
            await started.wait()
            # Let run_probe own the returned handle before cancelling its reads.
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(original_handler)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
