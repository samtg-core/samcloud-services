"""Actual CPU-only receipt child and SQLite locks; no fleet/model/lease work."""

import copy
import json
import os
import sqlite3
import subprocess
import sys
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from . import test_exo_group_host_receipts as fixtures
from .exo_groups import host_receipts
from .exo_groups.host_receipts import observe_local_attempt
from .exo_groups.receipt_files import collect_attempt, ReceiptUnavailable


class LocalReceiptControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.ReceiptControls()
        await self.fixture.asyncSetUp()
        self.addAsyncCleanup(self.fixture.asyncTearDown)

    def competing_writer(self):
        code = '''import sqlite3,sys,json
c=sqlite3.connect(sys.argv[1],timeout=0)
try:
 c.execute('BEGIN IMMEDIATE');print(json.dumps(True));c.rollback()
except sqlite3.OperationalError:print(json.dumps(False))
finally:c.close()
'''
        return json.loads(subprocess.check_output((sys.executable, '-I', '-S', '-B', '-c', code,
                                                  str(self.fixture.journal.path)), text=True))

    async def test_actual_isolated_collector_returns_same_exact_metadata(self):
        expected = collect_attempt(self.fixture.spec)
        result = await observe_local_attempt(self.fixture.spec)
        self.assertEqual(result, expected)
        self.assertEqual(result['phase'], 'prepared')
        self.assertIsNone(result['started'])
        self.assertIsNone(result['cleanup'])
        self.assertIsNone(self.fixture.actor.process)

    async def test_real_observer_cannot_drop_the_parent_gateway_writer_lock(self):
        connection = sqlite3.connect(self.fixture.journal.path)
        try:
            connection.execute('BEGIN IMMEDIATE')
            self.assertFalse(self.competing_writer())
            result = await observe_local_attempt(self.fixture.spec)
            self.assertEqual(result['phase'], 'prepared')
            self.assertFalse(self.competing_writer(), 'receipt observation released the parent writer lock')
        finally:
            connection.rollback(); connection.close()
        self.assertTrue(self.competing_writer())

    async def test_same_process_raw_descriptor_is_the_negative_lock_loss_control(self):
        connection = sqlite3.connect(self.fixture.journal.path)
        try:
            connection.execute('BEGIN IMMEDIATE')
            self.assertFalse(self.competing_writer())
            # Actual old production call path: its raw descriptor close drops
            # this process's POSIX locks, even though SQLite thinks it owns one.
            collect_attempt(self.fixture.spec)
            self.assertTrue(self.competing_writer())
        finally:
            connection.rollback(); connection.close()

    async def test_fixed_isolated_command_has_no_inherited_credential_or_caller_code(self):
        async def run(argv, raw, *, environment):
            self.assertEqual(argv[:5], (sys.executable, '-I', '-S', '-B', '-c'))
            self.assertEqual(set(environment), {'PATH', 'PYTHONDONTWRITEBYTECODE'})
            self.assertNotIn('private-test-bearer', repr((argv, raw, environment)))
            request = json.loads(raw)
            frame = {**request, 'report': collect_attempt(request['spec'])}
            frame.pop('spec')
            return host_receipts.PREFIX + json.dumps(frame).encode() + b'\n'
        with patch.dict(os.environ, {'SC_TOKEN': 'private-test-bearer'}), \
                patch.object(host_receipts, 'run_probe', new=AsyncMock(side_effect=run)) as runner:
            await observe_local_attempt(self.fixture.spec)
            self.assertEqual(runner.await_count, 1)

    async def test_nonce_source_binding_and_frame_bounds_are_enforced(self):
        actual = collect_attempt(self.fixture.spec)
        for kind in ('nonce', 'source', 'binding', 'uid', 'boot', 'duplicate', 'oversize'):
            async def run(_argv, raw, *, environment):
                request = json.loads(raw)
                frame = {'request_id': request['request_id'], 'collector_sha256': request['collector_sha256'],
                         'report': copy.deepcopy(actual)}
                if kind == 'nonce': frame['request_id'] = 'old'
                elif kind == 'source': frame['collector_sha256'] = 'f' * 64
                elif kind == 'binding': frame['report']['binding']['lease_id'] = 'another'
                elif kind == 'uid': frame['report']['uid'] = True
                elif kind == 'boot': frame['report']['boot_id'] = 'another'
                data = host_receipts.PREFIX + json.dumps(frame).encode() + b'\n'
                if kind == 'duplicate': data += data
                elif kind == 'oversize': data = b'x' * (host_receipts.MAX_FRAME + 1)
                return data
            with self.subTest(kind=kind), patch.object(host_receipts, 'run_probe', new=AsyncMock(side_effect=run)):
                with self.assertRaises(ReceiptUnavailable):
                    await observe_local_attempt(self.fixture.spec)

    async def test_failed_observer_does_not_expose_diagnostics_or_retry(self):
        with patch.object(host_receipts, 'run_probe', new=AsyncMock(side_effect=OSError('private diagnostic'))) as runner:
            with self.assertRaises(ReceiptUnavailable) as caught:
                await observe_local_attempt(self.fixture.spec)
            self.assertNotIn('private diagnostic', str(caught.exception))
            self.assertEqual(runner.await_count, 1)


class HistoricalPreparationControls(IsolatedAsyncioTestCase):
    def test_fenced_prepare_uses_original_runtime_and_binds_both_records(self):
        import hashlib
        from pathlib import Path
        import tempfile
        from .exo_groups.receipt_files import verify_prepare_only

        launch = dict(group_id='group-old', generation=1, lease_id='lease-old',
                      resource_id='slice/gpu', rank_manifest_sha256='1' * 64)
        expected = dict(binding=launch, member={'rank': 'slice'})
        runtime = 'a' * 64
        argv = ['/old/runtime/python', '/old/runtime/runner']
        deployment = dict(kind='prepared_host_configuration', binding=expected,
                          host_boot='boot', runtime_sha256=runtime, argv=argv)
        prepared = dict(state='prepared_not_launched',
                        binding=dict(launch, expected=expected, runtime_sha256=runtime),
                        argv_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest())
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp).resolve() / 'old'
            attempt = session / 'attempt'
            session.mkdir(mode=0o700)
            attempt.mkdir(mode=0o700)
            def save(path, value):
                path.write_text(json.dumps(value))
                path.chmod(0o600)
            def check():
                verify_prepare_only(str(session), os.getuid(), 'unconfirmed', None,
                                    dict(launch, expected=expected, runtime_sha256='c' * 64), 'boot')
            save(session / 'deployment.json', deployment)
            save(attempt / 'prepared.json', prepared)
            check()
            for change in (dict(runtime_sha256='b' * 64), dict(argv=['another']),
                           dict(host_boot='another'), dict(binding={})):
                with self.subTest(change=change):
                    save(session / 'deployment.json', deployment | change)
                    with self.assertRaises(ReceiptUnavailable):
                        check()
