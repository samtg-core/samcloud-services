"""Compiler ownership/protocol with real sealed files and an isolated CPU child.

The child supplies an existing wire fixture; it is not a native type/model test.
The shipping worker is separately checked with the actual frozen interpreter.
"""

import asyncio
from dataclasses import asdict, replace
import os
import sys
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from . import test_exo_group_deployment as deployment_controls
from . import test_exo_group_instance as instance_controls
from .exo_groups import compiler, contract_worker
from .exo_groups.compiler import CompilerUnavailable, ContractCompiler
from .inference_arbiter import Conflict


class CompilerControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = deployment_controls.DeploymentControls()
        self.fixture.addAsyncCleanup = self.addAsyncCleanup
        await self.fixture.asyncSetUp()
        self.supervisor = self.fixture.runtime.queue.supervisor
        self.launches = tuple(replace(self.fixture.binding, rank=member.rank,
            pipeline_rank=member.pipeline_rank, resource_id=member.resource_id, layers=member.layers,
            lease_id='lease_'+f'{i+1:016x}') for i, member in enumerate(sorted(
                self.fixture.plan.members, key=lambda m: m.pipeline_rank)))
        native = instance_controls.PrivateInstanceControls()
        native.setUp()
        self.reference = asdict(native.contract)
        self.mode, self.calls, self.compilers = 'ok', [], []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        self.release.set()
        for owner in self.compilers:
            if owner._task is not None:
                await asyncio.gather(owner._task, return_exceptions=True)

    def owner(self):
        async def spawn(*argv, **options):
            self.calls.append((argv, options))
            self.entered.set()
            await self.release.wait()
            code = '''import hashlib,json,sys
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
raw=sys.stdin.buffer.read();r=json.loads(raw);c=CONTRACT
c.update(plan_sha256=hashlib.sha256(canon(r['plan'])).hexdigest(),group_id=r['group_id'],generation=r['generation'],runtime_sha256=r['runtime_sha256'])
v={'nonce':r['nonce'],'request_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':hashlib.sha256(canon(r['source_pins'])).hexdigest(),'contract':c,'contract_sha256':hashlib.sha256(canon(c)).hexdigest(),'native_compute_imported':False}
if MODE=='error':sys.stderr.write('private fake diagnostic');sys.exit(1)
if MODE=='nonce':v['nonce']='0'*64
if MODE=='source':v['source_sha256']='0'*64
if MODE=='native':v['native_compute_imported']=True
if MODE=='contract':v['contract_sha256']='0'*64
if MODE=='huge':sys.stdout.buffer.write(b'x'*524289)
else:sys.stdout.buffer.write(canon(v))
'''.replace('CONTRACT', repr(self.reference)).replace('MODE', repr(self.mode))
            return await asyncio.create_subprocess_exec(sys.executable, '-I', '-S', '-B', '-c', code,
                **options)

        value = ContractCompiler(self.fixture.config, self.supervisor, spawn=spawn)
        self.compilers.append(value)
        return value

    async def test_fixed_interpreter_and_minimal_env_never_inherit_credentials(self):
        owner = self.owner()
        with patch.dict(os.environ, {'SC_TOKEN': 'must-not-forward', 'OTHER_SECRET': 'also-not-forward'}):
            result = await owner.compile(self.launches)
        argv, options = self.calls[0]
        self.assertEqual(argv[:4], (str(self.fixture.config.runtime_root/'python/bin/python3.13'), '-I', '-S', '-B'))
        self.assertEqual(argv[4], str(compiler.Path(compiler.__file__).with_name('contract_worker.py')))
        self.assertFalse(any(any(s in k for s in ('TOKEN', 'SECRET', 'AUTH', 'BEARER')) for k in options['env']))
        self.assertTrue(options['start_new_session'])
        self.assertEqual(result.group_id, self.launches[0].group_id)
        self.assertEqual(owner.process.returncode, 0)
        self.assertEqual(await owner.compile(self.launches), result)
        self.assertEqual(len(self.calls), 1)

    async def test_helper_cannot_advertise_a_context_different_from_the_host_profile(self):
        # The isolated fake helper intentionally returns the original card.
        self.fixture.config.served_context_length = 1024
        with self.assertRaises(CompilerUnavailable):
            await self.owner().compile(self.launches)
        self.assertEqual(len(self.calls), 1)

    async def test_observer_disconnect_keeps_one_supervisor_owned_compilation(self):
        owner = self.owner()
        self.release.clear()
        observer = asyncio.create_task(owner.compile(self.launches))
        await self.entered.wait()
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertIn(owner._task, self.supervisor.preparations)
        self.assertFalse(owner._task.done())
        self.release.set()
        result = await owner.wait()
        self.assertEqual(result.runtime_sha256, self.fixture.config.runtime_sha256)
        self.assertEqual(len(self.calls), 1)

    async def test_changed_group_never_creates_another_helper(self):
        owner = self.owner()
        await owner.compile(self.launches)
        changed = tuple(replace(binding, group_id='group_'+'b'*24) for binding in self.launches)
        with self.assertRaises(Conflict):
            await owner.compile(changed)
        self.assertEqual(len(self.calls), 1)

    async def test_wrong_response_identity_native_import_or_bounds_retain_same_failure(self):
        for mode in ('nonce', 'source', 'native', 'contract', 'huge', 'error'):
            self.mode = mode
            owner = self.owner()
            count = len(self.calls)
            with self.subTest(mode=mode), self.assertRaises(CompilerUnavailable) as caught:
                await owner.compile(self.launches)
            self.assertNotIn('private fake', str(caught.exception))
            self.assertIsNotNone(owner.process.returncode)
            with self.assertRaises(CompilerUnavailable):
                await owner.compile(self.launches)
            self.assertEqual(len(self.calls), count+1)

    async def test_source_change_refuses_before_process_creation(self):
        owner = self.owner()
        changed = dict(owner.pins, **{'ollama/exo_groups/compiler.py': '0'*64})
        with patch.object(compiler, 'source_pins', return_value=changed):
            with self.assertRaises(CompilerUnavailable):
                await owner.compile(self.launches)
        self.assertFalse(self.calls)

    async def test_changed_metadata_or_boolean_member_refuses_before_process_creation(self):
        owner = self.owner()
        for changes in ({'generation': True}, {'pipeline_rank': False}, {'lease_id': 'not-a-lease'}):
            invalid = (replace(self.launches[0], **changes), *self.launches[1:])
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                await owner.compile(invalid)
        path = self.fixture.config.model_root/'parent-manifest.json'
        raw = path.read_bytes()
        path.chmod(0o600)
        path.write_bytes(b'changed')
        path.chmod(0o400)
        try:
            with self.assertRaises(CompilerUnavailable):
                await owner.compile(self.launches)
        finally:
            path.chmod(0o600)
            path.write_bytes(raw)
            path.chmod(0o400)
        self.assertFalse(self.calls)


class WorkerDecoderControls(TestCase):
    def test_worker_rejects_ambiguous_or_nonfinite_metadata(self):
        for raw in (b'[]', b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e9999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                contract_worker.decode(raw)

    def test_worker_audit_rejects_network_child_and_write_actions(self):
        for event, args in (('socket.connect', ()), ('socket.bind', ()), ('socket.sendto', ()),
                            ('subprocess.Popen', ()), ('os.fork', ()), ('os.mkdir', ()),
                            ('os.rename', ()), ('os.remove', ()), ('os.symlink', ()), ('os.chmod', ()),
                            ('open', ('fixture', 'w', 0)), ('open', ('fixture', None, os.O_CREAT))):
            with self.subTest(event=event), self.assertRaises(RuntimeError):
                contract_worker.audit(event, args)
