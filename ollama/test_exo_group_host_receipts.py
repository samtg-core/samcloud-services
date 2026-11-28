"""CPU host journal/file/SSH-probe receipt controls; no lease/model calls."""

from dataclasses import asdict
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

import httpx

from . import test_exo_group_member as fixtures
from . import test_exo_group_host_authority as alarms
from . import test_exo_group_rank_lifecycle as processes
from .exo_groups.cleanup import NoStartExpectation
from .exo_groups.host_authority import HostAuthorityAdapter,HostPlaneReader
from .exo_groups.host_channel import HostRoute,run_probe
from .exo_groups.host_receipts import AuthenticatedReceiptReader,PREFIX,prepared_from_report
from .exo_groups.plan import BindingRefused
from .exo_groups.rank_lifecycle import RankLifecycle,RankSpec,darwin_boot_id
from .exo_groups.rank_process import Listener,RunBinding
from .exo_groups.rank_runtime import HostRankRuntime,RuntimeObservation,launch_sha256,record_host_attempt
from .exo_groups import receipt_files
from .exo_groups.receipt_files import collect_attempt,ReceiptUnavailable
from .inference_arbiter import Arbiter


class ReceiptControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        f=fixtures.MemberAuthorityControls()
        f.setUp()
        self.plan,self.expected,self.projection=f.plan,f.expected,f.projection
        self.temp=tempfile.TemporaryDirectory(dir='/private/tmp')
        self.root=Path(self.temp.name).resolve()
        self.boot=darwin_boot_id()
        self.journal=Arbiter(self.root/'queue.sqlite')
        binding=RunBinding.from_expected(self.expected,'b'*64)
        self.child=processes.FakeProcess(binding)
        spec=RankSpec(binding,(sys.executable,'-I','-S','-c','pass'),self.root,
                      (Listener('127.0.0.1',39999),),0)
        self.actor=RankLifecycle(self.expected,spec,self.root/'rank',memory_reader=lambda:100,
            port_reader=lambda _:True,boot_reader=lambda:self.boot,monotonic=lambda:101,
            spawner=lambda *_args,**_kwargs:self.child,emit=lambda _:None)
        def http(request):
            if request.url.path.endswith('/auth/verify'):
                return httpx.Response(200,json={'valid':True,'owner_id':'user:999'})
            return httpx.Response(200,json=self.projection)
        plane=HostPlaneReader(lambda:'Bearer fake-host',owner_id='user:999',transport=httpx.MockTransport(http))
        adapter=HostAuthorityAdapter(self.actor,plane,alarm_factory=alarms.FakeAlarm)
        async def verify():
            return RuntimeObservation('b'*64,launch_sha256(self.actor,{}),self.boot,os.getuid())
        class Watch:
            def __init__(self,*_):pass
            def close(self):pass
        self.owner=HostRankRuntime(self.actor,adapter,verify_runtime=verify,environment={},
            boot_id=self.boot,watcher_factory=Watch)
        record_host_attempt(self.owner,self.journal)
        self.spec={'resource_id':binding.resource_id,'group_id':binding.group_id,
            'generation':binding.generation,'lease_id':binding.lease_id,'runtime_sha256':'b'*64,
            'launch_sha256':self.owner.launch_sha256,'boot_id':self.boot,'uid':os.getuid(),
            'state_path':str(self.actor.state),'journal_path':str(self.root/'queue.sqlite')}
        source=files('ollama.exo_groups').joinpath('receipt_files.py').read_bytes()
        self.source_sha=hashlib.sha256(source).hexdigest()
        self.route=HostRoute('slice',binding.resource_id,os.getuid(),('/unused-ssh',),sys.executable)
        self.requests=[]
        self.mutate=None
        async def runner(argv,raw,*,environment):
            request=json.loads(raw)
            self.requests.append(request)
            report=collect_attempt(request['spec'])
            if self.mutate is not None:
                self.mutate(request,report)
            frame={'request_id':request['request_id'],'collector_sha256':request['collector_sha256'],
                   'report':report}
            return PREFIX+json.dumps(frame).encode()+b'\n'
        self.runner=runner
        self.reader=AuthenticatedReceiptReader(self.route,collector_sha256=self.source_sha,
                                               environment={},runner=runner)

    async def asyncTearDown(self):
        if self.actor.process is not None:
            self.child.code=0
        self.owner.adapter.alarm.close()
        await self.owner.adapter.reader.close()
        self.journal.close()
        self.temp.cleanup()

    async def read(self,reader=None,**changes):
        args={k:self.spec[k] for k in ('journal_path','state_path','boot_id','launch_sha256','runtime_sha256')}
        args.update(changes)
        return await (reader or self.reader).read(self.plan,binding=self.expected.binding,**args)

    def test_historical_journal_directory_churn_reobserves_only_this_attempt(self):
        original_read = receipt_files._read
        marker = self.root / 'unrelated-queue-write'
        reads = 0
        def change_journal_parent(directory, name, *args, **kwargs):
            nonlocal reads
            if name == 'host-prepared.json':
                reads += 1
                if reads == 1 or persistent:
                    marker.touch()
                    marker.unlink()
            return original_read(directory, name, *args, **kwargs)

        expected = collect_attempt(self.spec)
        for persistent in (False, True):
            reads = 0
            with self.subTest(persistent=persistent), mock.patch.object(
                    receipt_files, '_read', side_effect=change_journal_parent):
                if persistent:
                    with self.assertRaises(ReceiptUnavailable):
                        receipt_files.collect_historical_attempt(self.spec)
                else:
                    report = receipt_files.collect_historical_attempt(self.spec)
                    self.assertEqual(report, {**expected, 'observed_host_boot': self.boot})
                self.assertEqual(reads, 2)

        reads = 0
        with mock.patch.object(receipt_files, '_read', side_effect=change_journal_parent):
            with self.assertRaises(ReceiptUnavailable):
                collect_attempt(self.spec)
        self.assertEqual(reads, 1)

    def metadata_events(self, count):
        # Actual immutable private files, not a mocked directory enumeration.
        for number in range(1, count + 1):
            path = self.actor.state / f'event-{number:06d}.json'
            if not path.exists():
                path.write_text('{"type":"authority_read_required"}')
                path.chmod(0o600)

    async def test_complete_history_above_ten_thousand_events_retains_exact_cleanup(self):
        await self.owner.on_connect()
        await self.owner.start()
        self.child.code = 0
        self.actor.process_exited()
        self.metadata_events(10168)
        before = (self.actor.state / 'cleanup-evidence.json').read_bytes()
        try:
            result = await self.read()
        except BindingRefused:
            self.fail('A complete 10,168-event history within the byte budget must remain readable')
        self.assertEqual(result.cleanup_bytes, before)
        self.assertEqual(result.started.child, self.child.identity)
        self.assertEqual(len(list(self.actor.state.glob('event-*.json'))), 10168)

    async def test_extra_launch_after_ten_thousand_events_is_not_hidden(self):
        await self.owner.on_connect()
        await self.owner.start()
        self.child.code = 0
        self.actor.process_exited()
        original = next(path.read_bytes() for path in self.actor.state.glob('event-*.json')
                        if json.loads(path.read_bytes()).get('type') == 'rank_started')
        self.metadata_events(10168)
        path = self.actor.state / 'event-010169.json'
        path.write_bytes(original)
        path.chmod(0o600)
        with self.assertRaisesRegex(ReceiptUnavailable, 'ambiguous host launch'):
            collect_attempt(self.spec)

    async def test_explicit_larger_file_budget_still_refuses_oversize_index(self):
        self.metadata_events(20001)
        with self.assertRaisesRegex(ReceiptUnavailable, 'event index'):
            collect_attempt(self.spec)

    async def test_preparation_read_is_independent_and_has_no_compute_or_cleanup_claim(self):
        result=await self.read()
        self.assertEqual(result.prepared,self.owner.prepared)
        self.assertIsNone(result.started)
        self.assertIsNone(result.cleanup_bytes)
        self.assertEqual(result.phase,'prepared')
        self.assertIsNone(self.actor.process)
        self.assertIsInstance(result.prepared,NoStartExpectation)
        self.assertEqual(self.journal.db.execute('SELECT phase FROM exo_host_rank_attempts').fetchone()[0],'prepared')

    async def test_fenced_no_start_bytes_are_read_exactly_not_synthesized(self):
        receipt=await self.owner.attest_never_started()
        result=await self.read()
        raw=(self.actor.state/'cleanup-evidence.json').read_bytes()
        self.assertEqual(result.cleanup_bytes,raw)
        self.assertEqual(result.cleanup_sha256,receipt['evidence_sha256'])
        self.assertIsNone(result.started)
        self.assertEqual(result.phase,'launch_fenced')

    async def test_actual_launch_pin_comes_from_independent_launch_journal(self):
        await self.owner.on_connect()
        await self.owner.start()
        result=await self.read()
        self.assertEqual(result.started.child,self.child.identity)
        self.assertEqual(result.started.baseline_in_use_bytes,100)
        self.assertIsNone(result.cleanup_bytes)
        self.child.code=0
        self.actor.process_exited()
        self.assertEqual((await self.read()).cleanup_sha256,self.actor.receipt['evidence_sha256'])

    async def test_once_committed_unknown_launch_is_not_reported_as_no_start(self):
        self.owner._transition('prepared','launch_committed')
        result=await self.read()
        self.assertEqual(result.phase,'launch_committed')
        self.assertIsNone(result.started)
        self.assertIsNone(result.cleanup_bytes)

    async def test_nonce_binding_and_remote_identity_cannot_be_replayed(self):
        for name,value in (('uid',os.getuid()+1),('boot_id','other')):
            self.mutate=lambda _request,report:report.update({name:value})
            with self.assertRaises(BindingRefused):
                await self.read()
        self.mutate=lambda request,_report:request.update(request_id='old-frame')
        with self.assertRaises(BindingRefused):
            await self.read()

    async def test_prepared_file_must_equal_exact_host_index(self):
        p=self.actor.state/'host-prepared.json'
        value=json.loads(p.read_bytes())
        value['attempt_sha256']='a'*64
        p.write_text(json.dumps(value))
        with self.assertRaises(ReceiptUnavailable):
            collect_attempt(self.spec)

    async def test_unknown_legacy_row_and_missing_attempt_prove_nothing(self):
        self.journal.db.execute("UPDATE exo_host_rank_attempts SET phase='unknown'")
        with self.assertRaises(ReceiptUnavailable):
            collect_attempt(self.spec)
        self.journal.db.execute('DELETE FROM exo_host_rank_attempts')
        with self.assertRaises(ReceiptUnavailable):
            collect_attempt(self.spec)

    async def test_receipt_symlink_or_public_write_is_refused(self):
        p=self.actor.state/'host-prepared.json'
        raw=p.read_bytes()
        p.chmod(0o666)
        with self.assertRaises(ReceiptUnavailable):
            collect_attempt(self.spec)
        p.unlink()
        target=self.root/'fake.json'
        target.write_bytes(raw)
        p.symlink_to(target)
        with self.assertRaises(OSError):
            collect_attempt(self.spec)

    async def test_changed_state_during_read_establishes_no_observation(self):
        original=os.listdir
        def change(fd):
            result=original(fd)
            (self.actor.state/'new-event').write_text('new')
            return result
        with mock.patch('ollama.exo_groups.receipt_files.os.listdir',side_effect=change):
            with self.assertRaises(ReceiptUnavailable):
                collect_attempt(self.spec)

    async def test_wrong_route_or_collector_pin_refuses_before_probe(self):
        with self.assertRaises(ValueError):
            AuthenticatedReceiptReader(self.route,collector_sha256='a'*64,environment={})
        with self.assertRaises(ValueError):
            AuthenticatedReceiptReader(self.route,collector_sha256=self.source_sha,
                                       environment={'SAMCLOUD_AUTH_HDR':'never-forward'})
        with self.assertRaises(BindingRefused):
            await self.read(runtime_sha256='a'*64)
        self.assertIsNone(self.actor.process)

    async def test_actual_isolated_python_probe_preserves_input_and_reads_only_receipts(self):
        async def execute(argv,raw,*,environment):
            return await run_probe(tuple(shlex.split(argv[-1])),raw,environment=environment)
        reader=AuthenticatedReceiptReader(self.route,collector_sha256=self.source_sha,
                                          environment={},runner=execute)
        result=await self.read(reader)
        self.assertEqual(result.prepared,self.owner.prepared)
        self.assertFalse(self.actor.launch_attempted)

    async def test_foreign_preparation_plan_or_forged_negative_child_is_refused(self):
        value=asdict(self.owner.prepared)
        value['binding']['expected']['member']['memory_mb']=999
        with self.assertRaises(BindingRefused):
            prepared_from_report(self.plan,value)
        await self.owner.attest_never_started()
        self.mutate=lambda _request,report:report['cleanup'].update(sha256='f'*64)
        with self.assertRaises(BindingRefused):
            await self.read()

    async def test_event_metadata_has_an_aggregate_read_bound(self):
        path=self.actor.state/'event-999999.json'
        path.write_bytes(json.dumps({'type':'authority_observed','detail':'x'*2000}).encode())
        path.chmod(0o600)
        with mock.patch('ollama.exo_groups.receipt_files.MAX_EVENT_BYTES',1024):
            with self.assertRaises(ReceiptUnavailable):
                collect_attempt(self.spec)
