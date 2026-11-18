"""CPU-only reboot recovery; no process, GPU or lease side effects."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import unittest
from . import test_exo_group_host_receipts as fixtures
from .exo_groups.cleanup import CleanupExpectation, validate_cleanup
from .exo_groups.host_receipts import prepared_from_report
from .exo_groups.plan import BindingRefused
from .exo_groups.prepare import canonical
from .exo_groups.rank_process import ProcessIdentity
from .exo_groups.reboot_cleanup import RebootExpectation, validate_reboot_cleanup
from .exo_groups.receipt_files import collect_attempt, collect_historical_attempt, ReceiptUnavailable

class RebootEvidenceControls(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.ReceiptControls.asyncSetUp
    asyncTearDown = fixtures.ReceiptControls.asyncTearDown

    def report(self):
        return collect_historical_attempt(self.spec, boot_reader=lambda:'11111111-2222-4333-8444-555555555555')

    async def prepared(self):
        await self.owner.on_connect();await self.owner.start()
        report=self.report();prepared=prepared_from_report(self.plan,report['prepared'])
        child=ProcessIdentity(**report['started']['child'])
        started=CleanupExpectation(prepared.binding,child,prepared.listeners,
                                   report['launch_intent']['baseline_in_use_bytes'],prepared.tolerance_bytes)
        self.pin=RebootExpectation(prepared,started)
        self.value={'kind':'observed_rank_rebooted',**asdict(self.pin.binding),
            'lease_released':False,'settlement_recorded':False,
            'observation':{'binding':asdict(self.pin.binding),'prepared':asdict(prepared),
                'child':asdict(child),'previous_host_boot':prepared.host_boot,
                'current_host_boot':'11111111-2222-4333-8444-555555555555',
                'booted_at':{'seconds':child.started_seconds+1,'microseconds':0},
                'observed_at':datetime.fromtimestamp(child.started_seconds+2,timezone.utc).isoformat(),
                'fenced_generation':self.pin.binding.generation+1,
                'kernel_inventory_complete':True,'boot_rechecked':True,
                'native_processes':[],'reserved_sockets':[],
                'listeners_closed':[True]*len(prepared.listeners)}}

    async def test_reboot_is_end_proof_not_fake_exit_or_memory_comparison(self):
        await self.prepared();raw=canonical(self.value)
        self.assertEqual(validate_reboot_cleanup(raw,self.pin),json.loads(raw))
        with self.assertRaises(BindingRefused):validate_cleanup(raw,self.pin.started)

    async def test_normal_collector_refuses_reboot_and_history_is_only_metadata(self):
        await self.prepared()
        with self.assertRaises(ReceiptUnavailable):
            collect_attempt(self.spec,boot_reader=lambda:'11111111-2222-4333-8444-555555555555')
        report=self.report();self.assertIsNone(report['cleanup'])
        self.assertEqual(report['boot_id'],self.boot);self.assertNotEqual(report['observed_host_boot'],self.boot)

    async def test_reused_pid_wrong_boot_and_boot_before_original_birth_are_refused(self):
        await self.prepared()
        for key,value in [('child',{**self.value['observation']['child'],'started_seconds':123}),
                          ('current_host_boot',self.boot),
                          ('booted_at',{'seconds':self.pin.started.child.started_seconds,'microseconds':0}),
                          ('previous_host_boot','22222222-2222-4222-8222-222222222222')]:
            with self.subTest(key=key):
                data=deepcopy(self.value);data['observation'][key]=value
                with self.assertRaises(BindingRefused):validate_reboot_cleanup(canonical(data),self.pin)

    async def test_unknown_inventory_live_socket_or_unfenced_generation_retains_claim(self):
        await self.prepared()
        for key,value in [('kernel_inventory_complete',False),('boot_rechecked',False),
                          ('native_processes',[{'pid':123}]),('reserved_sockets',['active']),
                          ('listeners_closed',[False]*len(self.pin.prepared.listeners)),
                          ('fenced_generation',self.pin.binding.generation),('fenced_generation',True),
                          ('exit_code',0),('observed_in_use_bytes',0)]:
            with self.subTest(key=key):
                data=deepcopy(self.value);data['observation'][key]=value
                with self.assertRaises(BindingRefused):validate_reboot_cleanup(canonical(data),self.pin)

    async def test_changed_member_preparation_or_observation_before_boot_is_refused(self):
        await self.prepared()
        for mutate in (lambda d:d.update(lease_id='foreign'),
                       lambda d:d['observation']['prepared'].update(attempt_sha256='a'*64),
                       lambda d:d['observation'].update(observed_at='1970-01-01T00:00:00+00:00')):
            data=deepcopy(self.value);mutate(data)
            with self.assertRaises(BindingRefused):validate_reboot_cleanup(canonical(data),self.pin)

    async def test_boot_change_during_read_never_returns_historical_snapshot(self):
        values=iter(['11111111-2222-4333-8444-555555555555','22222222-2222-4222-8222-222222222222'])
        with self.assertRaises(ReceiptUnavailable):collect_historical_attempt(self.spec,boot_reader=lambda:next(values))

    async def test_actual_boot_pin_mismatch_is_refused(self):
        await self.prepared()
        with self.assertRaises(BindingRefused):
            validate_reboot_cleanup(canonical(self.value),self.pin,observed_boot_id=self.boot)
