"""Offline privilege-boundary and once-only operation controls; no actual PF."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BASE=Path(__file__).parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value

class ControlTests(unittest.TestCase):
    def setUp(self):
        self.c=module('pf_control_test',BASE/'exo-pf-control.py')
        self.u=module('pf_update_control_test',BASE/'update-exo-pf.py')
        self.u.m=module('pf_transport_control_test',BASE.parents[1]/'ollama/exo_groups/pf_transport.py')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.c.ROOT=root;self.c.CONTROL=root/'control';(self.c.CONTROL/'requests').mkdir(parents=True)
        before={'rank':'slice','uid':502,'boot_id':'3c17741b-ea8f-4430-b3b3-39b4964574d1',
                'interface':'en18','peers':['192.168.1.3','192.168.1.80','192.168.1.84'],
                'ports':[52515,52514,52513,52518]}
        after=deepcopy(before);after['peers'][1]='192.168.1.19'
        self.plan={'before_policy':before,'after_policy':after,'before_sha256':{},'pfctl_sha256':'a'*64,
                   'gateway_fence':{'pid':777,'identity':'502 exact owned gateway',
                    'journal':'/Users/fixture-service/var/queue.sqlite','journal_inode':[1,2]}}
        self.cap={'rank':'slice','uid':502,'ports':[52515,52514,52513,52518],
            'journal':self.plan['gateway_fence']['journal'],'private_networks':['192.168.1.0/24']}
        for name,raw in [('receipt.json',b'{"phase":"active"}'),('policy.json',b'old'),('rules.conf',b'old rules')]:
            (root/name).write_bytes(raw)
        self.plan['before_sha256']={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('receipt.json','policy.json','rules.conf')}
        self.value={'request_id':'a'*64,'plan':self.plan};self.calls=[]
        self.read=patch.object(self.c,'protected',side_effect=lambda path, maximum=262144:Path(path).read_bytes());self.read.start();self.addCleanup(self.read.stop)
        self.resume=patch.object(self.c,'gateway_resumed',return_value=True);self.resume.start();self.addCleanup(self.resume.stop)
        def apply(plan):
            self.calls.append(deepcopy(plan))
            (root/'receipt.json').write_bytes(self.c.canonical({'phase':'active','policy_sha256':self.u.policy(plan['after_policy']).policy_sha256}))
        self.u.apply=apply

    def test_actual_bounded_plan_is_accepted(self):
        self.assertEqual(self.c.validate_request(self.value,self.cap,self.u),self.plan)

    def test_no_code_path_or_extra_top_level_keys(self):
        for field in ('program','module','command','source_path'):
            value=deepcopy(self.value);value['plan'][field]='/tmp/payload'
            with self.subTest(field=field),self.assertRaises(ValueError):self.c.validate_request(value,self.cap,self.u)
        value=deepcopy(self.value);value['argv']=['/bin/sh']
        with self.assertRaises(ValueError):self.c.validate_request(value,self.cap,self.u)

    def test_other_uid_rank_or_port_cannot_expand_capability(self):
        for field,changed in [('uid',0),('rank','wafer'),('ports',[52515,52514,52513,22])]:
            value=deepcopy(self.value)
            for side in ('before_policy','after_policy'):value['plan'][side][field]=changed
            with self.subTest(field=field),self.assertRaises(ValueError):self.c.validate_request(value,self.cap,self.u)

    def test_foreign_journal_and_invalid_pid_refuse(self):
        for field,changed in [('journal','/private/etc/sudoers'),('pid',1),('pid',True),('journal_inode',[1,False])]:
            value=deepcopy(self.value);value['plan']['gateway_fence'][field]=changed
            with self.subTest(field=field),self.assertRaises(ValueError):self.c.validate_request(value,self.cap,self.u)

    def test_public_or_outside_approved_private_network_refuses(self):
        for peer in ('8.8.8.8','10.8.0.1'):
            value=deepcopy(self.value);value['plan']['after_policy']['peers'][1]=peer
            with self.subTest(peer=peer),self.assertRaises(ValueError):self.c.validate_request(value,self.cap,self.u)

    def test_duplicate_json_and_path_like_request_ids_refuse(self):
        with self.assertRaises(ValueError):self.c.read_object(b'{"request_id":1,"request_id":2}')
        value=deepcopy(self.value);value['request_id']='../escape'
        with self.assertRaises(ValueError):self.c.validate_request(value,self.cap,self.u)

    def test_success_reobservation_never_calls_apply_twice(self):
        first=self.c.apply_request(self.value,self.cap,self.u)
        self.assertEqual(first['state'],'verified')
        self.assertEqual(self.c.apply_request(self.value,self.cap,self.u),first)
        self.assertEqual(len(self.calls),1)

    def test_same_id_different_data_refuses_without_second_apply(self):
        self.c.apply_request(self.value,self.cap,self.u)
        value=deepcopy(self.value);value['plan']['gateway_fence']['pid']+=1
        with self.assertRaisesRegex(ValueError,'different data'):self.c.apply_request(value,self.cap,self.u)
        self.assertEqual(len(self.calls),1)

    def test_proven_premutation_refusal_can_have_a_fresh_explicit_attempt(self):
        def refuse(plan):self.calls.append(plan);raise ValueError('ordinary work has not drained')
        self.u.apply=refuse
        first=self.c.apply_request(self.value,self.cap,self.u)
        self.assertEqual(first['state'],'refused_before_policy_change')
        self.assertEqual(self.c.apply_request(self.value,self.cap,self.u),first);self.assertEqual(len(self.calls),1)
        value=deepcopy(self.value);value['request_id']='b'*64
        self.assertEqual(self.c.apply_request(value,self.cap,self.u)['state'],'refused_before_policy_change')
        self.assertEqual(len(self.calls),2)

    def test_unknown_result_blocks_new_attempts(self):
        def unknown(plan):
            (self.c.ROOT/'receipt.json').write_text('{"phase":"updating"}')
            self.calls.append(plan);raise RuntimeError('lost load reply')
        self.u.apply=unknown
        self.assertEqual(self.c.apply_request(self.value,self.cap,self.u)['state'],'unconfirmed')
        value=deepcopy(self.value);value['request_id']='b'*64
        with self.assertRaisesRegex(ValueError,'earlier operation'):self.c.apply_request(value,self.cap,self.u)
        self.assertEqual(len(self.calls),1)

    def test_missing_result_is_observed_without_reexecution(self):
        directory=self.c.CONTROL/'requests'/self.value['request_id'];directory.mkdir()
        self.c.write(directory/'intent.json',{'request_sha256':hashlib.sha256(self.c.canonical(self.value)).hexdigest()})
        self.assertEqual(self.c.apply_request(self.value,self.cap,self.u)['state'],'unconfirmed')
        self.assertEqual(self.calls,[])

    def test_unconfirmed_gateway_resume_never_becomes_a_safe_refusal(self):
        def refuse(plan):raise ValueError('refused')
        self.u.apply=refuse
        with patch.object(self.c,'gateway_resumed',return_value=False):
            self.assertEqual(self.c.apply_request(self.value,self.cap,self.u)['state'],'unconfirmed')

    def test_parent_drops_all_child_credentials_and_never_signals_as_root(self):
        self.cap.update(username='fixture-service',fence_sha256=hashlib.sha256(b'fixed fence').hexdigest())
        (self.c.CONTROL/'service-fence.py').write_bytes(b'fixed fence')
        out=self.c.ROOT/'operation';out.mkdir()
        fence=self.plan['gateway_fence']
        ready={'state':'frozen','pid':fence['pid'],'identity':fence['identity'],'uid':502,
               'queue_writer_barrier':True,'journal_inode':fence['journal_inode']}
        resumed={'state':'resumed','pid':fence['pid'],'identity':fence['identity'],'writer_barrier_released':True}
        class Child:
            def __init__(self):
                import io
                self.stdin=io.BytesIO();self.stdout=io.BytesIO(json.dumps(ready).encode()+b'\n');self.returncode=None
            def poll(self):return self.returncode
            def communicate(self):self.returncode=0;return json.dumps(resumed).encode()+b'\n',b''
        child=Child()
        with patch.object(self.c.pwd,'getpwuid',return_value=SimpleNamespace(pw_name='fixture-service',pw_gid=20)), \
             patch.object(self.c.os,'getgrouplist',return_value=[20]), \
             patch.object(self.c.subprocess,'Popen',return_value=child) as spawn, \
             patch.object(self.c.os,'kill',side_effect=AssertionError('parent must not signal')):
            with self.c.service_fence(self.plan,self.u.policy(self.plan['after_policy']),out,self.cap,self.u):
                self.assertTrue(self.u.FENCE_ALIVE())
        kwargs=spawn.call_args.kwargs
        self.assertEqual((kwargs['user'],kwargs['group'],kwargs['extra_groups']),(502,20,[20]))
        self.assertEqual(kwargs['cwd'],'/')
        self.assertEqual(spawn.call_args.args[0][:4],['/usr/bin/python3','-I','-S','-B'])
        self.assertTrue((out/'gateway-resumed.json').exists())

    def test_changed_installed_code_refuses_before_execution(self):
        with patch.object(self.c,'protected',return_value=b'raise AssertionError("must not execute")'):
            with self.assertRaisesRegex(ValueError,'module changed'):
                self.c.load_module('never_loaded','update.py','0'*64)

    def test_unprivileged_parent_cannot_supply_protected_code(self):
        # Exercise the actual reader, not the test fixture's permissive file read.
        self.read.stop()
        if os.getuid()==0:self.skipTest('requires unprivileged test runner')
        with self.assertRaisesRegex(ValueError,'root-owned'):self.c.protected(self.c.ROOT/'receipt.json')
        self.read.start()

class InstallerTests(unittest.TestCase):
    def test_actual_payload_reader_checks_bytes_and_refuses_symlinks(self):
        i=module('pf_control_payload_test',BASE/'install-exo-pf-control.py')
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);(directory/'payload').write_bytes(b'reviewed code')
            with patch.object(i,'__file__',str(directory/'install.py')):
                self.assertEqual(i.read_payload('payload',hashlib.sha256(b'reviewed code').hexdigest()),b'reviewed code')
                with self.assertRaises(ValueError):i.read_payload('payload','0'*64)
                (directory/'redirect').symlink_to('payload')
                with self.assertRaises(OSError):i.read_payload('redirect',hashlib.sha256(b'reviewed code').hexdigest())

    def test_sudoers_has_only_two_fixed_digest_bound_commands(self):
        i=module('pf_control_installer_test',BASE/'install-exo-pf-control.py')
        raw=i.sudoers_text('claude-services',i.ROOT/'control/run','a'*64).decode()
        rows=[r for r in raw.splitlines() if not r.startswith('#')]
        self.assertEqual(len(rows),2)
        self.assertTrue(rows[0].endswith('/control/run --apply'))
        self.assertTrue(rows[1].endswith('/control/run --status'))
        self.assertNotIn('*',raw);self.assertNotIn('/bin/sh',raw);self.assertNotIn('/usr/bin/python',raw)
        self.assertTrue(all('NOSETENV: sha256:'+'a'*64 in row for row in rows))
        for user in ('root\nALL','claude-*'):
            with self.assertRaises(ValueError):i.sudoers_text(user,i.ROOT/'control/run','a'*64)

if __name__=='__main__':unittest.main()
