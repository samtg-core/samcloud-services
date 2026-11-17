"""Real SQLite lock tests; gateway signals/listener probes are mocked offline."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

p=Path(__file__).with_name('exo-pf-service-fence.py')
spec=importlib.util.spec_from_file_location('service_fence_tests',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.db=Path(self.tmp.name)/'queue.sqlite';c=sqlite3.connect(self.db)
        c.executescript('CREATE TABLE executions(state TEXT);CREATE TABLE grants(state TEXT,active_execution TEXT,idle_since REAL);');c.close()
        st=self.db.stat();self.value={'uid':os.getuid(),'fence':{'pid':777,'identity':'exact owned gateway',
            'journal':str(self.db),'journal_inode':[st.st_dev,st.st_ino]}}
        self.signals=[]

    def lock_probe(self):
        code='''import sqlite3,sys
c=sqlite3.connect(sys.argv[1],timeout=0)
try:
 c.execute('BEGIN IMMEDIATE')
 print('unlocked')
except sqlite3.OperationalError as e:
 assert 'locked' in str(e)
 print('locked')
finally:c.close()
'''
        return subprocess.check_output([sys.executable,'-I','-S','-B','-c',code,str(self.db)],text=True).strip()

    def execute(self,reply=b'release\n',during=None):
        def read(_):
            if during:during()
            return reply
        stdin=SimpleNamespace(buffer=SimpleNamespace(readline=read))
        output=io.StringIO()
        with patch.object(m,'identity',return_value='exact owned gateway'), \
             patch.object(m.os,'kill',side_effect=lambda pid,sig:self.signals.append((pid,sig))), \
             patch.object(m.subprocess,'check_output',return_value='T'), \
             patch.object(m.sys,'stdin',stdin),contextlib.redirect_stdout(output):
            m.run(self.value)
        return [json.loads(x) for x in output.getvalue().splitlines()]

    def test_child_holds_real_cross_process_writer_lock_until_release(self):
        self.assertEqual(self.lock_probe(),'unlocked')
        # The patch on m.subprocess applies to the shared module, so use Popen
        # for the actual independent kernel-lock probe during the fence.
        def probe():
            code="import sqlite3,sys;c=sqlite3.connect(sys.argv[1],timeout=0)\ntry:c.execute('BEGIN IMMEDIATE');print('unlocked')\nexcept sqlite3.OperationalError:print('locked')\nfinally:c.close()"
            r=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(self.db)],capture_output=True,text=True)
            self.assertEqual(r.stdout.strip(),'locked')
        rows=self.execute(during=probe)
        self.assertEqual([x['state'] for x in rows],['frozen','resumed'])
        self.assertEqual(self.signals,[(777,signal.SIGSTOP),(777,signal.SIGCONT)])
        self.assertEqual(self.lock_probe(),'unlocked')

    def test_controller_eof_resumes_and_releases_lock(self):
        rows=self.execute(reply=b'')
        self.assertEqual(rows[-1]['state'],'resumed')
        self.assertEqual(self.signals[-1],(777,signal.SIGCONT))
        self.assertEqual(self.lock_probe(),'unlocked')

    def test_queued_running_uncertain_refuse_before_signalling(self):
        for state in ('queued','running','uncertain'):
            with self.subTest(state=state):
                c=sqlite3.connect(self.db);c.execute('DELETE FROM executions');c.execute('INSERT INTO executions VALUES(?)',(state,));c.commit();c.close()
                with self.assertRaisesRegex(ValueError,'not drained'):self.execute()
                self.assertEqual(self.signals,[])
        self.assertEqual(self.lock_probe(),'unlocked')

    def test_changed_journal_and_uid_refuse_before_signalling(self):
        self.value['fence']['journal_inode'][1]+=1
        with self.assertRaisesRegex(ValueError,'journal changed'):self.execute()
        self.value['uid']+=1
        with self.assertRaisesRegex(ValueError,'actual service user'):self.execute()
        self.assertEqual(self.signals,[])

    def test_bad_parent_command_still_resumes(self):
        with self.assertRaisesRegex(ValueError,'only release'):self.execute(reply=b'other\n')
        self.assertEqual(self.signals[-1],(777,signal.SIGCONT))
        self.assertEqual(self.lock_probe(),'unlocked')

    def test_live_stint_refuses_before_signalling(self):
        c=sqlite3.connect(self.db);c.execute('INSERT INTO grants VALUES(?,?,?)',('granted',None,None));c.commit();c.close()
        with self.assertRaisesRegex(ValueError,'turn is still active'):self.execute()
        self.assertEqual(self.signals,[])

if __name__=='__main__':unittest.main()
