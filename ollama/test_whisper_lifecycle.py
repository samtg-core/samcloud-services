"""Drained Whisper cleanup controls; fake children/audio, no GPU or network API."""
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import config, manager, whisper_lifecycle as lifecycle
from .manager import Backend, ModelManager
from .inference_shared_resource import ResourceHandoff


class WhisperLifecycle(unittest.TestCase):
    def model(self):
        process = Mock(pid=1234)
        process.args = [manager.WHISPER_PYTHON, manager.WHISPER_SCRIPT,
            '--model', manager.WHISPER_MODELS[manager.WHISPER_DEFAULT]['repo'],
            '--host', config.WHISPER_HOST, '--port', str(config.WHISPER_PORT),
            '--spool', str(config.WHISPER_SPOOL_DIR), '--ffmpeg', config.FFMPEG_BIN]
        process.poll.side_effect = [None, 0]
        return SimpleNamespace(name=manager.WHISPER_DEFAULT, managed=True, whisper_process=process,
            port=config.WHISPER_PORT, backend=Backend.WHISPER, in_flight=0, lease_id='exact',
            memory_mb=2600, request_count=1, loaded_at=0)

    def test_owned_idle_child_is_reaped_before_absence_check(self):
        mm = self.model()
        def absent():
            mm.whisper_process.wait.assert_called_once_with(timeout=2)
            return True
        with patch.object(lifecycle, 'cold_whisper_absent', side_effect=absent):
            lifecycle.stop_owned_whisper(mm, observation_seconds=2)
        mm.whisper_process.terminate.assert_called_once()
        mm.whisper_process.kill.assert_not_called()

    def test_unknown_exit_or_absence_never_escalates(self):
        for failure in ('wait', 'inventory', 'still-live'):
            with self.subTest(failure=failure):
                mm = self.model()
                if failure == 'wait':
                    mm.whisper_process.wait.side_effect = subprocess.TimeoutExpired('owned',2)
                if failure == 'still-live':
                    mm.whisper_process.poll.side_effect = [None, None]
                with patch.object(lifecycle, 'cold_whisper_absent', return_value=failure != 'inventory'):
                    with self.assertRaises(RuntimeError):
                        lifecycle.stop_owned_whisper(mm, observation_seconds=2)
                mm.whisper_process.kill.assert_not_called()

    def test_adopted_wrong_child_and_active_work_are_not_signalled(self):
        for change in ('managed','args','port','active','model'):
            with self.subTest(change=change):
                mm = self.model()
                if change == 'managed':mm.managed = False
                elif change == 'args':mm.whisper_process.args = ['python','unrelated.py']
                elif change == 'port':mm.port += 1
                elif change == 'active':mm.in_flight = 1
                else:mm.name = 'not-a-whisper-model'
                with self.assertRaises(RuntimeError):
                    lifecycle.stop_owned_whisper(mm, observation_seconds=2)
                mm.whisper_process.terminate.assert_not_called()

    def test_observation_bound_must_be_explicit_finite_and_positive(self):
        for value in (None, True, 0, -1, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                lifecycle.stop_owned_whisper(self.model(), observation_seconds=value)

    def test_manager_keeps_model_and_exact_claim_after_unknown_cleanup(self):
        mm = self.model()
        m = object.__new__(ModelManager)
        m.models, m.durable_inference = {mm.name:mm}, True
        m._release_lease_quietly = Mock()
        with patch.object(lifecycle, 'stop_owned_whisper', side_effect=RuntimeError('unknown')):
            with self.assertRaises(RuntimeError):m.unload(mm.name,require_confirmed=True)
        self.assertIs(m.models[mm.name],mm)
        m._release_lease_quietly.assert_not_called()
        with patch.object(lifecycle, 'stop_owned_whisper'):
            self.assertEqual(m.unload(mm.name,require_confirmed=True)['status'],'unloaded')
        m._release_lease_quietly.assert_called_once_with('exact')
        self.assertNotIn(mm.name,m.models)

    def test_force_cannot_stop_a_transcription_in_flight(self):
        mm = self.model();mm.in_flight = 1
        m = object.__new__(ModelManager)
        m.models,m.durable_inference = {mm.name:mm},True
        with self.assertRaises(RuntimeError):m.unload(mm.name,force=True)
        mm.whisper_process.terminate.assert_not_called()

    def test_idle_counter_or_force_alone_cannot_stop_unknown_native_work(self):
        mm = self.model()
        m = object.__new__(ModelManager)
        m.models,m.durable_inference = {mm.name:mm},True
        m._release_lease_quietly = Mock()
        for force in (False,True):
            with self.subTest(force=force), self.assertRaisesRegex(RuntimeError,'queue handoff'):
                m.unload(mm.name,force=force)
        mm.whisper_process.terminate.assert_not_called()
        m._release_lease_quietly.assert_not_called()

    def test_handoff_uses_whisper_absence_instead_of_empty_ollama(self):
        mm = self.model()
        m = SimpleNamespace(models={},ollama=SimpleNamespace(list_running=Mock(return_value=[])))
        actor = object.__new__(ResourceHandoff);actor.runtime = SimpleNamespace(manager=m)
        with patch.object(lifecycle, 'cold_whisper_absent', return_value=False):
            self.assertFalse(actor._absent(mm.name))
        with patch.object(lifecycle, 'cold_whisper_absent', return_value=True):
            self.assertTrue(actor._absent(mm.name))
        m.ollama.list_running.assert_not_called()

    def test_initial_retirement_uses_the_same_backend_absence(self):
        m = object.__new__(ModelManager)
        m.models,m.durable_inference = {},True
        m.ollama = SimpleNamespace(list_running=Mock(return_value=[]))
        m.initial_reservations = SimpleNamespace(reconcile_absent=Mock(
            side_effect=lambda name,confirm_absent:confirm_absent(name)))
        with patch.object(lifecycle,'cold_whisper_absent',return_value=False):
            self.assertFalse(m.reconcile_initial_reservation(manager.WHISPER_DEFAULT))
        m.ollama.list_running.assert_not_called()

    def test_cold_check_needs_both_inventory_and_refused_port(self):
        for output, code, port, expected in (
            ('/usr/bin/python /service/whisper_server.py --model x\n',0,True,False),
            ('/usr/bin/python unrelated.py\n',0,False,False),
            ('/usr/bin/python unrelated.py\n',0,True,True),
            ('',0,True,False), ('inventory failure',1,True,False)):
            with self.subTest(output=output,code=code,port=port), \
                 patch.object(lifecycle.subprocess,'run',return_value=SimpleNamespace(
                     stdout=output,returncode=code)), \
                 patch('ollama.inference_bootstrap.vision_backend_absent',return_value=port):
                self.assertEqual(lifecycle.cold_whisper_absent(),expected)

    def test_failed_inventory_stays_unknown(self):
        for error in (OSError('unreadable'),subprocess.TimeoutExpired('ps',5)):
            with self.subTest(error=error), patch.object(lifecycle.subprocess,'run',side_effect=error):
                self.assertFalse(lifecycle.cold_whisper_absent())

    def test_real_owned_child_exit_reap_and_closed_socket(self):
        """Real process/socket lifecycle, no transcription or model import."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);script = root/'whisper_server.py'
            script.write_text('import argparse,socket\n'
                'p=argparse.ArgumentParser();p.add_argument("--model");p.add_argument("--host");'
                'p.add_argument("--port",type=int);p.add_argument("--spool");p.add_argument("--ffmpeg");a=p.parse_args()\n'
                's=socket.socket();s.bind((a.host,a.port));s.listen()\n'
                'while True:\n c,_=s.accept();c.close()\n')
            with socket.socket() as reserve:
                reserve.bind(('127.0.0.1',0));port = reserve.getsockname()[1]
            args = [sys.executable,str(script),'--model',manager.WHISPER_MODELS[manager.WHISPER_DEFAULT]['repo'],
                '--host','127.0.0.1','--port',str(port),'--spool',str(root),'--ffmpeg','/unused/ffmpeg']
            child = subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                for _ in range(200):
                    if child.poll() is not None:self.fail('fake child exited before ready')
                    with socket.socket() as probe:
                        if probe.connect_ex(('127.0.0.1',port)) == 0:break
                    time.sleep(.01)
                else:self.fail('fake child failed to listen')
                mm = SimpleNamespace(name=manager.WHISPER_DEFAULT,managed=True,
                    in_flight=0,whisper_process=child,port=port)
                with patch.object(manager,'WHISPER_PYTHON',sys.executable), \
                     patch.object(manager,'WHISPER_SCRIPT',str(script)), \
                     patch.object(config,'WHISPER_HOST','127.0.0.1'),patch.object(config,'WHISPER_PORT',port), \
                     patch.object(config,'WHISPER_SPOOL_DIR',root),patch.object(config,'FFMPEG_BIN','/unused/ffmpeg'):
                    lifecycle.stop_owned_whisper(mm,observation_seconds=2)
                self.assertIsNotNone(child.returncode)
                with socket.socket() as probe:self.assertNotEqual(probe.connect_ex(('127.0.0.1',port)),0)
            finally:
                if child.poll() is None:child.terminate()
                child.wait(timeout=5)


if __name__ == '__main__':unittest.main()
