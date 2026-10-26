"""Owned vision shutdown with fake children; never starts a GPU backend."""
import subprocess
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import manager as module
from .manager import ModelManager, Backend
from .vision_lifecycle import stop_owned_vision, recover_owned_vision


class VisionLifecycle(unittest.TestCase):
    def test_dead_child_reload_keeps_claim_and_records_replacement_before_health(self):
        mm = self.model()
        mm.vlm_process.poll.side_effect = None
        mm.vlm_process.poll.return_value = 1
        replacement = Mock(pid=2345, args=mm.vlm_process.args)
        replacement.poll.return_value = None
        admission = Mock()
        def ready(*args, **kwargs):
            self.assertIs(mm.vlm_process, replacement)
            self.assertEqual(mm.lease_id, 'exact')
            return SimpleNamespace(status_code=200)
        with patch('ollama.inference_shared_resource.cold_vision_absent', return_value=True), \
             patch('ollama.vision_lifecycle.subprocess.Popen', return_value=replacement), \
             patch('httpx.get', side_effect=ready):
            recover_owned_vision(mm, confirm_admission=admission)
        self.assertEqual(admission.call_count, 2)
        replacement.terminate.assert_not_called()

    def test_recovery_observation_loss_resumes_same_child_without_new_spawn(self):
        mm = self.model()
        mm.vlm_process.poll.side_effect = None
        mm.vlm_process.poll.return_value = 1
        replacement = Mock(pid=2345, args=mm.vlm_process.args)
        replacement.poll.return_value = None
        admission = Mock(side_effect=[None, RuntimeError('lease observation unavailable')])
        with patch('ollama.inference_shared_resource.cold_vision_absent', return_value=True), \
             patch('ollama.vision_lifecycle.subprocess.Popen', return_value=replacement) as spawn, \
             patch('httpx.get', return_value=SimpleNamespace(status_code=200)):
            with self.assertRaises(RuntimeError):
                recover_owned_vision(mm, confirm_admission=admission)
            with patch('httpx.get', return_value=SimpleNamespace(status_code=200)):
                recover_owned_vision(mm, confirm_admission=Mock(), resume=True)
            spawn.assert_called_once()
        self.assertIs(mm.vlm_process, replacement)
        self.assertEqual(mm.lease_id, 'exact')

    def test_unknown_absence_or_admission_never_starts_replacement(self):
        for absent in (False, True):
            mm = self.model()
            mm.vlm_process.poll.side_effect = None
            mm.vlm_process.poll.return_value = 1
            with patch('ollama.inference_shared_resource.cold_vision_absent', return_value=absent), \
                 patch('ollama.vision_lifecycle.subprocess.Popen') as spawn:
                with self.assertRaises(RuntimeError):
                    recover_owned_vision(mm, confirm_admission=Mock(side_effect=RuntimeError('not admitted')))
                spawn.assert_not_called()
            self.assertEqual(mm.lease_id, 'exact')

    def model(self):
        process = Mock(pid=1234)
        process.args = ['python', '-m', 'mlx_vlm.server', '--model', 'vision',
                        '--port', str(module.VLM_PORT), '--host', module.VLM_HOST]
        process.poll.side_effect = [None, 0]
        return SimpleNamespace(name='vision', managed=True, vlm_process=process,
            port=module.VLM_PORT, backend=Backend.VLM, in_flight=0, lease_id='exact',
            memory_mb=5700, request_count=1, loaded_at=0)

    def test_owned_child_is_reaped_before_absence_observation(self):
        mm = self.model()
        def absent():
            mm.vlm_process.wait.assert_called_once_with(timeout=10)
            return True
        with patch('ollama.inference_shared_resource.cold_vision_absent', side_effect=absent):
            stop_owned_vision(mm)
        mm.vlm_process.terminate.assert_called_once()

    def test_live_or_unobservable_child_cannot_authorize_release(self):
        for failure in ('wait', 'inventory'):
            with self.subTest(failure=failure):
                mm = self.model()
                if failure == 'wait':
                    mm.vlm_process.wait.side_effect = subprocess.TimeoutExpired('owned',10)
                with patch('ollama.inference_shared_resource.cold_vision_absent', return_value=False):
                    with self.assertRaises(RuntimeError):
                        stop_owned_vision(mm)

    def test_adopted_or_wrong_process_is_not_signalled(self):
        for field in ('managed','args'):
            mm = self.model()
            if field == 'managed': mm.managed = False
            else: mm.vlm_process.args = ['python','unrelated.py']
            with self.assertRaises(RuntimeError): stop_owned_vision(mm)
            mm.vlm_process.terminate.assert_not_called()

    def test_manager_retains_exact_link_until_confirmed_stop(self):
        mm = self.model()
        m = object.__new__(ModelManager)
        m.models, m.durable_inference = {'vision':mm}, True
        m._release_lease_quietly = Mock()
        with patch('ollama.vision_lifecycle.stop_owned_vision', side_effect=RuntimeError('unknown')):
            with self.assertRaises(RuntimeError): m.unload('vision')
        self.assertIs(m.models['vision'],mm)
        m._release_lease_quietly.assert_not_called()
        with patch('ollama.vision_lifecycle.stop_owned_vision'):
            self.assertEqual(m.unload('vision')['status'],'unloaded')
        m._release_lease_quietly.assert_called_once_with('exact')
        self.assertNotIn('vision',m.models)

    def test_in_flight_work_is_never_terminated(self):
        mm = self.model();mm.in_flight=1
        m = object.__new__(ModelManager)
        m.models, m.durable_inference = {'vision':mm}, True
        with self.assertRaises(RuntimeError): m.unload('vision', force=True)
        mm.vlm_process.terminate.assert_not_called()

    def test_real_owned_child_exit_reap_and_closed_socket(self):
        """Real OS lifecycle, fake backend: no MLX import, model or GPU."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); package = root/'mlx_vlm';package.mkdir()
            (package/'__init__.py').write_text('')
            (package/'server.py').write_text(
                "import argparse,socket\n"
                "p=argparse.ArgumentParser();p.add_argument('--model');"
                "p.add_argument('--port',type=int);p.add_argument('--host');a=p.parse_args()\n"
                "s=socket.socket();s.bind((a.host,a.port));s.listen()\n"
                "while True:\n c,_=s.accept();c.close()\n")
            with socket.socket() as reserve:
                reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
            args=[sys.executable,'-m','mlx_vlm.server','--model','vision',
                  '--port',str(port),'--host','127.0.0.1']
            child=subprocess.Popen(args,env=dict(os.environ,PYTHONPATH=str(root)),
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                for _ in range(200):
                    if child.poll() is not None:self.fail('fake child exited before ready')
                    with socket.socket() as probe:
                        if probe.connect_ex(('127.0.0.1',port))==0:break
                    time.sleep(.01)
                else:self.fail('fake child failed to listen')
                mm=SimpleNamespace(name='vision',managed=True,vlm_process=child,port=port)
                with patch('ollama.config.VLM_PORT',port), \
                     patch('ollama.config.VLM_HOST','127.0.0.1'):
                    stop_owned_vision(mm)
                self.assertIsNotNone(child.returncode)
                with socket.socket() as probe:
                    self.assertNotEqual(probe.connect_ex(('127.0.0.1',port)),0)
            finally:
                if child.poll() is None:child.terminate()
                child.wait(timeout=5)

    def test_owned_load_survives_legacy_startup_age(self):
        m = object.__new__(ModelManager)
        m.models, m.durable_inference = {}, True
        m.refuse_if_device_in_use = Mock()
        m._kill_stray_vlm = Mock()
        m._request_lease = Mock(return_value='exact')
        process = Mock();process.poll.return_value=None
        replies = [SimpleNamespace(status_code=n) for n in (503,503,503,200)]
        with patch.object(module,'match_vlm_model',return_value=('vision',5700)), \
             patch.object(module,'VLM_STARTUP_TIMEOUT',1), \
             patch('ollama.inference_shared_resource.cold_vision_absent',return_value=True), \
             patch.object(module.subprocess,'Popen',return_value=process), \
             patch('httpx.get',side_effect=replies) as health, \
             patch.object(module.time,'sleep'):
            model=m.load_vlm_model('vision')
        self.assertEqual(health.call_count,4)
        self.assertEqual(model.lease_id,'exact')
        m._request_lease.assert_called_once_with('vision',5700,before_load=True)
        m._kill_stray_vlm.assert_not_called()
        process.terminate.assert_not_called()

    def test_unknown_existing_backend_is_not_killed_or_bought_again(self):
        m = object.__new__(ModelManager)
        m.models, m.durable_inference = {}, True
        m.refuse_if_device_in_use = Mock()
        m._kill_stray_vlm = Mock()
        m._request_lease = Mock()
        with patch.object(module,'match_vlm_model',return_value=('vision',5700)), \
             patch('ollama.inference_shared_resource.cold_vision_absent',return_value=False):
            with self.assertRaises(RuntimeError):m.load_vlm_model('vision')
        m._kill_stray_vlm.assert_not_called()
        m._request_lease.assert_not_called()

    def test_dead_child_defers_before_dispatch_without_silent_reload(self):
        from .inference_arbiter import CapacityDeferred
        mm=self.model();mm.vlm_process.poll.side_effect=None
        mm.vlm_process.poll.return_value=1
        m=object.__new__(ModelManager)
        m.models,m.durable_inference={'vision':mm},True
        m.load_vlm_model=Mock()
        with self.assertRaises(CapacityDeferred):m.ensure_running('vision')
        m.load_vlm_model.assert_not_called()
        self.assertIs(m.models['vision'],mm)


if __name__ == '__main__':
    unittest.main()
