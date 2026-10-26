"""Launcher selection without a listener, provider call or fleet mutation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import server
from .inference_launch import main


class Launch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.path = root/'config.json'
        self.path.write_text(json.dumps({'journal':str(root/'queue.sqlite'),
            'models':{'qwen':'gpu'},'reservation_mb':{'qwen':100}}))
        for name in ('inference_startup','inference_runtime','inference_dispatcher'):
            self.enterContext(patch.object(server,name,None))
        self.serve = self.enterContext(patch('uvicorn.run'))

    def test_explicit_launcher_binds_before_serving_single_process(self):
        main(['--queue-config',str(self.path),'--work-permission','allow','--port','8800'])
        self.assertTrue(server.inference_startup.work_permitted)
        self.assertEqual(server.inference_startup.settings.models,{'qwen':'gpu'})
        self.serve.assert_called_once_with(server.app,host='127.0.0.1',port=8800,workers=1)
        self.assertFalse((self.path.parent/'queue.sqlite').exists())

    def test_denied_permission_is_not_inferred_from_availability(self):
        main(['--queue-config',str(self.path),'--work-permission','deny'])
        self.assertFalse(server.inference_startup.work_permitted)

    def test_missing_permission_refuses_listener(self):
        with self.assertRaises(SystemExit):
            main(['--queue-config',str(self.path)])
        self.serve.assert_not_called()

    def test_invalid_configuration_refuses_listener_and_binding(self):
        self.path.write_text('{"PRIVATE":"invalid"}')
        with self.assertRaisesRegex(ValueError,'Invalid queue startup configuration'):
            main(['--queue-config',str(self.path),'--work-permission','allow'])
        self.assertIsNone(server.inference_startup)
        self.serve.assert_not_called()

    def test_existing_owner_is_never_hot_converted(self):
        server.inference_runtime = object()
        with self.assertRaisesRegex(RuntimeError,'fresh service process'):
            main(['--queue-config',str(self.path),'--work-permission','allow'])
        self.serve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
