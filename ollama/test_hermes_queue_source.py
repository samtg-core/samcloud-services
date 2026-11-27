"""Private compaction intent ownership, not prompt-equality deduplication."""
import json
from pathlib import Path
import tempfile
import unittest

from .hermes_queue_identity import HostBindings
from .hermes_queue_source import CompactionSource


class CompactionSources(unittest.TestCase):
    def test_snapshot_is_private_frozen_and_explicitly_reopenable(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'sources'
            messages = [{'role':'user','content':'private synthetic input'}]
            source = CompactionSource.capture(directory,'profile','session',messages)
            first = source.bind(Path(tmp)/'identity.sqlite',{'model':'qwen'})
            messages[0]['content']='newer input'
            self.assertEqual(source.messages()[0]['content'],'private synthetic input')
            restored = CompactionSource.load(directory,'profile','session',source.operation)
            self.assertEqual(restored.bind(Path(tmp)/'identity.sqlite',{'model':'qwen'}),first)
            another = CompactionSource.capture(directory,'profile','session',source.messages())
            self.assertNotEqual(another.operation,source.operation)
            self.assertNotEqual(another.bind(Path(tmp)/'identity.sqlite',{'model':'qwen'}),first)
            self.assertEqual((directory/(source.operation+'.json')).stat().st_mode & 0o777,0o600)
            self.assertEqual(directory.stat().st_mode & 0o777,0o700)

    def test_owner_id_and_snapshot_are_checked_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'sources'
            source=CompactionSource.capture(directory,'profile','session',[{'role':'user','content':'old'}])
            for profile,session,operation in [('sibling','session',source.operation),
                    ('profile','other-session',source.operation),('profile','session','../latest')]:
                with self.assertRaises(ValueError):
                    CompactionSource.load(directory,profile,session,operation)
            with self.assertRaises(ValueError):
                source.validate([{'role':'user','content':'different work'}])
            path=directory/(source.operation+'.json')
            row=json.loads(path.read_text())
            row['messages'][0]['content']='changed on disk'
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                source.bind(Path(tmp)/'identity.sqlite',{'model':'qwen'})

    def test_host_resume_requires_exact_operation_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'identity.sqlite'
            host=HostBindings(path,'profile')
            messages=[{'role':'user','content':'not yet a native row'}]
            with host.compression_scope('session',messages,persist_unstored=True) as source:
                operation=source.operation
                keys=host.auxiliary_keys({'model':'qwen'})
            restored=HostBindings(path,'profile')
            with restored.resume_compression('session',operation) as held:
                with restored.compression_scope('session',held,persist_unstored=True):
                    self.assertEqual(restored.auxiliary_keys({'model':'qwen'}),keys)
                with self.assertRaises(ValueError):
                    with restored.compression_scope('session',[{'role':'user','content':'new input'}],persist_unstored=True):
                        pass
            with self.assertRaises(ValueError):
                restored.auxiliary_keys({'model':'qwen'})
            with self.assertRaises(ValueError):
                with restored.compression_scope('session',messages):
                    pass  # unpersisted capture remains an explicit compaction-only opt-in


if __name__=='__main__':
    unittest.main()
