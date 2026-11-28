"""Durable queue inputs: synthetic bytes only, no audio/provider or live files."""
import asyncio
import hashlib
from pathlib import Path
import tempfile
import unittest

from .inference_arbiter import Arbiter, NotFound
from .inference_inputs import InferenceInputs


class Upload:
    filename = '../../not-a-destination'
    def __init__(self, *chunks):
        self.chunks = iter(chunks)
    async def read(self, limit):
        assert limit==1024*1024
        return next(self.chunks,b'')


class QueueInputs(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'inputs'
        self.store = InferenceInputs(self.root,100)
        self.journal = Arbiter(Path(self.temp.name)/'jobs.sqlite')

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    async def installed(self, key='one', data=b'synthetic audio'):
        grant = self.journal.submit('creative',key,'gpu','whisper')
        staged = await self.store.capture(Upload(data[:3],data[3:]))
        descriptor = await asyncio.to_thread(self.store.install,grant['id'],staged)
        return grant, descriptor

    def enqueue(self, grant, descriptor, key='call'):
        return self.journal.enqueue(grant['id'],'creative',key,
            {'operation':'transcription','body':{'input':descriptor}})

    async def test_capture_install_and_reopen_are_private_and_integrity_checked(self):
        grant, descriptor = await self.installed()
        self.assertEqual(descriptor,{'sha256':hashlib.sha256(b'synthetic audio').hexdigest(),'bytes':15})
        reopened = InferenceInputs(self.root,100)
        path = reopened.path(grant['id'],descriptor)
        self.assertEqual(path.read_bytes(),b'synthetic audio')
        self.assertEqual(path.stat().st_mode&0o777,0o600)
        self.assertEqual(path.parent.stat().st_mode&0o777,0o700)
        self.assertFalse(list(self.root.glob('.pending-*')))

    async def test_equal_bytes_under_distinct_intents_have_separate_lifetimes(self):
        first, descriptor = await self.installed('first')
        second, other = await self.installed('second')
        self.assertEqual(descriptor,other)
        self.assertNotEqual(self.store.path(first['id'],descriptor),self.store.path(second['id'],other))
        call = self.enqueue(first,descriptor)
        self.journal.dispatch_next('gpu')
        self.journal.finish(call['id'],{'text':'done'})
        self.assertTrue(self.store.retire(self.journal,call['id'],'creative'))
        self.assertEqual(self.store.path(second['id'],other).read_bytes(),b'synthetic audio')

    async def test_same_intent_reupload_never_overwrites_input(self):
        grant, descriptor = await self.installed()
        path = self.store.path(grant['id'],descriptor)
        inode = path.stat().st_ino
        staged = await self.store.capture(Upload(b'synthetic audio'))
        self.assertEqual(self.store.install(grant['id'],staged),descriptor)
        self.assertEqual(path.stat().st_ino,inode)

    async def test_failed_or_canceled_capture_leaves_no_unaccepted_copy(self):
        for chunks in ((), (b'x'*101,), ('not bytes',)):
            with self.subTest(chunks_type=type(chunks[0]).__name__ if chunks else 'empty'):
                with self.assertRaises(ValueError):
                    await self.store.capture(Upload(*chunks))
                self.assertEqual(list(self.root.iterdir()),[])
        class Canceled:
            async def read(self, limit):
                raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.store.capture(Canceled())
        self.assertEqual(list(self.root.iterdir()),[])

    async def test_tampering_is_rejected_and_reupload_does_not_replace_it(self):
        grant, descriptor = await self.installed()
        path = self.store.path(grant['id'],descriptor)
        path.write_bytes(b'X'*descriptor['bytes'])
        with self.assertRaisesRegex(ValueError,'integrity'):
            self.store.path(grant['id'],descriptor)
        staged = await self.store.capture(Upload(b'synthetic audio'))
        with self.assertRaisesRegex(ValueError,'integrity'):
            self.store.install(grant['id'],staged)
        self.assertEqual(path.read_bytes(),b'X'*descriptor['bytes'])

    async def test_discard_staged_is_idempotent_and_never_discards_installed_input(self):
        staged=await self.store.capture(Upload(b'synthetic audio'))
        self.assertTrue(await asyncio.to_thread(self.store.discard_staged,staged))
        self.assertFalse(self.store.discard_staged(staged))
        grant=self.journal.submit('creative','installed','gpu','whisper')
        staged=await self.store.capture(Upload(b'synthetic audio'))
        descriptor=self.store.install(grant['id'],staged)
        self.assertFalse(self.store.discard_staged(staged))
        self.assertTrue(self.store.path(grant['id'],descriptor).exists())

    async def test_discard_staged_does_not_delete_changed_file(self):
        staged=await self.store.capture(Upload(b'synthetic audio'))
        staged.path.write_bytes(b'X'*staged.size)
        with self.assertRaisesRegex(ValueError,'integrity'):
            self.store.discard_staged(staged)
        self.assertTrue(staged.path.exists())

    async def test_symlink_or_loose_permissions_are_not_accepted(self):
        grant, descriptor = await self.installed()
        path = self.store.path(grant['id'],descriptor)
        path.chmod(0o644)
        with self.assertRaises(ValueError):
            self.store.path(grant['id'],descriptor)
        path.chmod(0o600)
        original = path.with_suffix('.retained')
        path.rename(original)
        path.symlink_to(original)
        with self.assertRaises(OSError):
            self.store.path(grant['id'],descriptor)
        link = Path(self.temp.name)/'linked-inputs'
        link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(PermissionError):
            InferenceInputs(link,100)

    async def test_untrusted_names_and_descriptors_cannot_escape_store(self):
        grant, descriptor = await self.installed()
        for key in ('../outside','/tmp','',True):
            with self.assertRaises(ValueError):
                self.store.path(key,descriptor)
        for value in ({**descriptor,'bytes':True},{**descriptor,'sha256':'../escape'},
                      {**descriptor,'path':'/tmp'},{**descriptor,'bytes':101}):
            with self.assertRaises(ValueError):
                self.store.path(grant['id'],value)

    async def test_queued_running_and_uncertain_inputs_are_never_retired(self):
        grant, descriptor = await self.installed()
        call = self.enqueue(grant,descriptor)
        self.assertFalse(self.store.retire(self.journal,call['id'],'creative'))
        self.journal.dispatch_next('gpu')
        self.assertFalse(self.store.retire(self.journal,call['id'],'creative'))
        self.journal.uncertain(call['id'])
        self.assertFalse(self.store.retire(self.journal,call['id'],'creative'))
        self.assertTrue(self.store.path(grant['id'],descriptor).is_file())

    async def test_saved_terminal_result_allows_only_owner_scoped_idempotent_retirement(self):
        grant, descriptor = await self.installed()
        call = self.enqueue(grant,descriptor)
        self.journal.dispatch_next('gpu')
        self.journal.finish(call['id'],{'text':'done'})
        with self.assertRaises(NotFound):
            self.store.retire(self.journal,call['id'],'sibling-identity')
        self.assertTrue(self.store.path(grant['id'],descriptor).is_file())
        self.assertTrue(self.store.retire(self.journal,call['id'],'creative'))
        self.assertFalse(self.store.retire(self.journal,call['id'],'creative'))
        self.assertFalse((self.root/grant['id']).exists())
        self.assertEqual(self.journal.execution(call['id'],'creative')['state'],'complete')

    async def test_later_call_on_same_input_keeps_it_until_that_call_finishes(self):
        grant, descriptor = await self.installed()
        first = self.enqueue(grant,descriptor,'first')
        self.journal.dispatch_next('gpu')
        self.journal.finish(first['id'],{'text':'first answer'})
        second = self.enqueue(grant,descriptor,'second')
        self.assertFalse(self.store.retire(self.journal,first['id'],'creative'))
        self.journal.dispatch_next('gpu')
        self.assertFalse(self.store.retire(self.journal,first['id'],'creative'))
        self.journal.finish(second['id'],{'error':'confirmed refusal'},failed=True)
        self.assertTrue(self.store.retire(self.journal,second['id'],'creative'))

    async def test_off_loop_proof_cannot_delete_a_changed_copy(self):
        grant, descriptor = await self.installed()
        call = self.enqueue(grant,descriptor)
        self.journal.dispatch_next('gpu')
        self.journal.finish(call['id'],{'text':'done'})
        proof = await asyncio.to_thread(self.store.proof,grant['id'],descriptor)
        proof.path.write_bytes(b'X'*descriptor['bytes'])
        self.assertFalse(self.store.retire(self.journal,call['id'],'creative',proof=proof))
        self.assertTrue(proof.path.is_file())

    async def test_new_reference_after_off_loop_proof_prevents_retirement(self):
        grant, descriptor = await self.installed()
        first = self.enqueue(grant,descriptor,'first')
        self.journal.dispatch_next('gpu')
        self.journal.finish(first['id'],{'text':'done'})
        proof = await asyncio.to_thread(self.store.proof,grant['id'],descriptor)
        second = self.enqueue(grant,descriptor,'second')
        self.assertFalse(self.store.retire(self.journal,first['id'],'creative',proof=proof))
        self.assertTrue(proof.path.is_file())
        self.journal.dispatch_next('gpu')
        self.journal.finish(second['id'],{'text':'also done'})
        self.assertTrue(self.store.retire(self.journal,second['id'],'creative',proof=proof))


if __name__=='__main__':
    unittest.main()
